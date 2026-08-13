import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from script.slider_quant import SliderQuantLinear

def apply_fake_quantization_to_module(module, bits=4, group_size=128):
    """
    Dynamically replaces the Linear layers of a module with the SliderQuantLinear class.
    Returns a dictionary containing the original layers so they can be restored.
    """
    original_linears = {}
    
    def replace_linears(m, prefix=""):
        for name, child in m.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if any(skip in name for skip in ["norm1", "emb"]):
                continue
                
            if isinstance(child, nn.Linear):
                original_linears[full_name] = child
                
                sq_linear = SliderQuantLinear(
                    original_linear=child, 
                    weight_bits=bits, 
                    act_bits=bits, 
                    rank=0, 
                    gamma=1.0, 
                    group_size=group_size
                )
                setattr(m, name, sq_linear)
            else:
                replace_linears(child, full_name)
                
    replace_linears(module)
    return original_linears

def restore_original_weights(module, original_linears):
    """
    Restores the original Linear layers that were previously replaced.
    """
    def restore_linears(m, prefix=""):
        for name, child in m.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in original_linears:
                setattr(m, name, original_linears[full_name])
            else:
                restore_linears(child, full_name)
                
    restore_linears(module)

try:
    from torchmetrics.functional import signal_noise_ratio
except ImportError:
    raise ImportError("torchmetrics is required: pip install torchmetrics")

def compute_snr_divergence(out_baseline, out_quantized):
    """
    Calculates the relative error using torchmetrics' signal_noise_ratio.
    Transforms the SNR (dB) into its linear inverse (Noise-to-Signal Ratio).
    A higher value indicates greater sensitivity to quantization noise.
    """
    snr_db = signal_noise_ratio(out_quantized, out_baseline).mean().item()
    linear_nsr = 10 ** (-snr_db / 10)
    return linear_nsr


def compute_layer_sensitivity(model_id="facebook/DiT-XL-2-256", bits=4, lambda_param=0.1, num_samples=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_id}...")
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    transformer = pipe.transformer
    transformer.eval()

    print("Preparing realistic inputs (adding noise to latents)...")
    latent_size = pipe.vae.config.latent_channels if hasattr(pipe, 'vae') else 4
    sample_size = transformer.config.sample_size
    
    latents_x0 = torch.randn((num_samples, latent_size, sample_size, sample_size), dtype=torch.float16, device=device)
    timesteps = torch.linspace(0, pipe.scheduler.config.num_train_timesteps - 1, num_samples, device=device).long()
    noise = torch.randn_like(latents_x0)
    realistic_hidden_states = pipe.scheduler.add_noise(latents_x0, noise, timesteps).to(torch.float16)
    realistic_class_labels = torch.randint(0, 1000, (num_samples,), device=device)

    print("Computing baseline output...")
    with torch.no_grad():
        out_baseline = transformer(
            realistic_hidden_states, 
            timesteps, 
            class_labels=realistic_class_labels
        ).sample

    divergence_scores = []
    activation_magnitudes = []
    
    current_activation_mag = 0.0
    def get_activation_hook():
        def hook(module, input, output):
            nonlocal current_activation_mag
            current_activation_mag = output[0].abs().mean().item()
        return hook

    blocks = transformer.transformer_blocks
    print(f"Starting sensitivity analysis for {len(blocks)} blocks...")
    
    for i, block in enumerate(tqdm(blocks, desc="Analyzing Blocks")):
        
        handle = block.register_forward_hook(get_activation_hook())
        orig_weights = apply_fake_quantization_to_module(block, bits=bits)
        
        with torch.no_grad():
            out_quantized = transformer(
                realistic_hidden_states, 
                timesteps, 
                class_labels=realistic_class_labels
            ).sample
            
        handle.remove()
        restore_original_weights(block, orig_weights)
        
        divergence = compute_snr_divergence(out_baseline, out_quantized)
        
        divergence_scores.append(divergence)
        activation_magnitudes.append(current_activation_mag)

    divergence_scores = np.array(divergence_scores)
    activation_magnitudes = np.array(activation_magnitudes)
    
    max_act_mag = np.max(activation_magnitudes)
    final_scores = divergence_scores * (1 + lambda_param * (activation_magnitudes / max_act_mag))
    
    print("\n=== LAYER-WISE SENSITIVITY RESULTS (Higher Score = More Sensitive) ===")
    for i, score in enumerate(final_scores):
        print(f"Layer {i:02d}: Score={score:.6f} | Divergence={divergence_scores[i]:.6f} | Act Mag={activation_magnitudes[i]:.4f}")

    mu_score = np.mean(final_scores)
    sigma_score = np.std(final_scores)
    
    print("\n--- Adaptive Thresholds for Mixed-Precision ---")
    threshold_high = mu_score + 0.5 * sigma_score
    threshold_mid = mu_score
    print(f"FP32/BF16 Threshold (Highly Sensitive): > {threshold_high:.6f}")
    print(f"INT8 Threshold (Medium Sensitivity):    > {threshold_mid:.6f}")
    print(f"INT4 Threshold (Low Sensitivity):       <= {threshold_mid:.6f}")
    
    print("\nGenerating plot...")
    plt.figure(figsize=(12, 6))
    layers = np.arange(len(final_scores))
    
    plt.plot(layers, final_scores, marker='o', linestyle='-', color='b', linewidth=2, label='Sensitivity Score')
    
    plt.axhline(y=threshold_high, color='r', linestyle='--', linewidth=1.5, label='High Threshold (FP32/BF16)')
    plt.axhline(y=threshold_mid, color='orange', linestyle='--', linewidth=1.5, label='Medium Threshold (INT8)')
    
    ymin, ymax = plt.ylim()
    plt.axhspan(threshold_high, ymax, facecolor='red', alpha=0.1)
    plt.axhspan(threshold_mid, threshold_high, facecolor='orange', alpha=0.1)
    plt.axhspan(ymin, threshold_mid, facecolor='green', alpha=0.1)
    
    plt.title('Layer-Wise Sensitivity Curve (U-Shape)', fontsize=14, fontweight='bold')
    plt.xlabel('Transformer Block Index', fontsize=12)
    plt.ylabel('Sensitivity Score (Higher = More Sensitive)', fontsize=12)
    
    plt.xticks(layers)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('sensitivity_curve.png', dpi=300)
    print("Plot saved successfully as 'sensitivity_curve.png'!")
    plt.show()
    
    return final_scores

if __name__ == "__main__":
    compute_layer_sensitivity()
