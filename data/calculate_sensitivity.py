import torch
import torch.nn as nn
from diffusers import DiTPipeline
import numpy as np
from tqdm import tqdm
import sys
import os
import json
from torchmetrics.functional import signal_noise_ratio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from script.slider_quant import SliderQuantLinear, SKIP_NAMES

def apply_fake_quantization_to_module(module: nn.Module, bits: int, group_size: int):
    """
    Dynamically replaces the Linear layers of a module with the SliderQuantLinear class.
    Returns a dictionary containing the original layers so they can be restored.
    """
    original_linears = {}
    
    def replace_linears(m, prefix=""):
        for name, child in m.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
                
            if isinstance(child, nn.Linear):
                original_linears[full_name] = child
                
                sq_linear = SliderQuantLinear(
                    original_linear=child, 
                    bits_weight=bits, 
                    bits_act=bits, 
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
    

def compute_snr_divergence(out_baseline, out_quantized):
    """
    Calculates the relative error using torchmetrics' signal_noise_ratio.
    Transforms the SNR (dB) into its linear inverse (Noise-to-Signal Ratio).
    A higher value indicates greater sensitivity to quantization noise.
    """
    snr_db = signal_noise_ratio(out_quantized, out_baseline).mean().item()
    linear_nsr = 10 ** (-snr_db / 10)
    return linear_nsr


def compute_layer_sensitivity(model_id: str, bits: int, group_size: int, lambda_param: float, num_samples: int):
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
        orig_weights = apply_fake_quantization_to_module(block, bits, group_size)
        
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
    
    mu_score = np.mean(final_scores)
    sigma_score = np.std(final_scores)

    json_file = {
        "max_act_mag": float(max_act_mag),
        "mean": float(mu_score),
        "std": float(sigma_score),
        "layers": []
    }

    print("\n=== LAYER-WISE SENSITIVITY RESULTS (Higher Score = More Sensitive) ===")
    for i, score in enumerate(final_scores):
        print(f"Layer {i:02d}: Score={score:.6f} | Divergence={divergence_scores[i]:.6f} | Act Mag={activation_magnitudes[i]:.4f}")
        layer_values = {
            "score": float(score),
            "divergence": float(divergence_scores[i]),
            "act_mag": float(activation_magnitudes[i])
        }
        json_file["layers"].append(layer_values)

    with open("sensitivity_results.json", "w") as f:
        json.dump(json_file, f, indent=4)
    
    return final_scores

if __name__ == "__main__":
    compute_layer_sensitivity(
        "facebook/DiT-XL-2-256",
        4,
        128,
        0.1,
        16
    )
