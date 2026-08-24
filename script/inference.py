import os
import json
import torch
import torch.nn as nn
from diffusers import DiTPipeline
from slider_quant import WXAXLinear, SKIP_NAMES, HIGH_NAMES
from safetensors.torch import load_file
import random
import argparse

def inject_WXAX(module: nn.Module, bits_weight_low: int, bits_weight_high: int, bits_act: int, group_size: int, path: str =""):
    for name, child in module.named_children():
        full_name = f"{path}.{name}" if path else name
        if any(skip == name for skip in SKIP_NAMES):
            continue
        
        if isinstance(child, nn.Linear):
            current_weight_bits = bits_weight_low
            if any(high in full_name for high in HIGH_NAMES):
                current_weight_bits = bits_weight_high
            wxax = WXAXLinear(child.in_features, child.out_features, group_size, current_weight_bits, bits_act, bias=(child.bias is not None))
            setattr(module, name, wxax)
        elif not isinstance(child, nn.LayerNorm):
            inject_WXAX(child, bits_weight_low, bits_weight_high, bits_act, group_size, full_name)


def load_quantized_pipeline(quant_dir: str, device: str) -> DiTPipeline:
    """
    Loads a DiT model quantized with SliderQuant and returns the pipeline
    """
        
    if not os.path.exists(quant_dir):
        raise FileNotFoundError(f"Quantized model directory not found: {quant_dir}")
        
    config_path = os.path.join(quant_dir, "quantization_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found: {config_path}")
        
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        q_config = json.load(f)
        
    layer_shallow = q_config.get("layer_shallow")
    layer_deep = q_config.get("layer_deep")
    group_size = q_config.get("group_size")
    bits_low = q_config.get("bits_low")
    bits_high = q_config.get("bits_high")
    bits_act = q_config.get("bits_act")
    inference_step = q_config.get("inference_step")

    index_path = os.path.join(quant_dir, "model_index.json")
    with open(index_path, "r") as f:
        model_index = json.load(f)
    
    base_model_id = model_index.get("_name_or_path")

    print(f"Loading base model {base_model_id}...")
    pipe = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    layer_int = len(pipe.transformer.transformer_blocks) - layer_shallow - layer_deep
    
    print(f"Injecting mixed WXAX layers...")
    
    
                
    # Shallow Injection
    for layer_id in range(layer_shallow):
        inject_WXAX(pipe.transformer.transformer_blocks[layer_id], bits_high, bits_high, bits_act,group_size)
    # Intermediate Injection
    for layer_id in range(layer_shallow, layer_shallow + layer_int):
        inject_WXAX(pipe.transformer.transformer_blocks[layer_id], bits_low, bits_high, bits_act,group_size)
    # Deep Injection
    for layer_id in range(layer_shallow + layer_int, len(pipe.transformer.transformer_blocks)):
        inject_WXAX(pipe.transformer.transformer_blocks[layer_id], bits_high, bits_high, bits_act,group_size)
        
    print("Loading packed uint8 weights from safetensors...")
    transformer_state_dict = load_file(os.path.join(quant_dir, "transformer", "diffusion_pytorch_model.safetensors"))
    pipe.transformer.load_state_dict(transformer_state_dict, strict=True)
    
    pipe = pipe.to(device)
    return pipe, inference_step

def gen_quant_image(quant_dir: str, class_label: list[int], device: str, seed: int | None):
    pipe, inference_step = load_quantized_pipeline(quant_dir, device)

    if seed is None:
        seed = random.randint(0, 10000)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Generating test image...")
    output = pipe(class_labels=class_label, generator=generator, num_inference_steps=inference_step)
    output.images[0].save(f"quantized_image_seed_{seed}.png")
    print(f"Image saved as 'quantized_image_seed_{seed}.png'")


def gen_orig_image(model_id: str, inference_step: int, class_label: list[int], device: str, seed: int | None):
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    if seed is None:
        seed = random.randint(0, 10000)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Generating test image with original model ({model_id})...")
    output = pipe(class_labels=class_label, generator=generator, num_inference_steps=inference_step)
    output.images[0].save(f"original_image_seed_{seed}.png")
    print(f"Image saved as 'original_image_seed_{seed}.png'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from a quantized or original DiT model.")
    parser.add_argument("-d", "--quant_dir", type=str, default=None, help="Path to the quantized model directory")
    parser.add_argument("-m", "--model_id", type=str, default=None, help="HuggingFace model ID for original model (e.g. facebook/DiT-XL-2-256)")
    parser.add_argument("-i", "--inference_steps", type=int, default=20, help="Number of inference steps (required for original model)")
    parser.add_argument("-c", "--class_label", type=int, default=19, help="Class label to generate (default: 19)")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed (default: random)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.quant_dir:
        gen_quant_image(args.quant_dir, [args.class_label], device, seed=args.seed)
    if args.model_id:
        gen_orig_image(args.model_id, args.inference_steps, [args.class_label], device, seed=args.seed)
