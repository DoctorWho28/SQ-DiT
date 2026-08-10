import os
import json
import torch
import torch.nn as nn
from diffusers import DiTPipeline
from slider_quant import WXAXLinear, SKIP_NAMES
from safetensors.torch import load_file
import random

def load_quantized_pipeline(
    quant_dir: str,
) -> DiTPipeline:
    """
    Loads a DiT model quantized with SliderQuant and returns the ready-to-use pipeline.
    
    Args:
        quant_dir: The directory containing the safetensors weights and quantization_config.json.
        base_model_id: The ID of the unquantized HuggingFace model (required for the base architecture).
        device: Device to run the model on ('cuda' or 'cpu'). If None, automatically selects one.
        
    Returns:
        DiTPipeline: The diffusion pipeline modified with the quantized weights.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if not os.path.exists(quant_dir):
        raise FileNotFoundError(f"Quantized model directory not found: {quant_dir}")
        
    config_path = os.path.join(quant_dir, "quantization_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found: {config_path}")
        
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        q_config = json.load(f)
        
    layer_shallow = q_config.get("layer_shallow")
    layer_int = q_config.get("layer_int")
    group_size = q_config.get("group_size")
    bits_int = q_config.get("bits_int")
    bits_ext = q_config.get("bits_ext")
    act_bits_int = q_config.get("act_bits_int")
    act_bits_ext = q_config.get("act_bits_ext")
    inference_step = q_config.get("inference_step")

    index_path = os.path.join(quant_dir, "model_index.json")
    with open(index_path, "r") as f:
        model_index = json.load(f)
    
    base_model_id = model_index.get("_name_or_path")

    print(f"Loading base model {base_model_id}...")
    pipe = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    print(f"Injecting mixed WXAX layers...")
    
    def inject_wXax(module, weight_bits, act_bits):
        for name, child in module.named_children():
            if any(skip in name for skip in SKIP_NAMES):
                continue
            if isinstance(child, nn.Linear):
                wX = WXAXLinear(child.in_features, child.out_features, group_size, weight_bits=weight_bits, act_bits=act_bits, bias=(child.bias is not None))
                setattr(module, name, wX)
            else:
                inject_wXax(child, weight_bits, act_bits)
                
    # Shallow Injection
    for layer_id in range(layer_shallow):
        inject_wXax(pipe.transformer.transformer_blocks[layer_id], weight_bits=bits_ext, act_bits=act_bits_ext)
    # Intermediate Injection
    for layer_id in range(layer_shallow, layer_shallow + layer_int):
        inject_wXax(pipe.transformer.transformer_blocks[layer_id], weight_bits=bits_int, act_bits=act_bits_int)
    # Deep Injection
    for layer_id in range(layer_shallow + layer_int, len(pipe.transformer.transformer_blocks)):
        inject_wXax(pipe.transformer.transformer_blocks[layer_id], weight_bits=bits_ext, act_bits=act_bits_ext)
        
    print("Loading packed uint8 weights from safetensors...")
    transformer_state_dict = load_file(os.path.join(quant_dir, "transformer", "diffusion_pytorch_model.safetensors"))
    pipe.transformer.load_state_dict(transformer_state_dict, strict=True)
    
    pipe = pipe.to(device)
    return pipe, inference_step

def gen_image(quant_dir, class_label, seed=None):
    pipe, inference_step = load_quantized_pipeline(
        quant_dir= quant_dir
    )
    
    if not seed:
        seed = random.randint(0, 10000)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Generating test image...")
    output = pipe(class_labels=class_label, generator=generator, num_inference_steps=inference_step)
    output.images[0].save(f"quantized_image_seed_{seed}.png")
    print("Image saved as 'quantized_image_seed_{seed}.png'")
    
if __name__ == "__main__":
    # Practical usage example
    gen_image("Output/facebook/DiT-XL-2-256_v6", [19])
