import argparse
import torch
import yaml
import json
import os
from diffusers import DiTPipeline
from slider_quant import apply_sliderquant

def calculate_window_index(layer_shallow: int, layer_int: int, layer_deep: int, window_size: int, window_step: int) -> list[tuple[int]]:
    window_list = []
    curr_window = []

    for i in range(layer_shallow):
        curr_window.append(i)
        window_list.append(tuple(curr_window.copy()))

    i = layer_shallow-1
    curr_window = [i + x for x in range(window_size)]

    for i in range(layer_shallow+1, layer_shallow + layer_int+2):
        window_list.append(tuple(curr_window.copy()))
        for _ in range(window_step):
            curr_window.pop(0)
            curr_window.append(i)

    i = layer_shallow + layer_int
    curr_window = [i + x for x in range(layer_deep)]

    for i in range(i, layer_shallow + layer_int + layer_deep):
        window_list.append(tuple(curr_window.copy()))
        curr_window.pop(0)

    return window_list

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    #Da riattivare required nella versione finale
    parser.add_argument("-m", "--model", type=str ,required=False, help="Model name (required)")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to config yaml")

    args = parser.parse_args()
    model_id = args.model
    
    #TEMPORANEO, SOLO PER COMODITA
    model_id = "facebook/DiT-XL-2-256"

    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    epoch_num = config.get('epoch', 1)
    class_num = config.get('class_n', 1)
    bits_int = config.get('bits_int', 4)
    bits_ext = config.get('bits_ext', 8)
    batch_size = config.get('batch_size', 4)
    window_size = config.get('window_size', 2)
    window_step = config.get('window_step', 1)
    gamma = config.get('gamma', 0.5)
    rank = config.get('rank', 16)
    group_size = config.get('group_size', 128)
    layer_shallow = config.get('layer_shallow', 4)
    layer_int = config.get('layer_int', 20)
    layer_deep = config.get('layer_deep', 4)

    print(f"Model: {model_id}")
    print(f"Epochs: {epoch_num}")
    print(f"Class number: {class_num}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=False).to(device)

    pipe.scheduler.set_timesteps(20)
    timesteps = pipe.scheduler.timesteps.to(device)
    timesteps = [t.unsqueeze(0) for t in timesteps]

    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)
    
    # Apply SliderQuant (modifies the pipe)
    pipe = apply_sliderquant(pipe, device, timesteps, window_list, layer_shallow, layer_int, gamma, epoch_num, class_num, bits_int, bits_ext, rank, group_size, batch_size)

    # Save the quantized model
    out_dir = f"output/{model_id}"
    os.makedirs(out_dir, exist_ok=True)
    pipe.save_pretrained(out_dir, safe_serialization=True)

    # Save the quantization configuration
    config['quant_method'] = "slider_quant"
    config_out_path = os.path.join(out_dir, "quantization_config.json")
    with open(config_out_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved quantization config to {config_out_path}")