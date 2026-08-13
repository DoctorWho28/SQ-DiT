import argparse
import torch
import yaml
import json
import os
import time
from diffusers import DiTPipeline
from script.slider_quant import apply_sliderquant

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
    config_path = args.config


    
    #TEMPORANEO, SOLO PER COMODITA
    model_id = "facebook/DiT-XL-2-256"


    print(f"Model: {model_id}")
    print(f"Config file: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    # Config parameters and checks
    epoch_num = config.get('epoch', 1)
    assert(epoch_num>0),"Epoch num must be a positive number"

    class_num = config.get('class_n', 1)
    assert(class_num>0 and class_num<=1000),"Class num must be in range [1,1000]"

    bits_int = config.get('bits_int', 4)
    assert(bits_int > 0 and (bits_int & (bits_int - 1)) == 0),"Bits int must be a power of 2"

    bits_ext = config.get('bits_ext', 8)
    assert(bits_ext > 0 and (bits_ext & (bits_ext - 1)) == 0),"Bits ext must be a power of 2"

    bits_act = config.get('bits_act', 8)
    assert(bits_act > 0 and (bits_act & (bits_act - 1)) == 0),"Act bits must be a power of 2"

    batch_size = config.get('batch_size', 4)
    assert(batch_size>0),"Batch size must be a positive number"

    window_size = config.get('window_size', 2)
    assert(window_size>0),"Window size must be a positive number"

    window_step = config.get('window_step', 1)
    assert(window_step>0),"Window step must be a positive number"
    assert(window_step <= window_size),"Window step must be lower or equal to window size"

    gamma = config.get('gamma', 0.5)
    assert(gamma>0 and gamma<=1),"Gamma must be in range (0,1]"

    rank = config.get('rank', 16)
    assert(rank>0),"Rank must be a positive number"

    group_size = config.get('group_size', 128)
    assert(group_size>0),"Group size must be a positive number"

    layer_shallow = config.get('layer_shallow', 4)
    assert(layer_shallow>=0),"Layer shallow must be a positive number or 0"

    layer_deep = config.get('layer_deep', 4)
    assert(layer_deep>=0),"Layer deep must be a positive number or 0"

    inference_step = config.get('inference_step', 20)
    assert(inference_step>0 and inference_step<=1000),"Inference step must be in range [1,1000]"

    
    print(f"Epochs: {epoch_num}")
    print(f"Quantization: W{bits_int}A{bits_act} (Ext: W{bits_ext}A{bits_act})")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=False).to(device)


    num_layers = len(pipe.transformer.transformer_blocks)
    layer_int = num_layers - layer_shallow - layer_deep
    assert(layer_int>=0),"Layers shallow and deep cannot overlap" 
    assert(window_size<=num_layers),"Window size must be lower than the number of layers"


    pipe.scheduler.set_timesteps(inference_step)
    timesteps = pipe.scheduler.timesteps.to(device)
    timesteps = [t.unsqueeze(0) for t in timesteps]

    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)


    torch.cuda.reset_peak_memory_stats()
    start_quant = time.time()

    # Apply SliderQuant (modifies the pipe)
    pipe = apply_sliderquant(pipe, device, timesteps, window_list, layer_shallow, layer_int, gamma, epoch_num, class_num, bits_int, bits_ext, bits_act, rank, group_size, batch_size)


    quantization_time = time.time() - start_quant
    vram_end = torch.cuda.memory_allocated() / (1024**3)
    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)


    # Save the quantized model
    out_dir = f"output/{model_id}-W{bits_int}A{bits_act}"
    id = 1
    while os.path.exists(out_dir):
        id += 1
        out_dir = f"output/{model_id}-W{bits_int}A{bits_act}-{id}"
    

    os.makedirs(out_dir, exist_ok=True)
    pipe.save_pretrained(out_dir, safe_serialization=True)

    # Save the quantization configuration
    config['quant_method'] = "slider_quant"
    config_out_path = os.path.join(out_dir, "quantization_config.json")
    with open(config_out_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved quantization config to {config_out_path}")


    # Statistics saving

    def get_dir_size(path):
            return sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(path) for f in filenames)


    model_size = get_dir_size(out_dir) / (1024**2)


    # JSON of data
    json_path = f"json/{model_id}-W{bits_int}A{bits_act}-{id}.json"

    json_file = {"quantization":{
        "time": quantization_time,
        "vram_quant_model": vram_end,
        "vram_max_quant": vram_peak,
        "model_size (MB)": model_size}}

    with open(json_path,"w") as J:
        json.dump(json_file,J,indent=4)



    