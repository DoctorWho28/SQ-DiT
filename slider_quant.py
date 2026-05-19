from asyncio.windows_utils import pipe
import json
import os
import torch
import torch.nn as nn
from diffusers import DiTPipeline
from safetensors.torch import save_file, load_file

def sliding_quantize_tensors(tensor_list: list[torch.Tensor], bits: int=4, gamma: float=1.0) -> list[torch.Tensor]:
    '''
    Apply the quantization of a group of tensors all together, given as parameters:
    - tensor_list: the list of tensor to quantize
    - bits: number of bits to quantize to
    - gamma: portion of each tensor to quantize
    '''

    result_list = []
    qmax = (2 ** bits) - 1

    for original_tensor in tensor_list:
        num_rows = original_tensor.size(0)
        
        if gamma > 1:
            gamma = 1.0
        
        limit_row = int(num_rows * gamma)

        if limit_row == 0:
            result_list.append(original_tensor.clone())
            continue

        target_slice = original_tensor[:limit_row, ...]

        zmin = target_slice.min()
        zmax = target_slice.max()

        if zmax == zmin:
            result_list.append(original_tensor.clone())
            continue

        alpha = (zmax - zmin) / qmax
        beta = torch.round(zmin / alpha)

        quantized_slice = torch.round(target_slice / alpha) - beta
        quantized_slice = quantized_slice.clamp(0, qmax)
        dequantized_slice = (quantized_slice + beta) * alpha

        mixed_tensor = original_tensor.clone()
        mixed_tensor[:limit_row, ...] = dequantized_slice
        
        result_list.append(mixed_tensor)

    return result_list



def calculate_window_index(layer_shallow: int, layer_int: int, layer_deep: int, window_size: int, window_step: int) -> list[tuple[int]]:
    '''
    Returns the list of window for the SliderQuant algorithm, given the parameters:
    - layer_shallow: number of shallow layers
    - layer_int: number of intermediate layers
    - layer_deep: number of deep layers
    - window_size: window size for intermediate layers
    - window_step: window step for intermediate layers
    '''

    # Returns a list of tuple containing the windows
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

def calc_original_outputs(pipe: DiTPipeline,timesteps: list[torch.Tensor],class_id: list[torch.Tensor],latents: torch.Tensor) -> dict[ str, dict[str, dict[str, torch.Tensor]]]:
    '''
    Calculate the output of the original model and return it in a flat dictionary, where the key is a string that contains the class, timestep and layer id, and the value is the output of the model for that class, timestep and layer.
    '''
    latents_copy = latents.clone().detach()
    hidden_states = pipe.transformer.pos_embed(latents_copy)

    original_outputs = {} 
    
    with torch.no_grad():
        for c in class_id:
            c_value = c.item()
            for t in timesteps:
                t_value = t.item()
                latents_copy = hidden_states.clone().detach()
                
                for layer_id, layer in enumerate(pipe.transformer.transformer_blocks):
                    latents_copy = layer(latents_copy, timestep=t, class_labels=c).clone().detach()

                    key = f"class_{c_value}_time_{t_value}_layer_{layer_id}"
                    original_outputs[key] = latents_copy.clone().detach()
                    
    return original_outputs


def apply_sliderquant(pipe: DiTPipeline,device: str,timesteps: list[torch.Tensor],layer_shallow: int,layer_int: int,layer_deep: int,window_size: int,window_step: int,gamma: float,epoch_num: int,class_num: int,bits: int) -> DiTPipeline:
    '''
    Given a module apply the SliderQuant quantization, given the parameters:
    - pipe: pipeline of the model
    - device: device to use (cpu or cuda)
    - timesteps: list of timesteps to use when creating an image
    - layer_shallow: number of shallow layers
    - layer_int: number of intermediate layers
    - layer_deep: number of deep layers
    - window_size: window size for intermediate layers
    - window_step: window step for intermediate layers
    - gamma: portion of each tensor to quantize
    - epoch_num: number of epochs for each optimization step
    - class_num: number of class to generate images when optimizing
    - bits: number of bits to quantize to
    '''


    latents = torch.randn((1, 4, 32, 32), device=device, dtype=torch.float16)
    class_steps = int(1000/class_num)
    class_id = [torch.tensor([i],device=device) for i in range(0,1000,class_steps)]

    '''If already exists a json with the parameters, 
    load the original outputs and skip this function'''
    
    checkpoint_dir = "temp_quant_data"
    
    if os.path.exists("temp_quant_data/original_outputs.safetensors"):
        print("Loading original outputs from file...")
        original_outputs = load_file(f"{checkpoint_dir}/original_outputs.safetensors", device=device)
    else:
        original_outputs = calc_original_outputs(pipe, timesteps, class_id, latents)

        # save of original outputs in a safetensors file
        save_file(original_outputs, f"{checkpoint_dir}/original_outputs.safetensors")
    
    completed_epoch = 0
    
    # Calculate the windows for the algorithm
    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)

    latents_copy = latents.clone().detach()
    hidden_states = pipe.transformer.pos_embed(latents_copy)
    quantized_output = {}
    for window_id, window in enumerate(window_list):
        linear_modules = []
        original_weights = []
        
        for layer_id in window:
            for _, module in pipe.transformer.transformer_blocks[layer_id].named_modules():
                if isinstance(module, nn.Linear):
                    linear_modules.append(module)
                    original_weights.append(module.weight.data.clone().detach())
        
        print(f"Window: {window}, {window_id} number of linear modules: {len(linear_modules)}")
        for g in [gamma, 1.0]:       
            quantized_tensors = sliding_quantize_tensors(original_weights, bits=bits, gamma=g)
            
            for module, q_tensor in zip(linear_modules, quantized_tensors):
                module.weight.data.copy_(q_tensor)
            
            for current_epoch in range(completed_epoch, epoch_num+1):    
                for c in class_id:
                    c_value = c.item()
                    for t_id, t in enumerate(timesteps):
                        t_value = t.item()
                        if window[0] > 0:
                            latents_copy = quantized_output[f"window_{window_id-1}_class_{c_value}_time_{t_value}_layer_{str(window[0]-1)}"]
                        elif t_id > 0:
                            latents_copy = original_outputs[f"class_{c_value}_time_{timesteps[t_id-1].item()}_layer_{window_list[-1][-1]}"]
                        else:
                            latents_copy = hidden_states.clone().detach()
                        for layer_id in window:
                            layer = pipe.transformer.transformer_blocks[layer_id]
                            latents_copy = layer(latents_copy, timestep=t, class_labels=c).clone().detach()

                            # Write the output of the quantized model
                            key = f"window_{window_id}_class_{c_value}_time_{t_value}_layer_{layer_id}"
                            quantized_output[key] = latents_copy.clone().detach()
                    
                    print(f"Window: {window_id}, gamma: {g}, Epoch: {current_epoch}/{epoch_num+1}, Timestep: {t_value}, layer_id: {layer_id}, Class: {c.item()} - Output saved")
                    save_file(quantized_output, f"{checkpoint_dir}/quantized_output.safetensors")
                if current_epoch < epoch_num:
                    pass
                    # Optimizer

                # Save in the json the optimized weight and epoch completed

        # Reset weight after window finished
        for module, orig_tensor in zip(linear_modules, original_weights):
            module.weight.data.copy_(orig_tensor)
    

    return pipe
    
if __name__=="__main__":
    pass