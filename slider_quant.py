from asyncio.windows_utils import pipe
import json

import torch
import torch.nn as nn
from diffusers import DiTPipeline

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


def calc_original_outputs(pipe: DiTPipeline,timesteps: list[torch.Tensor],class_id: list[torch.Tensor],latents: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the output of the original model
    '''
    latents_copy = latents.clone().detach()
    hidden_states = pipe.transformer.pos_embed(latents_copy)
    original_outputs = {}
    with torch.no_grad():
        for c in class_id:
            c_value = c.item()
            original_outputs[str(c_value)] = {}
            for t in timesteps:
                t_value = t.item()
                original_outputs[str(c_value)][str(t_value)] = {}
                latents_copy = hidden_states.clone().detach()
                for id,layer in enumerate(pipe.transformer.transformer_blocks):
                    latents_copy = layer(latents_copy, timestep=t, class_labels=c).clone().detach()
                    print(f"{latents_copy.clone().detach().tolist()[0][0][0]} timestep: {t_value}, class: {c_value}, layer: {id} - Original output calculated")

                    # Write the output of the original model
                    original_outputs[str(c_value)][str(t_value)][str(id)] = latents_copy.clone().detach().tolist()
                #print(f"{original_outputs[str(c_value)][str(t_value)][str(id)][0][0][0]} - Original output calculated for class: {c_value}, timestep: {t_value}, layer: {id}") 
                if torch.isnan(torch.tensor(original_outputs[str(c_value)][str(t_value)][str(id)])).any():
                    print(f"NaN detected in original outputs for class {c_value} and timestep {t_value} at layer {id}")
                    break
            
            
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
    original_outputs = calc_original_outputs(pipe,timesteps,class_id,latents)
    with open("output/original_output.json", "w") as f:
        json.dump(original_outputs, f, indent=4)
    # Write the original outputs on a json file
    completed_epoch = 0
    exit()
    # Calculate the windows for the algorithm
    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)

    latents_copy = latents.clone().detach()
    quantized_output = {}
    
    for window_id, window in enumerate(window_list):
        quantized_output[window_id] = {}
        linear_modules = []
        original_weights_backup = []
        
        for layer_id in window:
            for _, module in pipe.transformer.transformer_blocks[layer_id].named_modules():
                if isinstance(module, nn.Linear):
                    linear_modules.append(module)
                    original_weights_backup.append(module.weight.data.clone().detach())
        print(f"Window: {window}, {window_id} number of linear modules: {len(linear_modules)}")
        for g in [gamma, 1.0]:       
            quantized_tensors = sliding_quantize_tensors(original_weights_backup, bits=bits, gamma=g)
            
            for module, q_tensor in zip(linear_modules, quantized_tensors):
                module.weight.data.copy_(q_tensor)
            
            for current_epoch in range(completed_epoch,epoch_num+1):            
                for c in class_id:
                    c_value = c.item()
                    quantized_output[window_id][str(c_value)] = {}
                    for t_id, t in enumerate(timesteps):
                        t_value = t.item()
                        quantized_output[window_id][str(c_value)][str(t_value)] = {}
                        if window[0] > 0:
                            latents_copy = torch.tensor(quantized_output[window_id-1][str(c_value)][str(t_value)][str(window[0]-1)], device=device, dtype=torch.float16)
                        elif t_id > 0:
                            latents_copy = torch.tensor(original_outputs[str(c_value)][str(timesteps[t_id-1].item())][str(window_list[-1][-1])].clone().detach(), device=device, dtype=torch.float16)
                        else:
                            latents_copy = latents.clone().detach()
                        for layer_id in window:
                            layer = pipe.transformer.transformer_blocks[layer_id]
                            latents_copy = layer(latents_copy, timestep=t, class_labels=c).clone().detach()
                            print(f"Window: {window}, Epoch: {current_epoch+1}/{epoch_num}, Timestep: {t_value}, Class: {c.item()}")
                            # Write the output of the quantized model
                            quantized_output[window_id][str(c_value)][str(t_value)][str(layer_id)] = latents_copy.clone().detach().tolist()
                            print(f"Window: {window_id}, Epoch: {current_epoch+1}/{epoch_num}, Timestep: {t_value}, layer_id: {layer_id}, Class: {c.item()} - Output saved")
                    with open("output/quantized_output.json", "w") as f:
                        json.dump(quantized_output, f, indent=4)
                if current_epoch < epoch_num:
                    pass
                    # Optimizer

                # Save in the json the optimized weight and epoch completed

        # Reset weight after window finished
        for module, orig_tensor in zip(linear_modules, original_weights_backup):
            module.weight.data.copy_(orig_tensor)
    

    return pipe
    
if __name__=="__main__":
    pass