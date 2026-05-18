import torch
import torch.nn as nn

import torch

def sliding_quantize_tensors(tensor_list, bits=4, gamma=1.0):
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



def calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step):
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


def calc_original_outputs(pipe,timesteps,class_id,latents):
    # Calculate the output of the original model
    latents_copy = latents.copy()
    original_outputs = {}
    with torch.no_grad():
        for c in class_id:
            for t in timesteps:
                t_value = t.unsqueeze(0)
                for id,layer in enumerate(pipe.transformer.transformer_blocks):
                    latents_copy = layer(latents_copy, t_value, class_labels=c).sample.clone()

                    # Write the output of the original model
                    if t_value not in original_outputs:
                        original_outputs[t_value] = {}
                    original_outputs[t_value][str(id)] = latents_copy.copy()
            
            # Reset latents after each class image
            latents_copy = latents.copy()
    return original_outputs


def apply_sliderquant(pipe,device,timesteps,layer_shallow,layer_int,layer_deep,window_size,window_step,gamma,epoch_num,class_num,bits):
    latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16)
    class_steps = int(1000/class_num)
    class_id = [[i] for i in range(0,1000,class_steps)]


    '''If already exists a json with the parameters, 
    load the original outputs and skip this function'''
    original_outputs = calc_original_outputs(pipe,timesteps,class_id,latents)
    # Write the original outputs on a json file
    completed_epoch = 0

    # Calculate the windows for the algorithm
    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)

    latents_copy = latents.copy()

    for window in window_list:
        linear_modules = []
        original_weights_backup = []
        
        for layer_id in window:
            for _, module in pipe.transformer.transformer_blocks[layer_id].named_modules():
                if isinstance(module, nn.Linear):
                    linear_modules.append(module)
                    original_weights_backup.append(module.weight.data.clone().detach())
        
        for g in [gamma, 1.0]:       
            quantized_tensors = sliding_quantize_tensors(original_weights_backup, bits=bits, gamma=g)
            
            for module, q_tensor in zip(linear_modules, quantized_tensors):
                module.weight.data.copy_(q_tensor)
            
            for epoch in range(completed_epoch,epoch_num):
                quantized_output = {}
                for c in class_id:
                    for t in timesteps:
                        t_value = t.unsqueeze(0)
                        for layer_id, in window:
                            layer = pipe.transformer.transformer_blocks[layer_id]
                            latents_copy = layer(latents_copy, t_value, class_labels=c).sample.clone()

                            # Write the output of the quantized model
                            if window not in quantized_output:
                                quantized_output[window] = {}
                            if t_value not in quantized_output[window]:
                                quantized_output[window][t_value] = {}
                            quantized_output[window][t_value][str(id)] = latents_copy.copy()
                
                    # Reset latents after each class image
                    latents_copy = latents.copy()

                # Optimizer

                # Save in the json the optimized weight and epoch completed

    

    
if __name__=="__main__":
    pass