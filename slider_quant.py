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
    original_outputs = {}
    with torch.no_grad():
        for c in class_id:
            for t in timesteps:
                t_value = t.item()
                for id,layer in enumerate(pipe.transformer.transformer_blocks):
                    latents_copy = layer(latents_copy, timestep=t, class_labels=c).sample.clone()

                    # Write the output of the original model
                    if str(t_value) not in original_outputs:
                        original_outputs[str(t_value)] = {}
                    original_outputs[str(t_value)][str(id)] = latents_copy.clone().detach()
            
            # Reset latents after each class image
            latents_copy = latents.clone().detach()
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


    latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16)
    class_steps = int(1000/class_num)
    class_id = [torch.tensor([i],device=device) for i in range(0,1000,class_steps)]


    '''If already exists a json with the parameters, 
    load the original outputs and skip this function'''
    original_outputs = calc_original_outputs(pipe,timesteps,class_id,latents)
    # Write the original outputs on a json file
    completed_epoch = 0

    # Calculate the windows for the algorithm
    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)

    latents_copy = latents.clone().detach()

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
                        t_value = t.item()
                        for layer_id, in window:
                            layer = pipe.transformer.transformer_blocks[layer_id]
                            latents_copy = layer(latents_copy, timestep=t, class_labels=c).sample.clone()

                            # Write the output of the quantized model
                            if window not in quantized_output:
                                quantized_output[window] = {}
                            if str(t_value) not in quantized_output[window]:
                                quantized_output[window][str(t_value)] = {}
                            quantized_output[window][str(t_value)][str(id)] = latents_copy.copy()
                
                    # Reset latents after each class image
                    latents_copy = latents.clone().detach()

                # Optimizer

                # Save in the json the optimized weight and epoch completed

        # Reset weight after window finished
        for module, orig_tensor in zip(linear_modules, original_weights_backup):
            module.weight._copy(orig_tensor)
    

    return pipe
    
if __name__=="__main__":
    pass