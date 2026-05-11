import torch
import torch.nn as nn

'''Totalmente da risistemare per far si che dia in output la lista di tensori quantizzati'''
def quantize_tensors(tensor_list, bits=4, gamma=1):
    # Idempotent function that quantize a list of tensor
    zmin = tensor_list.min()
    zmax = tensor_list.max()
    qmax = 2**bits - 1
    alpha = (zmax-zmin)/qmax
    beta = torch.round(zmin/alpha)

    quantized_list = [torch.round(tensor / alpha) - beta for tensor in tensor_list]
    quantized_list = [quantized.clamp(0, qmax) for quantized in quantized_list]
    dequantized_list = [(quantized + beta) * alpha for quantized in quantized_list]

    
    '''if gamma < 1.0:
        for i, dequantized in enumerate(dequantized_list):
            num_rows = dequantized.size(0)
            limit_row = int(num_rows * gamma)
            new_weight = tensor_list[i].clone()
            new_weight[:limit_row, :] = dequantized[:limit_row, :]
            module_list[i].weight.copy_(new_weight)
    else:
        module.weight.copy_(quantized_weight)'''



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
        # Quantize all layer in the window
        for g in [gamma,1]:
            # Mettere liste per salvare i pesi originali per poterli resettare dopo
            tensors_to_quantize = []
            for layer_id, in window:
                for _, module in pipe.transformer.transformer_blocks[layer_id].named_modules():
                    if isinstance(module, nn.Linear):
                        tensors_to_quantize.append(module.weight.copy())
            quantize_tensors(tensors_to_quantize, bits=bits, gamma=g)
            
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