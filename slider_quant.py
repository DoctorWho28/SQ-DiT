import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

def pack_weights_to_intX(quantized_weights: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Packs a tensor of integers into a uint8 tensor dynamically based on bits.
    Supports 8-bit, 4-bit, and 2-bit.
    """
    quantized_weights = quantized_weights.to(torch.uint8)
    
    if bits == 8:
        return quantized_weights
        
    elif bits== 4:
        assert quantized_weights.shape[-1] % 2 == 0, "Packing Error: Weight dimension must be even for int4 packing."
        even = quantized_weights[..., 0::2]
        odd = quantized_weights[..., 1::2]
        return (odd << 4) | even
        
    elif bits == 2:
        assert quantized_weights.shape[-1] % 4 == 0, "Packing Error: Weight dimension must be multiple of 4 for int2 packing."
        b0 = quantized_weights[..., 0::4]
        b1 = quantized_weights[..., 1::4]
        b2 = quantized_weights[..., 2::4]
        b3 = quantized_weights[..., 3::4]
        return (b3 << 6) | (b2 << 4) | (b1 << 2) | b0
        
    else:
        raise ValueError(f"Packing Error: {bits} bits not implemented, use 2, 4 or 8 bits")

def unpack_intX_weights(packed_weights: torch.Tensor, original_shape: tuple, bits: int) -> torch.Tensor:
    """
    Unpacks a uint8 tensor back to a tensor of X-bit integers.
    """
    if bits == 8:
        return packed_weights
        
    unpacked = torch.empty(original_shape, dtype=torch.uint8, device=packed_weights.device)
    
    if bits == 4:
        unpacked[..., 0::2] = packed_weights & 0x0F
        unpacked[..., 1::2] = (packed_weights >> 4) & 0x0F
        
    elif bits == 2:
        unpacked[..., 0::4] = packed_weights & 0x03
        unpacked[..., 1::4] = (packed_weights >> 2) & 0x03
        unpacked[..., 2::4] = (packed_weights >> 4) & 0x03
        unpacked[..., 3::4] = (packed_weights >> 6) & 0x03
        
    else:
        raise ValueError(f"Unpacking Error: {bits} bits not implemented, use 2, 4 or 8 bits")
        
    return unpacked

def group_quantize_tensor(tensor: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    '''
    Group Quantization implementation.
    Isolates outliers by dividing channels into small groups.
    '''
    qmax = (2 ** bits) - 1
    
    original_shape = tensor.shape
    assert original_shape[1] % group_size == 0, f"Group Quant Error: In-features {original_shape[1]} must be divisible by group_size {group_size}"
    
    tensor_grouped = tensor.view(original_shape[0], original_shape[1] // group_size, group_size)
    
    zmin = tensor_grouped.min(dim=-1, keepdim=True)[0]
    zmax = tensor_grouped.max(dim=-1, keepdim=True)[0]
    
    alpha = (zmax - zmin) / qmax
    safe_alpha = torch.where(alpha == 0, torch.ones_like(alpha), alpha)
    beta = torch.round(zmin / safe_alpha)
    
    quantized = RoundSTE.apply(tensor_grouped / safe_alpha) - beta
    quantized = quantized.clamp(0, qmax)
    
    dequantized = (quantized + beta) * safe_alpha
    dequantized = torch.where((zmax == zmin), tensor_grouped, dequantized)
    dequantized = dequantized.view(original_shape)
    return dequantized

class WXA16Linear(nn.Module):
    """
    A dynamic WXA16 linear layer. 
    Stores weights physically packed as uint8 based on specified bits (2, 4, 8).
    At runtime, unpacks to fp16, scales, and computes linear.
    """
    def __init__(self, in_features: int, out_features: int, group_size: int, bits: int, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        
        pack_factor = 8 // bits
        assert in_features % pack_factor == 0, f"in_features {in_features} not divisible by packing factor {pack_factor}"
        
        self.register_buffer("weight_packed", torch.zeros((out_features, in_features // pack_factor), dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros((out_features, in_features // group_size, 1), dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros((out_features, in_features // group_size, 1), dtype=torch.float16))
        
        if bias is not None:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack on the fly
        unpacked_int = unpack_intX_weights(self.weight_packed, (self.out_features, self.in_features), self.bits).half()
        
        # Reshape to group size
        unpacked_grouped = unpacked_int.view(self.out_features, self.in_features // self.group_size, self.group_size)
        
        # Dequantize
        dequantized_grouped = (unpacked_grouped + self.zeros) * self.scales
        
        # Flatten back
        w_fp16 = dequantized_grouped.view(self.out_features, self.in_features)
        
        return F.linear(x.half(), w_fp16, self.bias)

class SliderQuantLinear(nn.Module):
    """
    Wrapper for Fake Quantization during training with dynamic bits.
    """
    def __init__(self, original_linear: nn.Linear, bits: int, rank: int, gamma: float, group_size: int):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        self.rank = rank
        self.gamma = gamma
        self.group_size = group_size
        
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        
        orig_device = original_linear.weight.device
        
        self.alpha = nn.Parameter(torch.ones(self.in_features, dtype=torch.float32, device=orig_device))
        self.A = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=torch.float32, device=orig_device))
        self.B = nn.Parameter(torch.zeros(self.rank, self.in_features, dtype=torch.float32, device=orig_device))
        nn.init.normal_(self.A, std=0.01)

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        w_f32 = self.weight.float()
        
        safe_alpha = torch.clamp(self.alpha, min=0.1, max=10.0)
        x_scaled = x_f32 / safe_alpha
        
        w_adjusted = w_f32 * safe_alpha.view(1, -1) + torch.matmul(self.A, self.B)
        
        limit_row = int(self.out_features * self.gamma)
        
        if limit_row > 0:
            w_quant_part= group_quantize_tensor(w_adjusted[:limit_row, :], self.bits, self.group_size)
            if limit_row < self.out_features:
                w_final = torch.cat([w_quant_part, w_adjusted[limit_row:, :]], dim=0)
            else:
                w_final = w_quant_part
        else:
            w_final = w_adjusted

        bias_f32 = self.bias.float() if self.bias is not None else None
        out = F.linear(x_scaled, w_final, bias_f32)
        return out.to(orig_dtype)

def replace_linears_with_sliderquant(module: nn.Module, bits: int, rank: int, gamma: float, group_size: int, skip_names: list[str]) -> list[nn.Module]:
    replaced_modules = []
    for name, child in module.named_children():
        if any(skip in name for skip in skip_names):
            continue
            
        if isinstance(child, nn.Linear):
            sq_linear = SliderQuantLinear(child, bits=bits, rank=rank, gamma=gamma, group_size=group_size)
            setattr(module, name, sq_linear)
            replaced_modules.append(sq_linear)
        elif isinstance(child, SliderQuantLinear):
            replaced_modules.append(child)
        else:
            replaced_modules.extend(replace_linears_with_sliderquant(child, bits, rank, gamma, group_size, skip_names))
    return replaced_modules

def merge_and_pack_linears(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, SliderQuantLinear):
            with torch.no_grad():
                safe_alpha = torch.clamp(child.alpha, min=0.1, max=10.0)
                w_adjusted = child.weight * safe_alpha.view(1, -1) + (child.A @ child.B)
                
                w_final_merged = w_adjusted / safe_alpha.view(1, -1)
                
                qmax = (2 ** child.bits) - 1
                group_size = child.group_size
                tensor_grouped = w_final_merged.view(child.out_features, child.in_features // group_size, group_size)
                
                zmin = tensor_grouped.min(dim=-1, keepdim=True)[0]
                zmax = tensor_grouped.max(dim=-1, keepdim=True)[0]
                alpha = (zmax - zmin) / qmax
                safe_alpha_group = torch.where(alpha == 0, torch.ones_like(alpha), alpha)
                beta = torch.round(zmin / safe_alpha_group)
                
                quantized = torch.round(tensor_grouped / safe_alpha_group) - beta
                quantized = quantized.clamp(0, qmax)
                
                # Flatten the quantized weights
                quantized_flat = quantized.view(child.out_features, child.in_features)
                
                # Create True Integer Packed layer dynamically
                real_int_linear = WXA16Linear(child.in_features, child.out_features, group_size, bits=child.bits, bias=(child.bias is not None))
                
                real_int_linear.weight_packed.copy_(pack_weights_to_intX(quantized_flat, child.bits))
                real_int_linear.scales.copy_(safe_alpha_group.half())
                real_int_linear.zeros.copy_(beta.half())
                
                if child.bias is not None:
                    real_int_linear.bias.copy_(child.bias.half())
                
            setattr(module, name, real_int_linear)
            
        else:
            merge_and_pack_linears(child)



def calc_original_outputs(pipe: DiTPipeline,timesteps: list[torch.Tensor],class_id: list[torch.Tensor], timestep_hidden_states: dict, batch_size: int) -> dict[ str, dict[str, dict[str, torch.Tensor]]]:
    original_outputs = {} 
    
    with torch.no_grad():
        for c in class_id:
            c_value = c.item()
            c_batch = torch.tensor([c_value] * batch_size, device=pipe.device)
            for t in timesteps:
                t_value = t.item()
                latents_copy = timestep_hidden_states[t_value].clone().detach()
                t_batch = torch.tensor([t_value] * batch_size, device=pipe.device, dtype=torch.float32)
                
                for layer_id, layer in enumerate(pipe.transformer.transformer_blocks):
                    latents_out = layer(latents_copy, timestep=t_batch, class_labels=c_batch)
                    original_outputs[(c_value, t_value, layer_id)] = latents_out.clone().detach()

    return original_outputs
                    


def apply_sliderquant(pipe: DiTPipeline,device: str,timesteps: list[torch.Tensor], window_list: list, layer_shallow: int, layer_int: int, gamma: float,epoch_num: int,class_num: int,bits_int: int,bits_ext: int, rank: int, group_size: int, batch_size: int) -> DiTPipeline:
    latents_x0 = torch.randn((batch_size, 4, 32, 32), device=device, dtype=torch.float16)
    class_steps = int(1000/class_num)
    class_id = [torch.tensor([i],device=device) for i in range(0,1000,class_steps)]
    
    skip_names = ["norm1", "emb"] # Vediamo se globale
    
    timestep_hidden_states = {}
    with torch.no_grad():
        for t in timesteps:
            t_value = t.item()
            noise = torch.randn_like(latents_x0)
            t_tensor = torch.tensor([t_value] * batch_size, device=device)
            noisy_latents = pipe.scheduler.add_noise(latents_x0, noise, t_tensor).to(latents_x0.dtype)
            timestep_hidden_states[t_value] = pipe.transformer.pos_embed(noisy_latents).clone().detach()
    
    original_outputs = calc_original_outputs(pipe, timesteps, class_id, timestep_hidden_states, batch_size)
    
    total_start_time = time.time()
    
    # === STAGE: WINDOW DISTILLATION ===
    print("=== STAGE: WINDOW DISTILLATION ===")
    for window_id, window in enumerate(window_list):
        print(f"===================================")
        print(f"Window {window_id}/{len(window_list)-1}: {window}")
        linear_modules = []
        opt_parameters = []

        for layer_id in window:
            layer_module = pipe.transformer.transformer_blocks[layer_id]
            layer_module.float()
            
            if layer_id < layer_shallow or layer_id >= (layer_shallow + layer_int):
                print(f"Layer {layer_id}: SHALLOW/DEEP -> Quantized in {bits_ext}-bit")
                linear_modules += replace_linears_with_sliderquant(layer_module, bits_ext, rank, gamma, group_size, skip_names)
            else:
                print(f"Layer {layer_id}: INTERMEDIATE -> Quantized in {bits_int}-bit")
                linear_modules += replace_linears_with_sliderquant(layer_module, bits_int, rank, gamma, group_size, skip_names)

        for sq_mod in linear_modules:
            if isinstance(sq_mod, SliderQuantLinear):
                opt_parameters.extend([sq_mod.alpha, sq_mod.A, sq_mod.B])

        if not opt_parameters:
            for layer_id in window:
                pipe.transformer.transformer_blocks[layer_id].half()
            continue

        optimizer = torch.optim.Adam(opt_parameters, lr=1e-4)
        print(f"Found {len(linear_modules)} quantizable linear layers in the window.")
        for g_id, g in enumerate([gamma, 1.0]):
            print(f"  --- Phase {g_id+1}/2 window with gamma = {g}")
            for sq_mod in linear_modules:
                sq_mod.gamma = g
            
            window_inputs_cache = {}
            with torch.no_grad():
                for c_val in class_id:
                    c = torch.tensor([c_val.item()] * batch_size, device=device)
                    for t_id, t in enumerate(timesteps):
                        t_value = t.item()
                        
                        base_latents = timestep_hidden_states[t_value].clone()
                            
                        latents_copy = base_latents
                        for prev_layer_id in range(window[0]):
                            prev_layer = pipe.transformer.transformer_blocks[prev_layer_id]
                            prev_dtype = next(prev_layer.parameters()).dtype
                            latents_copy = latents_copy.to(prev_dtype)
                            t_tensor = torch.tensor([t.item()] * batch_size, device=device, dtype=prev_dtype)
                            latents_copy = prev_layer(latents_copy, timestep=t_tensor, class_labels=c)
                            
                        window_inputs_cache[(c_val.item(), t_value)] = latents_copy.detach().half()

            for current_epoch in range(epoch_num):   
                optimizer.zero_grad()
                epoch_loss = 0 
                for c_val in class_id:
                    c = torch.tensor([c_val.item()] * batch_size, device=device)
                    for t_id, t in enumerate(timesteps):
                        t_value = t.item()
                        latents_copy = window_inputs_cache[(c_val.item(), t_value)].clone().float()

                        for layer_id in window:
                            layer = pipe.transformer.transformer_blocks[layer_id]
                            t_tensor = torch.tensor([t.item()] * batch_size, device=device, dtype=torch.float32)
                            latents_copy = layer(latents_copy, timestep=t_tensor, class_labels=c)
                    
                        target_output = original_outputs[(c_val.item(), t_value, window[-1])].to(device)
                        
                        loss = F.mse_loss(latents_copy.float(), target_output.float())
                        scaled_loss = loss / len(timesteps)
                        scaled_loss.backward()
                        epoch_loss += loss.item()
                
                print(f"    Window {window_id} - Epoch {current_epoch+1}/{epoch_num} completed | Average Loss: {epoch_loss/class_num:.6f}")
                torch.nn.utils.clip_grad_norm_(opt_parameters, max_norm=1.0)
                optimizer.step()

        for layer_id in window:
            pipe.transformer.transformer_blocks[layer_id].half()

    print("=== PACKING E MERGING WXA16 ===")
    merge_and_pack_linears(pipe.transformer)
    
    total_end_time = time.time()
    print(f"Total OPTIMIZATION time (full): {total_end_time - total_start_time:.2f} seconds")

    return pipe

if __name__=="__main__":
    pass
