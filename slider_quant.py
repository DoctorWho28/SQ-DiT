from asyncio.windows_utils import pipe
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline
from safetensors.torch import save_file, load_file

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def sliding_quantize_tensors(tensor_list: list[torch.Tensor], bits: int=4) -> list[torch.Tensor]:
    '''
    Apply the quantization of a group of tensors all together, given as parameters:
    - tensor_list: the list of tensor to quantize
    - bits: number of bits to quantize to
    '''

    result_list = []
    qmax = (2 ** bits) - 1

    for original_tensor in tensor_list:
        # FONDAMENTALE: Quantizzazione per-canale (per riga). 
        # Se usiamo il min/max dell'intero tensore, i pesi vengono distrutti!
        zmin = original_tensor.min(dim=1, keepdim=True)[0]
        zmax = original_tensor.max(dim=1, keepdim=True)[0]

        alpha = (zmax - zmin) / qmax
        
        # Evitiamo la divisione per zero dove zmax == zmin
        safe_alpha = torch.where(alpha == 0, torch.ones_like(alpha), alpha)
        
        beta = torch.round(zmin / safe_alpha)

        quantized_slice = RoundSTE.apply(original_tensor / safe_alpha) - beta
        quantized_slice = quantized_slice.clamp(0, qmax)
        dequantized_slice = (quantized_slice + beta) * safe_alpha
        
        # Ripristiniamo intatti i canali che avevano zmax == zmin
        dequantized_slice = torch.where((zmax == zmin), original_tensor, dequantized_slice)
        
        result_list.append(dequantized_slice)

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
            c_batch = torch.tensor([c_value] * latents.shape[0], device=latents.device)
            for t in timesteps:
                t_value = t.item()
                latents_copy = hidden_states.clone().detach()
                t_batch = torch.tensor([t_value] * latents.shape[0], device=latents.device, dtype=torch.float32)
                
                for layer_id, layer in enumerate(pipe.transformer.transformer_blocks):
                    latents_copy = layer(latents_copy, timestep=t_batch, class_labels=c_batch).clone().detach()

                    key = f"class_{c_value}_time_{t_value}_layer_{layer_id}"
                    original_outputs[key] = latents_copy.clone().detach()

    return original_outputs

class SliderQuantLinear(nn.Module):
    """
    Wrapper per nn.Linear che implementa SliderQuant (Eq 2 del paper).
    Introduce Channel Scaling (alpha) e LoRA (A, B) come parametri apprendibili.
    """
    def __init__(self, original_linear: nn.Linear, bits: int, rank: int, gamma: float=1.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        self.rank = rank
        self.gamma = gamma
        
        # Pesi originali (congelati)
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        
        orig_device = original_linear.weight.device
        
        # Inizializziamo i parametri in float32 per prevenire overflow dei gradienti!
        self.alpha = nn.Parameter(torch.ones(self.in_features, dtype=torch.float32, device=orig_device))
        self.A = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=torch.float32, device=orig_device))
        self.B = nn.Parameter(torch.zeros(self.rank, self.in_features, dtype=torch.float32, device=orig_device))
        nn.init.normal_(self.A, std=0.01)

    def forward(self, x):
        orig_dtype = x.dtype
        # Cast input e pesi a float32 per la stabilità del backward pass
        x_f32 = x.float()
        w_f32 = self.weight.float()
        
        # 1. Scala gli input in modo sicuro
        # Limitiamo alpha per evitare divisioni per zero o amplificazioni estreme del contributo LoRA
        safe_alpha = torch.clamp(self.alpha, min=0.1, max=10.0)
        x_scaled = x_f32 / safe_alpha
        
        # 2. Modifica i pesi
        w_adjusted = w_f32 * safe_alpha.view(1, -1) + torch.matmul(self.A, self.B)
        
        # Applica quantizzazione parziale in base a gamma
        limit_row = int(self.out_features * self.gamma)
        
        if limit_row > 0:
            w_quant_part = sliding_quantize_tensors([w_adjusted[:limit_row, :]], self.bits)[0]
            if limit_row < self.out_features:
                w_final = torch.cat([w_quant_part, w_adjusted[limit_row:, :]], dim=0)
            else:
                w_final = w_quant_part
        else:
            w_final = w_adjusted

        # Esegui l'operazione lineare con i pesi quantizzati in float32
        bias_f32 = self.bias.float() if self.bias is not None else None
        out = F.linear(x_scaled, w_final, bias_f32)
        return out.to(orig_dtype)
    
def replace_linears_with_sliderquant(module, bits=4, rank=4, gamma=1.0, skip_names=["norm1", "emb"]):
    """
    Sostituisce ricorsivamente tutti i layer nn.Linear di un modulo 
    con i nostri SliderQuantLinear.
    """
    replaced_modules = []
    for name, child in module.named_children():
        # VERO MRQ: saltiamo completamente i layer sensibili (AdaLN/Embedders) mantenendoli in FP16!
        if any(skip in name for skip in skip_names):
            continue
            
        if isinstance(child, nn.Linear):
            sq_linear = SliderQuantLinear(child, bits=bits, rank=rank, gamma=gamma)
            setattr(module, name, sq_linear)
            replaced_modules.append(sq_linear)
        elif isinstance(child, SliderQuantLinear):
            replaced_modules.append(child)
        else:
            replaced_modules.extend(replace_linears_with_sliderquant(child, bits, rank, gamma, skip_names))
    return replaced_modules

def merge_and_restore_linears(module):
    """
    Fonde (merge) i parametri alpha, A, B nei pesi quantizzati definitivi
    e ripristina i layer nn.Linear standard per un salvataggio universale.
    """
    for name, child in module.named_children():
        if isinstance(child, SliderQuantLinear):
            # 1. Calcoliamo la matrice definitiva
            with torch.no_grad():
                # min 1/10 e max 10 volte rispetto al valore iniziale che è 1
                safe_alpha = torch.clamp(child.alpha, min=0.1, max=10.0)
                w_adjusted = child.weight * safe_alpha.view(1, -1) + (child.A @ child.B)
                
                limit_row = int(child.out_features * child.gamma)
                
                if limit_row > 0:
                    w_quant_part = sliding_quantize_tensors([w_adjusted[:limit_row, :]], child.bits)[0]
                    if limit_row < child.out_features:
                        w_final = torch.cat([w_quant_part, w_adjusted[limit_row:, :]], dim=0)
                    else:
                        w_final = w_quant_part
                else:
                    w_final = w_adjusted
            
            # 2. Ripristiniamo la scala matematica (SmoothQuant folding)
            # Nel forward facevamo x_scaled = x / safe_alpha.
            # Siccome il layer standard nn.Linear non dividerà più x per alpha,
            # dobbiamo ripiegare quella divisione dentro ai pesi finali!
            w_final_merged = w_final / safe_alpha.view(1, -1)
            
            # 3. Creiamo un layer lineare PyTorch standard
            standard_linear = nn.Linear(child.in_features, child.out_features, bias=(child.bias is not None))
            
            # 4. Mettiamo la nostra matrice quantizzata e fusa come peso ufficiale a 16-bit
            standard_linear.weight.data = w_final_merged.half()
            if child.bias is not None:
                standard_linear.bias.data = child.bias.half()
                
            # 4. Sostituiamo il layer custom con quello standard
            setattr(module, name, standard_linear)
            
        else:
            # Ricorsione per entrare in tutti i sottomoduli
            merge_and_restore_linears(child)

def apply_sliderquant(pipe: DiTPipeline,device: str,timesteps: list[torch.Tensor],layer_shallow: int,layer_int: int,layer_deep: int,window_size: int,window_step: int,gamma: float,epoch_num: int,class_num: int,bits: int, rank: int) -> DiTPipeline:
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
    - rank: rank of the low-rank approximation
    '''

    # AUMENTATO IL BATCH SIZE DA 1 A 4 PER MAGGIORE GENERALIZZAZIONE (Meno Overfitting)
    latents = torch.randn((4, 4, 32, 32), device=device, dtype=torch.float16)
    class_steps = int(1000/class_num)
    class_id = [torch.tensor([i],device=device) for i in range(0,1000,class_steps)]

    '''If already exists a json with the parameters, 
    load the original outputs and skip this function'''
    

    original_outputs = calc_original_outputs(pipe, timesteps, class_id, latents)
    
    # Calculate the windows for the algorithm
    window_list = calculate_window_index(layer_shallow, layer_int, layer_deep, window_size, window_step)

    latents_copy = latents.clone().detach()
    hidden_states = pipe.transformer.pos_embed(latents_copy)
    
    total_start_time = time.time()
    
    for window_id, window in enumerate(window_list):
        window_start_time = time.time()
        linear_modules = []
        #original_weights = []
        
        opt_parameters = []

        for layer_id in window:
            layer_module = pipe.transformer.transformer_blocks[layer_id]
            
            # Cast dell'intero blocco a float32 per stabilità numerica nel backward pass
            layer_module.float()
            
            # VERO MRQ Logic: Lasciamo Shallow e Deep in FP16 originale senza LoRA!
            if layer_id < layer_shallow or layer_id >= (layer_shallow + layer_int):
                print(f"Layer {layer_id}: SHALLOW/DEEP -> Quantized in 8-bit")
                linear_modules += replace_linears_with_sliderquant(layer_module, bits=8, rank=rank, gamma=gamma)
                continue
            
            # Inizializziamo i layer intermedi a 4-bit (saltando norm1)
            linear_modules += replace_linears_with_sliderquant(layer_module, bits=bits, rank=rank, gamma=gamma)

        for sq_mod in linear_modules:
            if isinstance(sq_mod, SliderQuantLinear):
                #original_weights.append(sq_mod.weight.data.clone().detach())
                opt_parameters.extend([sq_mod.alpha, sq_mod.A, sq_mod.B])

        if not opt_parameters:
            print(f"===================================")
            print(f"Window {window_id}: {window} contiene solo layer protetti in FP16. Salto addestramento!")
            continue

        optimizer = torch.optim.Adam(opt_parameters, lr=1e-4)
        print("===================================")
        print(f"Window {window_id}: {window},  number of linear modules: {len(linear_modules)}")
        for g_id, g in enumerate([gamma, 1.0]):
            
            for sq_mod in linear_modules:
                sq_mod.gamma = g
            
            print(f"--- Avvio Stage {g_id+1}/2 con gamma = {g}")
            
            # === PRE-CALCOLO INPUT DELLA FINESTRA ===
            # Calcoliamo una volta sola l'input della finestra corrente per tutti i timestep.
            # Questo elimina la necessità del dizionario gigante 'quantized_output' e risparmia tantissima RAM!
            window_inputs_cache = {}
            with torch.no_grad():
                for c_val in class_id:
                    c = torch.tensor([c_val.item()] * latents.shape[0], device=device)
                    for t_id, t in enumerate(timesteps):
                        t_value = t.item()
                        
                        # Input base del Transformer
                        if t_id > 0:
                            prev_t = timesteps[t_id-1].item()
                            base_latents = original_outputs[f"class_{c_val.item()}_time_{prev_t}_layer_{window_list[-1][-1]}"].clone()
                        else:
                            base_latents = hidden_states.clone().detach()
                            
                        # Propaghiamo in avanti solo fino al layer precedente alla finestra
                        latents_copy = base_latents.clone()
                        for prev_layer_id in range(window[0]):
                            prev_layer = pipe.transformer.transformer_blocks[prev_layer_id]
                            prev_dtype = next(prev_layer.parameters()).dtype
                            latents_copy = latents_copy.to(prev_dtype)
                            t_tensor = torch.tensor([t.item()] * latents.shape[0], device=device, dtype=prev_dtype)
                            latents_copy = prev_layer(latents_copy, timestep=t_tensor, class_labels=c)
                            
                        window_inputs_cache[f"{c_val.item()}_{t_value}"] = latents_copy.detach().half()
            # ========================================

            for current_epoch in range(epoch_num):   
                optimizer.zero_grad()
                epoch_loss = 0 


                for c_val in class_id:
                    c = torch.tensor([c_val.item()] * latents.shape[0], device=device)
                    for t_id, t in enumerate(timesteps):
                        t_value = t.item()
                        
                        # Prendiamo l'input pre-calcolato dalla cache
                        latents_copy = window_inputs_cache[f"{c_val.item()}_{t_value}"].clone().float()

                        for layer_id in window:
                            layer = pipe.transformer.transformer_blocks[layer_id]
                            t_tensor = torch.tensor([t.item()] * latents.shape[0], device=device, dtype=torch.float32)
                            latents_copy = layer(latents_copy, timestep=t_tensor, class_labels=c)
                    
                        last_layer = window[-1]
                        target_key = f"class_{c_val.item()}_time_{t_value}_layer_{last_layer}"
                        target_output = original_outputs[target_key].to(device)
                        
                        loss = F.mse_loss(latents_copy.float(), target_output.float())
                        scaled_loss = loss / len(timesteps)
                        scaled_loss.backward()
                        epoch_loss += loss.item()

                        
                print(f"Window: {window_id}, Epoch: {current_epoch+1}/{epoch_num}, Loss: {epoch_loss/class_num}") 
                
                torch.nn.utils.clip_grad_norm_(opt_parameters, max_norm=1.0)
                optimizer.step()

        window_end_time = time.time()
        print(f"Tempo per ottimizzare Window {window_id}: {window_end_time - window_start_time:.2f} secondi")

        # Reset weight after window finished
        for layer_id in window:
            pipe.transformer.transformer_blocks[layer_id].half()
    
    total_end_time = time.time()
    print(f"=====================================")
    print(f"Tempo TOTALE ottimizzazione: {total_end_time - total_start_time:.2f} secondi")
    print(f"=====================================")
    
    print("Salvataggio di emergenza pre-fusione (backup) in corso...")
    pipe.save_pretrained("output/output_unmerged_checkpoint", safe_serialization=True)

    print("Fondo i layer quantizzati in un'architettura standard per il salvataggio finale...")
    merge_and_restore_linears(pipe.transformer)

    return pipe
    
if __name__=="__main__":
    pass