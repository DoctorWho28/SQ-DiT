import json
import torch
import torch.nn as nn
from diffusers import DiTPipeline
import matplotlib.pyplot as plt

# --- 1. SETUP E FUNZIONI DI SUPPORTO ---

NUM_STEPS = 25
NUM_GROUPS = 5
current_group = "Group_0"
ts_to_group = {}

def setup_time_buckets(scheduler, num_steps, num_groups):
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.tolist()
    steps_per_group = num_steps // num_groups
    mapping = {}
    for i, ts in enumerate(timesteps):
        group_idx = min(i // steps_per_group, num_groups - 1)
        mapping[float(ts)] = f"Group_{group_idx}"
    return mapping

def trace_timestep_hook(module, input, output):
    """Aggiorna il gruppo temporale attuale durante l'inferenza"""
    global current_group
    timestep = input[0]
    ts_val = float(timestep[0].item())
    if ts_val in ts_to_group:
        current_group = ts_to_group[ts_val]
    else:
        closest_ts = min(ts_to_group.keys(), key=lambda k: abs(k - ts_val))
        current_group = ts_to_group[closest_ts]

# --- 2. QUANTIZZAZIONE DEI PESI (W8) ---

def apply_weight_quantization(model, bit_width=8):
    """
    Scorre tutto il modello e applica la Fake Quantization Uniforme 
    a tutti i pesi dei layer lineari (W8).
    """
    print("Quantizzazione dei pesi a 8-bit in corso...")
    qmax = 2**(bit_width - 1) - 1 # 127
    
    quantized_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Calcoliamo lo step size per i pesi (quantizzazione simmetrica min-max)
            w = module.weight.data
            max_val = w.abs().max()
            step_size = max_val / qmax
            
            # FAKE QUANTIZATION: Arrotondiamo al gradino e ricalcoliamo il valore in Float
            w_int = torch.clamp(torch.round(w / step_size), -qmax, qmax)
            w_quantized = w_int * step_size
            
            # Sostituiamo i pesi perfetti con quelli quantizzati "sporchi"
            module.weight.data = w_quantized
            quantized_layers += 1
            
    print(f"Quantizzati {quantized_layers} layer lineari.")

# --- 3. QUANTIZZAZIONE DELLE ATTIVAZIONI (A8 con JSON) ---

def get_mrq_inference_hook(layer_name, json_data):
    """
    Questo hook scatta ogni volta che un layer Softmax o GELU emette dati.
    Legge i parametri dal JSON e applica la quantizzazione MRQ.
    """
    def hook(module, input_tensor, output_tensor):
        global current_group
        
        # Gestione compatibilità output e SALVATAGGIO DEL DTYPE ORIGINALE
        x = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        original_dtype = x.dtype  # <--- NUOVO: Salviamo se era float16 o float32
        
        # Recuperiamo i parametri per il gruppo temporale ATTUALE e per il LAYER attuale
        if current_group in json_data and layer_name in json_data[current_group]:
            params = json_data[current_group][layer_name]
            s1 = params["scale_factor_s1"]
            s2 = params["scale_factor_s2"]
            qmax = 127 # 8-bit
            
            # (Tutta la matematica MRQ rimane invariata...)
            if params["layer_type"] == "Softmax":
                threshold = qmax * s1
                mask_small = (x < threshold).float()
                mask_large = (x >= threshold).float()
                
                x_small = torch.clamp(torch.round(x / s1), -qmax, qmax) * s1
                
                fixed_step = 1.0 / qmax
                x_large = torch.clamp(torch.round(x / fixed_step), -qmax, qmax) * fixed_step
                
                x_q = x_small * mask_small + x_large * mask_large
                
            else: # GELU
                mask_neg = (x < 0).float()
                mask_pos = (x >= 0).float()
                
                x_neg = torch.clamp(torch.round(x / s1), -qmax, qmax) * s1
                x_pos = torch.clamp(torch.round(x / s2), -qmax, qmax) * s2
                
                x_q = x_neg * mask_neg + x_pos * mask_pos
                
            # --- NUOVO: Riconvertiamo il tensore quantizzato al formato originale ---
            x_q = x_q.to(original_dtype)
                
            # Restituiamo il tensore
            if isinstance(output_tensor, tuple):
                return (x_q,) + output_tensor[1:]
            return x_q
            
        return output_tensor # Se non ci sono dati JSON, non fare nulla
    return hook


# --- 4. ORCHESTRAZIONE FINALE ---

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/DiT-XL-2-256"
    
    print("Caricamento JSON...")
    with open("tq_dit_optimized_parameters.json", "r") as f:
        quantization_data = json.load(f)

    print("Caricamento Pipeline originale...")
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    # 1. Quantizziamo staticamente i pesi W8
    apply_weight_quantization(pipe.transformer)

    # 2. Prepariamo i gruppi temporali per la TGQ
    global ts_to_group
    ts_to_group = setup_time_buckets(pipe.scheduler, NUM_STEPS, NUM_GROUPS)

    # 3. Attacchiamo gli Hook di Inferenza per le attivazioni A8 al primo blocco
    block = pipe.transformer.transformer_blocks[0]
    block.norm1.emb.time_proj.register_forward_hook(trace_timestep_hook)
    
    # Nota: Usiamo gli hook di FORWARD per modificare l'output mentre viaggia nella rete
    block.attn1.to_q.register_forward_hook(get_mrq_inference_hook('Attention_Q_PostSoftmax', quantization_data))
    block.ff.net[0].proj.register_forward_hook(get_mrq_inference_hook('FeedForward_PostGELU', quantization_data))

    print("\nInizio generazione immagine Quantizzata (W8A8)...")
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Poiché è inferenza reale, usiamo torch.no_grad()
    with torch.no_grad():
        image = pipe(
            class_labels=[0], # 0 potrebbe essere la classe per "Tench, Tinca" in ImageNet
            num_inference_steps=NUM_STEPS, 
            generator=generator
        ).images[0]

    # Salviamo l'immagine quantizzata
    image.save("quantized_dit_image.png")
    print("Finito! Immagine salvata come 'quantized_dit_image.png'.")
    
if __name__ == "__main__":
    main()