import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline

# --- 1. CLASSI E FUNZIONI DI QUANTIZZAZIONE ---

def hessian_guided_loss(quantized_activations, fp_activations, hessian_weights):
    """
    Calcola la loss di quantizzazione pesata secondo l'approssimazione 
    dell'Hessiano (Diagonale di Fisher), come descritto nell'Eq. 15 del paper.
    """
    # 1. Calcoliamo l'errore standard al quadrato (la differenza)
    squared_error = (quantized_activations - fp_activations) ** 2
    
    # 2. Moltiplichiamo l'errore per i "pesi" dell'Hessiano
    # hessian_weights deve avere la stessa esatta forma di fp_activations
    weighted_error = hessian_weights * squared_error
    
    # 3. Ritorniamo il valore atteso (la media)
    return weighted_error.mean()

# --- NUOVA CLASSE STE (Straight-Through Estimator) ---
class RoundSTE(torch.autograd.Function):
    """
    Inganna PyTorch: durante il forward fa un normale arrotondamento,
    ma durante il backward lascia passare il gradiente intatto.
    Questo permette di ottimizzare i fattori di scala s1 e s2.
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Passiamo il gradiente all'indietro senza bloccarlo
        return grad_output

# --- CLASSE MRQ AGGIORNATA ---
class MRQQuantizer(nn.Module):
    def __init__(self, bit_width=8, is_softmax=True):
        super().__init__()
        self.bit_width = bit_width
        self.is_softmax = is_softmax
        
        # Calcoliamo il valore massimo rappresentabile.
        self.qmax = 2**(bit_width - 1) - 1 # per 8-bit è 127
        
        # --- MODIFICA CRITICA: Inizializzazione Intelligente ---
        if is_softmax:
            # La Softmax va da 0 a 1. Un buon gradino di partenza è 1 diviso i gradini totali
            # 1 / 127 = ~0.007. Partiamo da un valore ancora più piccolo per la regione R1.
            self.s1 = nn.Parameter(torch.tensor(0.001))
            self.s2 = None 
        else:
            # La GELU (Fig 2b del paper) ha valori negativi fino a -2 e positivi fino a ~20.
            # s1 gestisce i negativi (spazio piccolo, serve precisione)
            # s2 gestisce i positivi (spazio ampio)
            self.s1 = nn.Parameter(torch.tensor(0.01))
            self.s2 = nn.Parameter(torch.tensor(0.1))
            
    def forward(self, x):
        round_ste = RoundSTE.apply

        if self.is_softmax:
            threshold = self.qmax * self.s1
            mask_small = (x < threshold).float()
            mask_large = (x >= threshold).float()
            
            # AGGIUNTA FONDAMENTALE: Il Clamp!
            # Limitiamo il valore arrotondato tra -qmax e qmax
            x_small_int = torch.clamp(round_ste(x / self.s1), -self.qmax, self.qmax)
            x_small = x_small_int * self.s1
            
            fixed_step = 1.0 / self.qmax
            x_large_int = torch.clamp(torch.round(x / fixed_step), -self.qmax, self.qmax)
            x_large = x_large_int * fixed_step
            
            x_q = x_small * mask_small + x_large * mask_large
        else:
            mask_neg = (x < 0).float()
            mask_pos = (x >= 0).float()
            
            # AGGIUNTA FONDAMENTALE: Il Clamp!
            x_neg_int = torch.clamp(round_ste(x / self.s1), -self.qmax, self.qmax)
            x_neg = x_neg_int * self.s1
            
            x_pos_int = torch.clamp(round_ste(x / self.s2), -self.qmax, self.qmax)
            x_pos = x_pos_int * self.s2
            
            x_q = x_neg * mask_neg + x_pos * mask_pos
            
        return x_q
# Aggiungiamo 'hessian_weights' tra gli argomenti
def optimize_mrq_parameters_ho(fp_activations, hessian_weights, quantizer_module, iterations=50, lr=1e-2):
    optimizer = torch.optim.Adam(quantizer_module.parameters(), lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass del quantizzatore
        quantized_activations = quantizer_module(fp_activations)
        
        # --- SOSTITUZIONE RIGA 71 ---
        # Usiamo l'Hessiano al posto di F.mse_loss
        loss = hessian_guided_loss(quantized_activations, fp_activations, hessian_weights)
        
        # Backward e step di ottimizzazione dei parametri di quantizzazione (s1, s2)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            quantizer_module.s1.clamp_(min=1e-5)
            if not quantizer_module.is_softmax:
                quantizer_module.s2.clamp_(min=1e-5)
                
    return loss.item()

# --- 2. VARIABILI GLOBALI E SETUP TGQ ---

NUM_STEPS = 50
NUM_GROUPS = 10
NUM_IMAGES = 32 
current_group = "Group_0"
ts_to_group = {}
activation_tensor_cache = {}

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
    global current_group
    timestep = input[0]
    ts_val = float(timestep[0].item())
    if ts_val in ts_to_group:
        current_group = ts_to_group[ts_val]
    else:
        closest_ts = min(ts_to_group.keys(), key=lambda k: abs(k - ts_val))
        current_group = ts_to_group[closest_ts]

def get_caching_hook(layer_name):
    def hook(module, input_tensor, output_tensor):
        global current_group
        
        if isinstance(output_tensor, tuple):
            # .cpu() sposta il dato sulla RAM di sistema per non esaurire la GPU
            attivazioni = output_tensor[0].detach().clone().float().cpu()
        else:
            attivazioni = output_tensor.detach().clone().float().cpu()
            
        if current_group not in activation_tensor_cache:
            activation_tensor_cache[current_group] = {}
            
        # Invece di sommare, creiamo una LISTA di tensori
        if layer_name not in activation_tensor_cache[current_group]:
            activation_tensor_cache[current_group][layer_name] = [attivazioni]
        else:
            activation_tensor_cache[current_group][layer_name].append(attivazioni)
            
    return hook

hessian_tensor_cache = {}

def get_backward_hook(layer_name):
    def hook(module, grad_input, grad_output):
        global current_group
        
        # Il paper usa la "diagonal Fisher information matrix" per approssimare l'Hessiano.
        # Questa è letteralmente il quadrato dei gradienti in uscita dal layer.
        # Aggiungiamo un piccolo valore (1e-6) per evitare pesi a zero assoluto.
        grad_squared = (grad_output[0].detach().clone().float()) ** 2 + 1e-6
        
        if current_group not in hessian_tensor_cache:
            hessian_tensor_cache[current_group] = {}
            
        if layer_name not in hessian_tensor_cache[current_group]:
            hessian_tensor_cache[current_group][layer_name] = grad_squared
        else:
            # Accumuliamo per fare una media su più immagini
            hessian_tensor_cache[current_group][layer_name] += grad_squared
            
    return hook
# --- 3. CODICE BASE (ORCHESTRAZIONE) ---

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Esecuzione su dispositivo: {device}")
    
    model_id = "facebook/DiT-XL-2-256"
    print(f"Caricamento del modello {model_id} in FP16...")
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    global ts_to_group
    ts_to_group = setup_time_buckets(pipe.scheduler, NUM_STEPS, NUM_GROUPS)
    
    block = pipe.transformer.transformer_blocks[0]
    block.norm1.emb.time_proj.register_forward_hook(trace_timestep_hook)
    block.attn1.to_q.register_forward_hook(get_caching_hook('Attention_Q_PostSoftmax'))
    block.ff.net[0].proj.register_forward_hook(get_caching_hook('FeedForward_PostGELU'))

    # --- NUOVO: Registriamo gli hook per intercettare l'Hessiano (i gradienti) ---
    # Usiamo register_full_backward_hook per sicurezza con le ultime versioni di PyTorch
    block.attn1.to_q.register_full_backward_hook(get_backward_hook('Attention_Q_PostSoftmax'))
    block.ff.net[0].proj.register_full_backward_hook(get_backward_hook('FeedForward_PostGELU'))

    # --- FASE 1: RACCOLTA DATI CON CALCOLO HESSIANO ---
    print(f"\n--- FASE 1: Calibrazione con Calcolo Gradienti (HO) ---")
    
    # Mettiamo il transformer in modalità training per permettere ai gradienti di fluire
    pipe.transformer.train() 
    
    for seed in range(NUM_IMAGES):
        print(f"Generazione immagine e calcolo Hessiano {seed + 1}/{NUM_IMAGES}...")
        generator = torch.Generator(device=device).manual_seed(seed)
        
        random_class_id = seed % 1000 
        current_class = torch.tensor([random_class_id], device=device)
        print(f"  -> Calibrazione sulla classe ImageNet: {random_class_id}")
        
        # 1. Creiamo un "latente" di partenza puro rumore (come fa il DiT normalmente)
        # 32x32 è la dimensione latente standard per un'immagine 256x256 nel DiT
        latents = torch.randn(
            (1, pipe.transformer.config.in_channels, 32, 32), 
            generator=generator, device=device, dtype=torch.float16
        )
        
        pipe.scheduler.set_timesteps(NUM_STEPS)
        
        for t in pipe.scheduler.timesteps:
            # Sincronizziamo il timer globale per gli hook
            trace_timestep_hook(None, [torch.tensor([t])], None)
            
            # --- IL CUORE DELL'HO: ABILITIAMO I GRADIENTI ---
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            t_tensor = torch.tensor([t_val], dtype=torch.long, device=device)
            trace_timestep_hook(None, [t_tensor], None)
            
            # --- IL CUORE DELL'HO: ABILITIAMO I GRADIENTI ---
            with torch.enable_grad():
                # Diciamo a PyTorch che vogliamo tracciare le operazioni su questo latente
                latent_model_input = latents.detach().clone()
                latent_model_input.requires_grad_(True)
                
                # Forward pass manuale (fa scattare i forward_hook)
                model_output = pipe.transformer(
                    latent_model_input, 
                    timestep=t_tensor, 
                    class_labels=current_class
                ).sample
                
                noise_pred, _ = model_output.chunk(2, dim=1)
                
                # Creiamo una Loss fittizia per far fluire i gradienti
                dummy_target_noise = torch.randn_like(noise_pred)
                loss = F.mse_loss(noise_pred, dummy_target_noise)
                
                # BACKWARD PASS! (Fa scattare i backward_hook e popola hessian_tensor_cache)
                loss.backward()
            
            # Passiamo allo step successivo rimuovendo il rumore (modalità inferenza normale)
            with torch.no_grad():
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # PULIZIA MEMORIA ESTREMA: senza questo, la scheda video esaurirà la VRAM in 3 step.
            torch.cuda.empty_cache()

    # --- FASE 2: OTTIMIZZAZIONE MRQ + HO ---
    print("\n--- FASE 2: Ottimizzazione Quantizzazione MRQ ---")
    
    # Dizionario strutturato per il salvataggio in JSON
    export_data = {}
    
    for group_name, layers in activation_tensor_cache.items():
        print(f"\nOttimizzazione per il bucket: {group_name}")
        export_data[group_name] = {}
        
        for layer_name, tensor_list in layers.items():
            is_softmax_layer = "Softmax" in layer_name
            
            quantizer = MRQQuantizer(bit_width=8, is_softmax=is_softmax_layer).to(device)
            
            # --- MODIFICA QUI ---
            # Uniamo l'intera lista di decine di tensori in un unico mega-tensore lungo la dimensione 0
            # e lo riportiamo sulla GPU (.to(device)) solo nel momento del bisogno
            fp_tensor = torch.cat(tensor_list, dim=0).to(device)
            
            # Recuperiamo l'Hessiano (lui era rimasto un singolo tensore sommato, va benissimo)
            hessian_weights = hessian_tensor_cache[group_name][layer_name].to(device)
            hessian_weights = hessian_weights / (hessian_weights.max() + 1e-8)
            
            # --- MODIFICATO: Chiamiamo la nuova funzione _ho ---
            final_loss = optimize_mrq_parameters_ho(
                fp_tensor, 
                hessian_weights, 
                quantizer, 
                iterations=750, 
                lr=0.001           
            )
            
            # Estraiamo i valori float dai tensori PyTorch per poterli salvare
            s1_val = quantizer.s1.item()
            s2_val = quantizer.s2.item() if not is_softmax_layer else None
            
            # Popoliamo il dizionario di export
            export_data[group_name][layer_name] = {
                "layer_type": "Softmax" if is_softmax_layer else "GELU",
                "bit_width": quantizer.bit_width,
                "final_optimization_loss": round(final_loss, 6),
                "scale_factor_s1": round(s1_val, 6),
                "scale_factor_s2": round(s2_val, 6) if s2_val is not None else "N/A (Fixed Step)"
            }
            
            if is_softmax_layer:
                print(f"  [{layer_name}] s1: {s1_val:.4f} | Loss: {final_loss:.4f}")
            else:
                print(f"  [{layer_name}] s1: {s1_val:.4f}, s2: {s2_val:.4f} | Loss: {final_loss:.4f}")

    # --- FASE 3: SALVATAGGIO DEI DATI ---
    output_filename = "tq_dit_optimized_parameters.json"
    with open(output_filename, "w") as f:
        json.dump(export_data, f, indent=4)
        
    print(f"\n[SUCCESSO] Tutti i parametri sono stati salvati in '{output_filename}'. Puoi aprirlo con un editor di testo per analizzarli.")

if __name__ == "__main__":
    main()