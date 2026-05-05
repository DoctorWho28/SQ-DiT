import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline, AutoencoderKL
from datasets import load_dataset
from torchvision import transforms

def hessian_guided_loss(quantized_activations, fp_activations, hessian_weights):
    # Calcola l'errore tra i valori originali (fp_activations) e quelli a bassa precisione.
    squared_error = (quantized_activations - fp_activations) ** 2
    # Invece di una media semplice, moltiplica l'errore per il peso dell'Hessiano.
    # Questo dice all'ottimizzatore: "Se questo neurone è molto importante per la loss finale, 
    # fai in modo che il suo errore di quantizzazione sia il più piccolo possibile".
    weighted_error = hessian_weights.expand_as(squared_error) * squared_error
    return weighted_error.mean()

class RoundSTE(torch.autograd.Function):
    # L'operazione di arrotondamento (round) non ha una derivata matematica calcolabile.
    # Questo bloccherebbe il processo di addestramento (backpropagation).
    # Questa classe implementa uno "Straight-Through Estimator": arrotonda i valori all'andata (forward),
    # ma fa finta che non ci sia stato alcun arrotondamento al ritorno (backward), passando il gradiente intatto.
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MRQQuantizer(nn.Module):
    def __init__(self, bit_width=8, is_softmax=True):
        super().__init__()
        self.bit_width = bit_width
        self.is_softmax = is_softmax
        # Calcola il valore massimo rappresentabile con i bit scelti (es. 127 per 8-bit).
        self.qmax = 2 ** (bit_width - 1) - 1

        # Definisce i parametri di scala (s1, s2) da imparare durante l'ottimizzazione.
        if is_softmax:
            # La Softmax ha molti valori piccolissimi. Usa un parametro s1 da imparare per i valori piccoli,
            # e un gradino fisso per i valori grandi.
            self.s1 = nn.Parameter(torch.tensor(0.001))
            self.s2 = None
        else:
            # Il GELU ha valori negativi asimmetrici rispetto ai positivi. 
            # Impara due parametri separati: s1 per i negativi, s2 per i positivi.
            self.s1 = nn.Parameter(torch.tensor(0.01))
            self.s2 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        ste = RoundSTE.apply
        if self.is_softmax:
            threshold = self.qmax * self.s1
            
            # Crea maschere binarie per separare i valori sotto o sopra la soglia.
            mask_small = (x < threshold).float()
            mask_large = (x >= threshold).float()
            
            # Applica la quantizzazione fine ai valori piccoli usando s1.
            x_small = torch.clamp(ste(x / self.s1), -self.qmax, self.qmax) * self.s1
            # Usa una spaziatura fissa per i valori grandi.
            fixed_step = 1.0 / self.qmax
            x_large = torch.clamp(torch.round(x / fixed_step), -self.qmax, self.qmax) * fixed_step
            
            # Ricombina le due regioni.
            return x_small * mask_small + x_large * mask_large
        else:
            mask_neg = (x < 0).float()
            mask_pos = (x >= 0).float()
            
            # Quantizza i valori negativi con s1 e i positivi con s2.
            x_neg = torch.clamp(ste(x / self.s1), -self.qmax, self.qmax) * self.s1
            x_pos = torch.clamp(ste(x / self.s2), -self.qmax, self.qmax) * self.s2
            
            return x_neg * mask_neg + x_pos * mask_pos

def optimize_mrq_parameters_ho(fp_activations, hessian_weights, quantizer_module, iterations=750, lr=1e-3):
    # Questa funzione trova i valori ottimali di s1 (e s2) per il MRQQuantizer.
    optimizer = torch.optim.Adam(quantizer_module.parameters(), lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad()
        # Calcola la loss guidata dall'Hessiano vista prima.
        loss = hessian_guided_loss(quantizer_module(fp_activations), fp_activations, hessian_weights)
        loss.backward()
        optimizer.step()
        
        # Evita che i parametri di scala diventino zero o negativi, il che causerebbe divisioni per zero.
        with torch.no_grad():
            quantizer_module.s1.clamp_(min=1e-5)
            if not quantizer_module.is_softmax:
                quantizer_module.s2.clamp_(min=1e-5)
    return loss.item()

NUM_STEPS = 50
NUM_GROUPS = 10
NUM_IMAGES = 32

current_group = "Group_0"
ts_to_group = {}
activation_tensor_cache = {}
hessian_tensor_cache = {}

def setup_time_buckets(scheduler, num_steps, num_groups):
    # Implementa il TGQ: raggruppa i timestep in G blocchi contigui.
    # Associa ad ogni step temporale un'etichetta (es. "Group_0", "Group_1").
    scheduler.set_timesteps(num_steps)
    steps_per_group = num_steps // num_groups
    return {
        float(ts): f"Group_{min(i // steps_per_group, num_groups - 1)}"
        for i, ts in enumerate(scheduler.timesteps.tolist())
    }

def update_current_group(t_tensor):
    # Permette di sapere in quale gruppo temporale ci troviamo attualmente durante il passaggio dei dati.
    global current_group
    ts_val = float(t_tensor[0].item())
    if ts_val in ts_to_group:
        current_group = ts_to_group[ts_val]
    else:
        current_group = ts_to_group[min(ts_to_group, key=lambda k: abs(k - ts_val))]

def get_caching_hook(layer_name):
    # "Spia" la rete: durante il passaggio in avanti (forward), cattura le attivazioni originali (alta precisione).
    # Le salva raggruppandole per il gruppo temporale in cui ci troviamo.
    def hook(module, input_tensor, output_tensor):
        act = (output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor)
        act = act.detach().clone().float().cpu()
        activation_tensor_cache.setdefault(current_group, {})
        activation_tensor_cache[current_group].setdefault(layer_name, [])
        activation_tensor_cache[current_group][layer_name].append(act)
    return hook

def get_backward_hook(layer_name):
    # "Spia" la rete: durante il ritorno (backward), cattura i gradienti al quadrato.
    # Nel paper, la diagonale della Fisher Information Matrix (usata per approssimare l'Hessiano)
    # si calcola proprio facendo la media dei gradienti al quadrato.
    def hook(module, grad_input, grad_output):
        grad_sq = (grad_output[0].detach().clone().float()) ** 2 + 1e-6
        hessian_tensor_cache.setdefault(current_group, {})
        hessian_tensor_cache[current_group].setdefault(layer_name, [])
        hessian_tensor_cache[current_group][layer_name].append(grad_sq.cpu())
    return hook

def register_hooks(pipe):
    # Inserisce le "spie" create sopra nei layer critici del Transformer (Softmax e GELU).
    hooks = []
    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        if idx == 0:
            hooks.append(block.norm1.emb.time_proj.register_forward_hook(
                lambda m, i, o: update_current_group(i[0])
            ))

        attn_name = f"Block{idx}_Attention_PostSoftmax"
        hooks.append(block.attn1.register_forward_hook(get_caching_hook(attn_name)))
        hooks.append(block.attn1.register_full_backward_hook(get_backward_hook(attn_name)))

        gelu_name = f"Block{idx}_FeedForward_PostGELU"
        hooks.append(block.ff.net[0].register_forward_hook(get_caching_hook(gelu_name)))
        hooks.append(block.ff.net[0].register_full_backward_hook(get_backward_hook(gelu_name)))

    return hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

def build_calibration_latents(pipe, vae, device, num_images=NUM_IMAGES):
    # Prepara i dati reali per calibrare il modello. Il DiT non lavora su pixel, ma su rappresentazioni 
    # compresse (latenti) create da un VAE. Qui trasformiamo le immagini di ImageNet nel formato corretto.
    try:
        dataset = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)
        samples = list(dataset.take(num_images))
    except Exception as e:
        return None, None

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    latents_list = []
    class_ids_list = []

    vae.eval()
    with torch.no_grad():
        for sample in samples:
            img = preprocess(sample["image"].convert("RGB")).unsqueeze(0).to(device)
            # Moltiplica per 0.18215, un fattore di scala standard per questo tipo di architetture VAE.
            latent = vae.encode(img.to(torch.float32)).latent_dist.sample() * 0.18215
            latents_list.append(latent.to(torch.float16))
            class_ids_list.append(sample["label"])

    return latents_list, class_ids_list

def q_sample(x0, t_index, alphas_cumprod, noise, device):
    # Simula il processo di diffusione ("forward process"): prende l'immagine pulita e aggiunge 
    # una quantità di rumore che dipende dallo step temporale corrente, secondo le formule del paper.
    alpha_bar = alphas_cumprod[t_index].to(device)
    sqrt_ab = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_1mab * noise

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    global ts_to_group
    ts_to_group = setup_time_buckets(pipe.scheduler, NUM_STEPS, NUM_GROUPS)

    pipe.scheduler.set_timesteps(NUM_STEPS)
    alphas_cumprod = pipe.scheduler.alphas_cumprod

    latents_x0, class_ids = build_calibration_latents(pipe, vae, device, NUM_IMAGES)
    use_real_data = (latents_x0 is not None)

    # FASE 1: RACCOLTA DATI
    # Qui il modello gira in avanti per raccogliere le attivazioni, e calcola i gradienti (backward)
    # per avere i dati necessari alla matrice Hessiana. Non stiamo addestrando i pesi della rete,
    # stiamo solo collezionando statistiche.
    pipe.transformer.train()
    hooks = register_hooks(pipe)

    for img_idx in range(NUM_IMAGES):
        class_id = class_ids[img_idx] if use_real_data else (img_idx % 1000)
        current_class = torch.tensor([class_id], device=device)

        if use_real_data:
            x0 = latents_x0[img_idx].to(device)
        else:
            generator = torch.Generator(device=device).manual_seed(img_idx)
            x0 = torch.randn((1, pipe.transformer.config.in_channels, 32, 32), generator=generator, device=device, dtype=torch.float16)

        pipe.scheduler.set_timesteps(NUM_STEPS)

        for step_idx, t in enumerate(pipe.scheduler.timesteps):
            t_val = int(t.item())
            t_tensor = torch.tensor([t_val], dtype=torch.long, device=device)
            update_current_group(t_tensor)

            with torch.enable_grad():
                if use_real_data:
                    epsilon_true = torch.randn_like(x0)
                    
                    # Genera l'immagine rumorosa
                    x_t = q_sample(x0.float(), t_val, alphas_cumprod, epsilon_true.float(), device).to(torch.float16)
                    x_t = x_t.detach().clone()
                    x_t.requires_grad_(True)

                    # Chiede al modello di prevedere il rumore aggiunto
                    model_output = pipe.transformer(x_t, timestep=t_tensor, class_labels=current_class).sample
                    epsilon_pred, _ = model_output.chunk(2, dim=1)

                    # Calcola l'errore tra il rumore vero e quello previsto
                    loss = F.mse_loss(epsilon_pred, epsilon_true.to(torch.float16))
                else:
                    x_t = x0.detach().clone()
                    x_t.requires_grad_(True)
                    model_output = pipe.transformer(x_t, timestep=t_tensor, class_labels=current_class).sample
                    epsilon_pred, _ = model_output.chunk(2, dim=1)
                    loss = F.mse_loss(epsilon_pred, torch.randn_like(epsilon_pred))

                # Questo innesca gli hook backward registrati prima per catturare i gradienti quadrati
                loss.backward()

            torch.cuda.empty_cache()

    remove_hooks(hooks)
    pipe.transformer.eval()

    # FASE 2: OTTIMIZZAZIONE
    # Ora che abbiamo raccolto come si comporta la rete ad alta precisione e quali neuroni sono più
    # sensibili agli errori, cerchiamo i parametri di scala s1 e s2 migliori per ogni gruppo temporale
    # e per ogni layer.
    export_data = {}

    for group_name, layers in activation_tensor_cache.items():
        export_data[group_name] = {}

        for layer_name, tensor_list in layers.items():
            is_softmax = "PostSoftmax" in layer_name
            quantizer = MRQQuantizer(bit_width=8, is_softmax=is_softmax).to(device)

            fp_tensor = torch.cat(tensor_list, dim=0).to(device)

            hessian_list = hessian_tensor_cache.get(group_name, {}).get(layer_name, [])
            if not hessian_list:
                continue

            # Crea il peso finale dell'Hessiano normalizzandolo
            hessian_weights = (torch.stack(hessian_list, dim=0).mean(dim=0, keepdim=True).to(device))
            hessian_weights = hessian_weights / (hessian_weights.max() + 1e-8)

            # Avvia la ricerca dei parametri ottimali
            final_loss = optimize_mrq_parameters_ho(fp_tensor, hessian_weights, quantizer, iterations=750, lr=1e-3)

            s1_val = quantizer.s1.item()
            s2_val = quantizer.s2.item() if not is_softmax else None

            # FASE 3: SALVATAGGIO DEI PARAMETRI
            export_data[group_name][layer_name] = {
                "layer_type" : "PostSoftmax" if is_softmax else "PostGELU",
                "bit_width" : quantizer.bit_width,
                "calibration_data" : "ImageNet real" if use_real_data else "synthetic",
                "final_optimization_loss" : round(final_loss, 6),
                "scale_factor_s1" : round(s1_val, 6),
                "scale_factor_s2" : round(s2_val, 6) if s2_val else "N/A (Fixed Step)",
            }

    output_file = "tq_dit_optimized_parameters.json"
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=4)

if __name__ == "__main__":
    main()