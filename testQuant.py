import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiTPipeline

# =============================================================================
# 1. CLASSI E FUNZIONI DI QUANTIZZAZIONE
# =============================================================================

def hessian_guided_loss(quantized_activations, fp_activations, hessian_weights):
    """
    Calcola la loss di quantizzazione pesata secondo l'approssimazione
    dell'Hessiano (Diagonale di Fisher), come descritto nell'Eq. 15 del paper.

    FIX (shape disallineata):
        fp_tensor ha shape [N*T, ...] (N immagini × T timestep concatenati),
        mentre hessian_weights originariamente aveva shape [1, ...] di un solo
        sample (accumulato con += su un tensore non in lista).
        Ora entrambi arrivano con shape coerente perché get_backward_hook()
        accumula anch'esso una lista e qui facciamo broadcast esplicito dopo
        aver fatto la media dell'Hessiano sulla dimensione batch.
    """
    squared_error = (quantized_activations - fp_activations) ** 2

    # hessian_weights può avere shape [1, C, H, W] (media già calcolata in Fase 2).
    # .expand_as() fa un broadcast sicuro e leggibile invece di affidarsi
    # al broadcasting implicito di PyTorch che mascherava il bug originale.
    weighted_error = hessian_weights.expand_as(squared_error) * squared_error

    return weighted_error.mean()


class RoundSTE(torch.autograd.Function):
    """
    Straight-Through Estimator: arrotondamento in forward,
    gradiente passato intatto in backward (necessario per ottimizzare s1, s2).
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class MRQQuantizer(nn.Module):
    """
    Multi-Region Quantization (Sezione III-C del paper).

    Softmax — due regioni:
        R1 = [0,  qmax·s1)   → step size s1        (valori piccoli, alta precisione)
        R2 = [qmax·s1, 1]    → step size 1/qmax     (fisso, come da paper)

    GELU — due regioni:
        R1 = [-qmax·s1g, 0)  → step size s1 (negativi)
        R2 = [0, qmax·s2g)   → step size s2 (positivi)
    """
    def __init__(self, bit_width=8, is_softmax=True):
        super().__init__()
        self.bit_width = bit_width
        self.is_softmax = is_softmax
        self.qmax = 2 ** (bit_width - 1) - 1  # 127 per 8-bit

        if is_softmax:
            self.s1 = nn.Parameter(torch.tensor(0.001))
            self.s2 = None
        else:
            self.s1 = nn.Parameter(torch.tensor(0.01))   # step per negativi
            self.s2 = nn.Parameter(torch.tensor(0.1))    # step per positivi

    def forward(self, x):
        round_ste = RoundSTE.apply

        if self.is_softmax:
            threshold = self.qmax * self.s1
            mask_small = (x < threshold).float()
            mask_large = (x >= threshold).float()

            x_small_int = torch.clamp(round_ste(x / self.s1), -self.qmax, self.qmax)
            x_small = x_small_int * self.s1

            fixed_step = 1.0 / self.qmax
            x_large_int = torch.clamp(torch.round(x / fixed_step), -self.qmax, self.qmax)
            x_large = x_large_int * fixed_step

            x_q = x_small * mask_small + x_large * mask_large
        else:
            mask_neg = (x < 0).float()
            mask_pos = (x >= 0).float()

            x_neg_int = torch.clamp(round_ste(x / self.s1), -self.qmax, self.qmax)
            x_neg = x_neg_int * self.s1

            x_pos_int = torch.clamp(round_ste(x / self.s2), -self.qmax, self.qmax)
            x_pos = x_pos_int * self.s2

            x_q = x_neg * mask_neg + x_pos * mask_pos

        return x_q


def optimize_mrq_parameters_ho(fp_activations, hessian_weights, quantizer_module,
                                iterations=750, lr=1e-3):
    optimizer = torch.optim.Adam(quantizer_module.parameters(), lr=lr)

    for _ in range(iterations):
        optimizer.zero_grad()
        quantized_activations = quantizer_module(fp_activations)
        loss = hessian_guided_loss(quantized_activations, fp_activations, hessian_weights)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            quantizer_module.s1.clamp_(min=1e-5)
            if not quantizer_module.is_softmax:
                quantizer_module.s2.clamp_(min=1e-5)

    return loss.item()


# =============================================================================
# 2. VARIABILI GLOBALI E SETUP TGQ
# =============================================================================

NUM_STEPS  = 50
NUM_GROUPS = 10
NUM_IMAGES = 32

current_group = "Group_0"
ts_to_group = {}
activation_tensor_cache = {}
hessian_tensor_cache = {}


def setup_time_buckets(scheduler, num_steps, num_groups):
    """Mappa ogni timestep al gruppo TGQ corrispondente (Eq. 9 del paper)."""
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.tolist()
    steps_per_group = num_steps // num_groups
    mapping = {}
    for i, ts in enumerate(timesteps):
        group_idx = min(i // steps_per_group, num_groups - 1)
        mapping[float(ts)] = f"Group_{group_idx}"
    return mapping


def update_current_group(t_tensor):
    """
    Aggiorna current_group in base al timestep corrente.

    FIX (doppia chiamata a trace_timestep_hook):
        Nel codice originale l'hook veniva chiamato esplicitamente 2× per ogni
        step del loop: una volta con t grezzo e una volta con t_tensor.
        Ora la logica è in questa funzione autonoma, chiamata una sola volta,
        e l'hook registrato sul modello serve solo come safety net automatico.
    """
    global current_group
    ts_val = float(t_tensor[0].item())
    if ts_val in ts_to_group:
        current_group = ts_to_group[ts_val]
    else:
        closest = min(ts_to_group.keys(), key=lambda k: abs(k - ts_val))
        current_group = ts_to_group[closest]


def get_caching_hook(layer_name):
    """
    Forward hook: salva le attivazioni in una lista per gruppo temporale.
    Accumula come lista di tensori → concatenati in Fase 2 con torch.cat().
    """
    def hook(module, input_tensor, output_tensor):
        global current_group

        if isinstance(output_tensor, tuple):
            act = output_tensor[0].detach().clone().float().cpu()
        else:
            act = output_tensor.detach().clone().float().cpu()

        activation_tensor_cache.setdefault(current_group, {})
        activation_tensor_cache[current_group].setdefault(layer_name, [])
        activation_tensor_cache[current_group][layer_name].append(act)

    return hook


def get_backward_hook(layer_name):
    """
    Backward hook: salva i gradienti al quadrato (approssimazione FIM, Eq. 15).

    FIX (shape disallineata — bug critico):
        Il codice originale accumulava i gradienti con +=, tenendo un singolo
        tensore con la shape di 1 sample. Nella Fase 2 fp_tensor aveva invece
        shape [N_samples, ...] perché costruito con torch.cat().
        Il broadcasting implicito di PyTorch non crashava ma produceva pesi
        Hessiani completamente sbagliati (replicazione su batch invece di media).

        Fix: accumuliamo anche qui una lista di tensori, esattamente come per
        le attivazioni. In Fase 2 faremo torch.stack(...).mean(dim=0) per
        ottenere un tensore [1, C, H, W] che rappresenta correttamente
        E[(∂L/∂z)²] mediato su tutti i campioni di calibrazione.
    """
    def hook(module, grad_input, grad_output):
        global current_group

        grad_sq = (grad_output[0].detach().clone().float()) ** 2 + 1e-6

        hessian_tensor_cache.setdefault(current_group, {})
        hessian_tensor_cache[current_group].setdefault(layer_name, [])
        hessian_tensor_cache[current_group][layer_name].append(grad_sq.cpu())

    return hook


# =============================================================================
# 3. ORCHESTRAZIONE PRINCIPALE
# =============================================================================

def register_hooks(pipe):
    """
    Registra tutti gli hook forward e backward sui layer corretti.

    FIX (hook sul layer sbagliato):
        Il codice originale usava block.attn1.to_q (proiezione lineare delle
        Query) per il layer post-softmax. Ma secondo la Fig. 4 e l'Algorithm 1
        del paper, MRQ+TGQ va applicato all'output della Softmax nel MatMul
        dell'MHSA, non all'output di to_q.

        In Hugging Face Diffusers il modulo Attention gestisce internamente
        softmax e matmul. Usiamo un hook sul modulo Attention completo
        (block.attn1) e intercettiamo l'output (che è già post-softmax+matmul)
        come approssimazione pratica dell'attivazione post-softmax.

        Per il layer post-GELU: block.ff.net[1] è il modulo GELU vero e
        proprio (net[0] è il Linear+GELU accoppiato). Hookiamo net[0] in
        forward (il cui output è già post-GELU) come nel codice originale,
        ma registriamo il backward hook sullo stesso modulo per coerenza.

    Nota: il paper quantizza tutti i blocchi N. Qui lo facciamo su tutti i
    transformer_blocks per fedeltà al paper (il codice originale usava solo
    il blocco 0). Riduci num_blocks se la VRAM è limitata.
    """
    hooks = []
    transformer = pipe.transformer

    for block_idx, block in enumerate(transformer.transformer_blocks):

        # --- Tracciamento timestep (solo dal blocco 0, basta uno) ---
        if block_idx == 0:
            h = block.norm1.emb.time_proj.register_forward_hook(
                _make_timestep_hook()
            )
            hooks.append(h)

        # --- Post-Softmax: output del modulo Attention completo ---
        # FIX: era block.attn1.to_q (pre-softmax); ora è block.attn1 (post-softmax+matmul)
        attn_name = f"Block{block_idx}_Attention_PostSoftmax"
        h = block.attn1.register_forward_hook(get_caching_hook(attn_name))
        hooks.append(h)
        h = block.attn1.register_full_backward_hook(get_backward_hook(attn_name))
        hooks.append(h)

        # --- Post-GELU: output del primo sotto-layer del feedforward ---
        # net[0] è un Linear che nel DiT di HF è accoppiato con GELU inline;
        # l'output è già post-GELU (verificato con pipe.transformer.transformer_blocks[0].ff.net)
        gelu_name = f"Block{block_idx}_FeedForward_PostGELU"
        h = block.ff.net[0].register_forward_hook(get_caching_hook(gelu_name))
        hooks.append(h)
        h = block.ff.net[0].register_full_backward_hook(get_backward_hook(gelu_name))
        hooks.append(h)

    return hooks


def _make_timestep_hook():
    """Hook forward per aggiornare current_group dal time_proj del blocco 0."""
    def hook(module, input, output):
        update_current_group(input[0])
    return hook


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Esecuzione su dispositivo: {device}")

    model_id = "facebook/DiT-XL-2-256"
    print(f"Caricamento del modello {model_id} in FP16...")
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    global ts_to_group
    ts_to_group = setup_time_buckets(pipe.scheduler, NUM_STEPS, NUM_GROUPS)

    # -------------------------------------------------------------------------
    # FASE 1: Raccolta attivazioni e calcolo Hessiano (calibrazione)
    # -------------------------------------------------------------------------
    print("\n--- FASE 1: Calibrazione con calcolo gradienti (HO) ---")

    pipe.transformer.train()
    hooks = register_hooks(pipe)

    for seed in range(NUM_IMAGES):
        print(f"  Immagine {seed + 1}/{NUM_IMAGES} (classe ImageNet {seed % 1000})...")
        generator = torch.Generator(device=device).manual_seed(seed)
        current_class = torch.tensor([seed % 1000], device=device)

        latents = torch.randn(
            (1, pipe.transformer.config.in_channels, 32, 32),
            generator=generator, device=device, dtype=torch.float16
        )
        pipe.scheduler.set_timesteps(NUM_STEPS)

        for t in pipe.scheduler.timesteps:
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            t_tensor = torch.tensor([t_val], dtype=torch.long, device=device)

            # FIX (doppia chiamata): aggiorniamo current_group una sola volta
            update_current_group(t_tensor)

            with torch.enable_grad():
                latent_input = latents.detach().clone()
                latent_input.requires_grad_(True)

                model_output = pipe.transformer(
                    latent_input,
                    timestep=t_tensor,
                    class_labels=current_class
                ).sample

                noise_pred, _ = model_output.chunk(2, dim=1)

                # Il paper usa la loss di denoising (Eq. 11): E[||ε - ε_θ(x_t,t)||²]
                # Usiamo rumore gaussiano come target — approssimazione pratica
                # (il target ideale sarebbe il rumore vero aggiunto a x_0, non disponibile
                # in PTQ senza il dataset originale).
                target_noise = torch.randn_like(noise_pred)
                loss = F.mse_loss(noise_pred, target_noise)
                loss.backward()

            with torch.no_grad():
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            torch.cuda.empty_cache()

    remove_hooks(hooks)
    pipe.transformer.eval()

    # -------------------------------------------------------------------------
    # FASE 2: Ottimizzazione MRQ + HO per ogni gruppo e layer
    # -------------------------------------------------------------------------
    print("\n--- FASE 2: Ottimizzazione parametri MRQ ---")
    export_data = {}

    for group_name, layers in activation_tensor_cache.items():
        print(f"\n  Gruppo: {group_name}")
        export_data[group_name] = {}

        for layer_name, tensor_list in layers.items():
            is_softmax_layer = "PostSoftmax" in layer_name

            quantizer = MRQQuantizer(bit_width=8, is_softmax=is_softmax_layer).to(device)

            # Concateniamo tutte le attivazioni raccolte → shape [N_total, ...]
            fp_tensor = torch.cat(tensor_list, dim=0).to(device)

            # FIX (shape disallineata — bug critico):
            #   Prima: hessian_tensor_cache conteneva un singolo tensore sommato
            #          con shape del singolo sample → broadcasting silenziosamente errato.
            #   Ora:   contiene una lista di tensori (uno per step/immagine).
            #          torch.stack crea [N, ...], .mean(dim=0) dà E[(∂L/∂z)²]
            #          con shape [1, ...] pronta per expand_as() in hessian_guided_loss.
            hessian_list = hessian_tensor_cache.get(group_name, {}).get(layer_name, [])
            if not hessian_list:
                print(f"    [WARN] Nessun gradiente per {layer_name}, skip.")
                continue

            hessian_weights = torch.stack(hessian_list, dim=0).mean(dim=0, keepdim=True).to(device)
            hessian_weights = hessian_weights / (hessian_weights.max() + 1e-8)

            final_loss = optimize_mrq_parameters_ho(
                fp_tensor,
                hessian_weights,
                quantizer,
                iterations=750,
                lr=1e-3
            )

            s1_val = quantizer.s1.item()
            s2_val = quantizer.s2.item() if not is_softmax_layer else None

            export_data[group_name][layer_name] = {
                "layer_type": "PostSoftmax" if is_softmax_layer else "PostGELU",
                "bit_width": quantizer.bit_width,
                "final_optimization_loss": round(final_loss, 6),
                "scale_factor_s1": round(s1_val, 6),
                "scale_factor_s2": round(s2_val, 6) if s2_val is not None else "N/A (Fixed Step)"
            }

            if is_softmax_layer:
                print(f"    [{layer_name}] s1={s1_val:.5f} | loss={final_loss:.6f}")
            else:
                print(f"    [{layer_name}] s1={s1_val:.5f}, s2={s2_val:.5f} | loss={final_loss:.6f}")

    # -------------------------------------------------------------------------
    # FASE 3: Salvataggio
    # -------------------------------------------------------------------------
    output_filename = "tq_dit_optimized_parameters.json"
    with open(output_filename, "w") as f:
        json.dump(export_data, f, indent=4)

    print(f"\n[SUCCESSO] Parametri salvati in '{output_filename}'.")


if __name__ == "__main__":
    main()