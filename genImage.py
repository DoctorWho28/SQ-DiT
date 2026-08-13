import torch
from diffusers import DiTPipeline
import os
import contextlib
import sys
import time

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Reindirizza tutti i print sia sulla console che sul file di testo
sys.stdout = DualLogger("log_comparativo.txt")

@contextlib.contextmanager
def track_vram(operation_name):
    """
    Context manager per misurare precisamente quanta VRAM
    e quanto tempo viene utilizzato.
    """
    start_time = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    
    yield
    
    elapsed_time = time.time() - start_time
    
    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"\n📊 [Track] {operation_name}")
        print(f"  Tempo Trascorso:  {elapsed_time:.2f} secondi")
        print(f"  Memoria Iniziale: {start_mem / 1024**2:.2f} MB")
        print(f"  Memoria Finale:   {end_mem / 1024**2:.2f} MB")
        print(f"  Picco Massimo:    {peak_mem / 1024**2:.2f} MB (Quella effettivamente necessaria!)")
        print("-" * 50)
    else:
        print(f"\n📊 [Track] {operation_name}")
        print(f"  Tempo Trascorso:  {elapsed_time:.2f} secondi")
        print("-" * 50)
# Impostazioni generali
device = "cuda" if torch.cuda.is_available() else "cpu"
class_labels = [399] # 207 = Golden Retriever
inference_steps = 20 # FONDAMENTALE: usiamo gli stessi step per entrambi!
seed = 42
print("Inizio test comparativo...")

# ==========================================
# 1. GENERAZIONE CON MODELLO ORIGINALE
# ==========================================
print("\n--- TEST MODELLO ORIGINALE ---")
base_model_id = "facebook/DiT-XL-2-256"

# ESEMPIO: Misuriamo quanta RAM serve solo per caricare il modello base
with track_vram("Caricamento Modello Base (FP16)"):
    print(f"Caricamento del modello base {base_model_id} in corso...")
    pipe_base = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    pipe_base = pipe_base.to(device)

# IMPORTANTE: Resettiamo il seed esattamente prima della generazione!
generator = torch.Generator(device=device).manual_seed(seed)

# ESEMPIO: Misuriamo quanta RAM serve per eseguire la generazione vera e propria
with track_vram("Generazione Immagine (Modello Base)"):
    print(f"Generazione in corso (Originale)...")
    output_base = pipe_base(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)

image_base = output_base.images[0]
image_base.save("immagine_dit_original.png")
print("Immagine originale salvata come 'immagine_dit_original.png'")

# Liberiamo la memoria GPU per non far esplodere la scheda video!
del pipe_base
torch.cuda.empty_cache()

# ==========================================
# 2. GENERAZIONE CON MODELLO QUANTIZZATO
# ==========================================
print("\n--- TEST MODELLO QUANTIZZATO ---")
quant_model_id = "output/facebook/DiT-XL-2-256_v2"

if not os.path.exists(quant_model_id):
    print(f"ATTENZIONE: Cartella {quant_model_id} non trovata. Hai eseguito l'ottimizzazione prima?")
else:
    print(f"Caricamento del modello quantizzato {quant_model_id} in corso...")
    pipe_quant = DiTPipeline.from_pretrained(quant_model_id, torch_dtype=torch.float16)
    pipe_quant = pipe_quant.to(device)

    # IMPORTANTE: RESETTIAMO IL SEED DI NUOVO A 42!
    # Altrimenti la seconda generazione usa rumore diverso e non possiamo fare un confronto alla pari!
    generator = torch.Generator(device=device).manual_seed(seed)

    with track_vram("Generazione Immagine (Quantizzato V2)"):
        print(f"Generazione in corso (Quantizzato)...")
        output_quant = pipe_quant(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_quant = output_quant.images[0]
    image_quant.save("immagine_dit_quantized_v2.png")
    print("Immagine quantizzata salvata come 'immagine_dit_quantized_v2.png'")

    del pipe_quant
    torch.cuda.empty_cache()

# ==========================================
# 3. GENERAZIONE CON MODELLO V3 (W4A16 + Group Quant)
# ==========================================
print("\n--- TEST MODELLO V3 (W4A16 INT4) ---")
quant_v3_dir = "output/facebook/DiT-XL-2-256_v3"

if not os.path.exists(quant_v3_dir):
    print(f"ATTENZIONE: Cartella {quant_v3_dir} non trovata. Hai eseguito l'ottimizzazione V3?")
else:
    print(f"Caricamento del modello base per V3 in corso...")
    pipe_v3 = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    # Per caricare i pesi V3, dobbiamo prima ricreare la struttura W4A16 nei layer intermedi!
    from other_implementations.slider_quant_v3 import W4A16Linear
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    
    print("Iniezione dei layer W4A16 nell'architettura...")
    layer_shallow, layer_int = 4, 20
    group_size = 128
    skip_names = ["norm1", "emb"]
    
    def inject_w4a16(module):
        for name, child in module.named_children():
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear):
                w4 = W4A16Linear(child.in_features, child.out_features, group_size, bias=(child.bias is not None))
                setattr(module, name, w4)
            else:
                inject_w4a16(child)
                
    for layer_id in range(layer_shallow, layer_shallow + layer_int):
        inject_w4a16(pipe_v3.transformer.transformer_blocks[layer_id])
        
    print("Caricamento dei pesi impacchettati in uint8 da safetensors...")
    from safetensors.torch import load_file
    transformer_state_dict = load_file(os.path.join(quant_v3_dir, "transformer", "diffusion_pytorch_model.safetensors"))
    pipe_v3.transformer.load_state_dict(transformer_state_dict, strict=True)
    pipe_v3 = pipe_v3.to(device)

    # IMPORTANTE: RESETTIAMO IL SEED A 42!
    generator = torch.Generator(device=device).manual_seed(seed)

    print("\n--- VERIFICA DIAGNOSTICA V3 ---")
    test_layer = pipe_v3.transformer.transformer_blocks[4].attn1.to_q
    print(f"Layer 4 attn1.to_q è di tipo: {type(test_layer).__name__}")
    if hasattr(test_layer, 'weight_packed'):
        print(f"I pesi sono salvati fisicamente come: {test_layer.weight_packed.dtype} (8-bit intero!)")
    print(f"Numero di step di inferenza impostati per la pipeline: {inference_steps}")
    print("-------------------------------\n")

    with track_vram("Generazione Immagine (V3 INT4)"):
        print(f"Generazione in corso (V3 INT4)...")
        output_v3 = pipe_v3(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_v3 = output_v3.images[0]
    image_v3.save("immagine_dit_quantized_v3.png")
    print("Immagine V3 salvata come 'immagine_dit_quantized_v3.png'")

    del pipe_v3
    torch.cuda.empty_cache()

# ==========================================
# 4. GENERAZIONE CON MODELLO V1 (SliderQuant Attuale)
# ==========================================
print("\n--- TEST MODELLO V1 (SliderQuant Attuale) ---")
quant_v1_dir = "output/facebook/DiT-XL-2-256"

if not os.path.exists(quant_v1_dir):
    print(f"ATTENZIONE: Cartella {quant_v1_dir} non trovata.")
else:
    print(f"Caricamento del modello base per V1 in corso...")
    pipe_v1 = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    from slider_quant import WXA16Linear
    import torch.nn as nn
    import json
    
    config_path = os.path.join(quant_v1_dir, "quantization_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            q_config = json.load(f)
        layer_shallow = q_config.get("layer_shallow", 4)
        layer_int = q_config.get("layer_int", 20)
        group_size = q_config.get("group_size", 128)
        bits_int = q_config.get("bits", 4)
    else:
        print("Nessun quantization_config.json trovato, uso parametri di default.")
        layer_shallow, layer_int = 4, 20
        group_size = 128
        bits_int = 4
        
    print(f"Iniezione dei layer WXA16 misti (Shallow/Deep 8-bit, Intermediate {bits_int}-bit)...")
    skip_names = ["norm1", "emb"]
    
    def inject_wXa16_v1(module, bits):
        for name, child in module.named_children():
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear):
                wX = WXA16Linear(child.in_features, child.out_features, group_size, bits=bits, bias=(child.bias is not None))
                setattr(module, name, wX)
            else:
                inject_wXa16_v1(child, bits)
                
    # Shallow (8-bit)
    for layer_id in range(layer_shallow):
        inject_wXa16_v1(pipe_v1.transformer.transformer_blocks[layer_id], bits=8)
    # Intermediate (dinamico)
    for layer_id in range(layer_shallow, layer_shallow + layer_int):
        inject_wXa16_v1(pipe_v1.transformer.transformer_blocks[layer_id], bits=bits_int)
    # Deep (8-bit)
    for layer_id in range(layer_shallow + layer_int, len(pipe_v1.transformer.transformer_blocks)):
        inject_wXa16_v1(pipe_v1.transformer.transformer_blocks[layer_id], bits=8)
        
    print("Caricamento dei pesi impacchettati in uint8 da safetensors...")
    from safetensors.torch import load_file
    transformer_state_dict = load_file(os.path.join(quant_v1_dir, "transformer", "diffusion_pytorch_model.safetensors"))
    pipe_v1.transformer.load_state_dict(transformer_state_dict, strict=True)
    pipe_v1 = pipe_v1.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    with track_vram("Generazione Immagine (V1 WXA16)"):
        print(f"Generazione in corso (V1 WXA16)...")
        output_v1 = pipe_v1(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_v1 = output_v1.images[0]
    image_v1.save("immagine_dit_quantized_v1.png")
    print("Immagine V1 salvata come 'immagine_dit_quantized_v1.png'")

    del pipe_v1
    torch.cuda.empty_cache()

# ==========================================
# 5. GENERAZIONE CON MODELLO V4 (WXA16 Mixed Precision)
# ==========================================
print("\n--- TEST MODELLO V4 (WXA16 Dynamic Mixed Precision) ---")
quant_v4_dir = "output/facebook/DiT-XL-2-256_v4"

if not os.path.exists(quant_v4_dir):
    print(f"ATTENZIONE: Cartella {quant_v4_dir} non trovata. Hai eseguito l'ottimizzazione V4?")
else:
    print(f"Caricamento del modello base per V4 in corso...")
    pipe_v4 = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    from other_implementations.slider_quant_v4 import WXA16Linear
    import torch.nn as nn
    
    print("Iniezione dei layer WXA16 misti nell'architettura (8-bit Shallow/Deep, 4-bit Intermediate)...")
    layer_shallow, layer_int = 4, 20
    group_size = 128
    skip_names = ["norm1", "emb"]
    
    def inject_wXa16(module, bits):
        for name, child in module.named_children():
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear):
                wX = WXA16Linear(child.in_features, child.out_features, group_size, bits=bits, bias=(child.bias is not None))
                setattr(module, name, wX)
            else:
                inject_wXa16(child, bits)
                
    # Shallow (8-bit)
    for layer_id in range(layer_shallow):
        inject_wXa16(pipe_v4.transformer.transformer_blocks[layer_id], bits=8)
    # Intermediate (4-bit)
    for layer_id in range(layer_shallow, layer_shallow + layer_int):
        inject_wXa16(pipe_v4.transformer.transformer_blocks[layer_id], bits=4)
    # Deep (8-bit)
    for layer_id in range(layer_shallow + layer_int, len(pipe_v4.transformer.transformer_blocks)):
        inject_wXa16(pipe_v4.transformer.transformer_blocks[layer_id], bits=8)
        
    print("Caricamento dei pesi impacchettati in uint8 da safetensors...")
    from safetensors.torch import load_file
    transformer_state_dict = load_file(os.path.join(quant_v4_dir, "transformer", "diffusion_pytorch_model.safetensors"))
    pipe_v4.transformer.load_state_dict(transformer_state_dict, strict=True)
    pipe_v4 = pipe_v4.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    print("\n--- VERIFICA DIAGNOSTICA V4 ---")
    test_layer_shallow = pipe_v4.transformer.transformer_blocks[0].attn1.to_q
    test_layer_int = pipe_v4.transformer.transformer_blocks[4].attn1.to_q
    print(f"Layer 0 (Shallow) è: {type(test_layer_shallow).__name__} a {test_layer_shallow.bits}-bit")
    print(f"Layer 4 (Intermediate) è: {type(test_layer_int).__name__} a {test_layer_int.bits}-bit")
    print("-------------------------------\n")

    with track_vram("Generazione Immagine (V4 WXA16)"):
        print(f"Generazione in corso (V4 WXA16)...")
        output_v4 = pipe_v4(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_v4 = output_v4.images[0]
    image_v4.save("immagine_dit_quantized_v4.png")
    print("Immagine V4 salvata come 'immagine_dit_quantized_v4.png'")

    del pipe_v4
    torch.cuda.empty_cache()

# ==========================================
# 6. GENERAZIONE CON MODELLO V6 (SliderQuant WXAX)
# ==========================================
print("\n--- TEST MODELLO V6 (SliderQuant WXAX) ---")
quant_v6_dir = "output/facebook/DiT-XL-2-256_v6"

if not os.path.exists(quant_v6_dir):
    print(f"ATTENZIONE: Cartella {quant_v6_dir} non trovata. Hai eseguito l'ottimizzazione V6?")
else:
    with track_vram("Caricamento Modello Base per V6"):
        print(f"Caricamento del modello base per V6 in corso...")
        pipe_v6 = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    
    from other_implementations.slider_quant_v6 import WXAXLinear
    import torch.nn as nn
    import json
    
    config_path = os.path.join(quant_v6_dir, "quantization_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            q_config = json.load(f)
        layer_shallow = q_config.get("layer_shallow", 4)
        layer_int = q_config.get("layer_int", 20)
        group_size = q_config.get("group_size", 128)
        bits_int = q_config.get("bits", 4)
        bits_ext = q_config.get("bits_ext", 8)
        act_bits_int = q_config.get("act_bits_int", 4)
        act_bits_ext = q_config.get("act_bits_ext", 8)
    else:
        print("Nessun quantization_config.json trovato, uso parametri di default V6.")
        layer_shallow, layer_int = 4, 20
        group_size = 128
        bits_int, bits_ext = 4, 8
        act_bits_int, act_bits_ext = 4, 8
        
    print(f"Iniezione dei layer WXAX misti...")
    skip_names = ["norm1", "emb"]
    
    def inject_wXax_v6(module, weight_bits, act_bits):
        for name, child in module.named_children():
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear):
                wX = WXAXLinear(child.in_features, child.out_features, group_size, weight_bits=weight_bits, act_bits=act_bits, bias=(child.bias is not None))
                setattr(module, name, wX)
            else:
                inject_wXax_v6(child, weight_bits, act_bits)
                
    # Shallow
    for layer_id in range(layer_shallow):
        inject_wXax_v6(pipe_v6.transformer.transformer_blocks[layer_id], weight_bits=bits_ext, act_bits=act_bits_ext)
    # Intermediate
    for layer_id in range(layer_shallow, layer_shallow + layer_int):
        inject_wXax_v6(pipe_v6.transformer.transformer_blocks[layer_id], weight_bits=bits_int, act_bits=act_bits_int)
    # Deep
    for layer_id in range(layer_shallow + layer_int, len(pipe_v6.transformer.transformer_blocks)):
        inject_wXax_v6(pipe_v6.transformer.transformer_blocks[layer_id], weight_bits=bits_ext, act_bits=act_bits_ext)
        
    with track_vram("Caricamento Pesi Quantizzati e Spostamento su GPU (V6)"):
        print("Caricamento dei pesi impacchettati in uint8 da safetensors per V6...")
        from safetensors.torch import load_file
        transformer_state_dict = load_file(os.path.join(quant_v6_dir, "transformer", "diffusion_pytorch_model.safetensors"))
        pipe_v6.transformer.load_state_dict(transformer_state_dict, strict=True)
        pipe_v6 = pipe_v6.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("\n--- VERIFICA DIAGNOSTICA V6 ---")
    test_layer_shallow = pipe_v6.transformer.transformer_blocks[0].attn1.to_q
    test_layer_int = pipe_v6.transformer.transformer_blocks[4].attn1.to_q
    print(f"Layer 0 (Shallow) è: {type(test_layer_shallow).__name__} (W{test_layer_shallow.weight_bits} A{test_layer_shallow.act_bits})")
    print(f"Layer 4 (Intermediate) è: {type(test_layer_int).__name__} (W{test_layer_int.weight_bits} A{test_layer_int.act_bits})")
    print("-------------------------------\n")

    with track_vram("Generazione Immagine (V6 WXAX)"):
        print(f"Generazione in corso (V6 WXAX)...")
        output_v6 = pipe_v6(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_v6 = output_v6.images[0]
    image_v6.save("immagine_dit_quantized_v6.png")
    print("Immagine V6 salvata come 'immagine_dit_quantized_v6.png'")

    del pipe_v6
    torch.cuda.empty_cache()

print("\nConfronto completato! Apri le immagini per vedere le differenze (Originale, V1, V2, V3, V4, V6).")