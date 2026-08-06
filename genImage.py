import torch
from diffusers import DiTPipeline
import os

# Impostazioni generali
device = "cuda" if torch.cuda.is_available() else "cpu"
class_labels = [0] # 207 = Golden Retriever
inference_steps = 20 # FONDAMENTALE: usiamo gli stessi step per entrambi!
seed = 42
print("Inizio test comparativo...")

# ==========================================
# 1. GENERAZIONE CON MODELLO ORIGINALE
# ==========================================
print("\n--- TEST MODELLO ORIGINALE ---")
base_model_id = "facebook/DiT-XL-2-256"
print(f"Caricamento del modello base {base_model_id} in corso...")
pipe_base = DiTPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe_base = pipe_base.to(device)

# IMPORTANTE: Resettiamo il seed esattamente prima della generazione!
generator = torch.Generator(device=device).manual_seed(seed)

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

    print(f"Generazione in corso (V3 INT4)...")
    output_v3 = pipe_v3(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_v3 = output_v3.images[0]
    image_v3.save("immagine_dit_quantized_v3.png")
    print("Immagine V3 salvata come 'immagine_dit_quantized_v3.png'")

    del pipe_v3
    torch.cuda.empty_cache()

# ==========================================
# 4. GENERAZIONE CON MODELLO V1 (10 Step)
# ==========================================
print("\n--- TEST MODELLO V1 (Addestrato a 10 step) ---")
quant_v1_dir = "output/facebook/DiT-XL-2-256"

if not os.path.exists(quant_v1_dir):
    print(f"ATTENZIONE: Cartella {quant_v1_dir} non trovata.")
else:
    print(f"Caricamento del modello V1 {quant_v1_dir} in corso...")
    pipe_v1 = DiTPipeline.from_pretrained(quant_v1_dir, torch_dtype=torch.float16)
    pipe_v1 = pipe_v1.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generazione in corso (V1)...")
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

    print(f"Generazione in corso (V4 WXA16)...")
    output_v4 = pipe_v4(class_labels=class_labels, generator=generator, num_inference_steps=inference_steps)
    image_v4 = output_v4.images[0]
    image_v4.save("immagine_dit_quantized_v4.png")
    print("Immagine V4 salvata come 'immagine_dit_quantized_v4.png'")

    del pipe_v4
    torch.cuda.empty_cache()

print("\nConfronto completato! Apri le immagini per vedere le differenze (Originale, V1, V2, V3, V4).")