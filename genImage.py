import torch
from diffusers import DiTPipeline
from safetensors.torch import load_file
import os

# Importiamo la funzione che "inietta" la nostra architettura nel Transformer
from slider_quant import replace_linears_with_sliderquant

# Specifica l'ID del modello
model_id = "facebook/DiT-XL-2-256"

# Carica la pipeline pre-addestrata
# Utilizziamo torch.float16 per ottimizzare l'uso della VRAM se hai una GPU
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Sposta il modello sulla GPU per velocizzare l'inferenza (se disponibile)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print("1. Modifico l'architettura base per supportare SliderQuant...")
# Dobbiamo applicare replace_linears_with_sliderquant a tutti i 28 layer
for layer in pipe.transformer.transformer_blocks:
    # Usiamo gli stessi parametri (bits=4, rank=4) usati in quantizer.py
    replace_linears_with_sliderquant(layer, bits=4, rank=4, gamma=1.0)
    layer.to(device)

print("2. Carico i pesi ottimizzati (salvati da save_pretrained)...")
# save_pretrained salva i pesi del transformer all'interno della sua sottocartella specifica.
model_path = f"output/{model_id}/transformer/diffusion_pytorch_model.safetensors"

if os.path.exists(model_path):
    state_dict = load_file(model_path, device=device)
    pipe.transformer.load_state_dict(state_dict)
    print("Pesi caricati con successo!")
else:
    print(f"ATTENZIONE: File {model_path} non trovato. Verrà usato il modello base!")

# Imposta il seed a 42 tramite un generatore per garantire la riproducibilità
generator = torch.Generator(device=device).manual_seed(42)

# Scegli la classe ImageNet (da 0 a 999)
# Ad esempio, la classe 207 corrisponde a "Golden Retriever"
class_labels = [0] 

# Genera l'immagine
print(f"Generazione in corso sul dispositivo: {device}...")
output = pipe(class_labels=class_labels, generator=generator)

# Estrai l'immagine dalla tupla generata
image = output.images[0]

# Salva l'immagine
filename = "immagine_dit_seed42.png"
image.save(filename)
print(f"Immagine salvata con successo come '{filename}'!")