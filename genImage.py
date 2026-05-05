import torch
from diffusers import DiTPipeline

# Specifica l'ID del modello
model_id = "facebook/DiT-XL-2-256"

# Carica la pipeline pre-addestrata
# Utilizziamo torch.float16 per ottimizzare l'uso della VRAM se hai una GPU
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Sposta il modello sulla GPU per velocizzare l'inferenza (se disponibile)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

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