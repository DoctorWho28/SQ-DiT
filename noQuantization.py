import os
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

# 1. Creazione della cartella per la Baseline
output_dir = "baseline_fp16"

# 2. Caricamento del modello DiT in FP16
model_id = "facebook/DiT-XL-2-256"
print(f"Caricamento del modello {model_id} in FP16...")

# Aggiungi il tuo token di Hugging Face qui
tuo_token = "TUO_TOKEN_HUGGING_FACE"

pipe = DiTPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,

)

BATCH_SIZE = 32      # Genera 10 immagini alla volta. Alzalo a 16 o 25 se hai molta VRAM!
NUM_STEPS = 25

seedFile = "seeds.txt"
seeds = []

try:
    with open(seedFile, "r") as file:
        seeds = [int(linea.strip()) for linea in file]
except FileNotFoundError:
    print(f"Errore: Il file '{seedFile}' non esiste.")
    exit()
except ValueError:
    print("Errore: Il file contiene dati che non possono essere convertiti in interi.")
    exit()
# DiT è condizionato sulle classi di ImageNet. 
# Usiamo la stessa classe per tutte le immagini per avere un Ground Truth coerente.
# Esempio: classe 207 (Golden Retriever) o 1 (Pesce rosso). Usiamo 207.

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# 3. CICLO DI GENERAZIONE OTTIMIZZATO
print("Inizio generazione delle immagini a blocchi (batch)...")

for j in range(5):
    print(f"\n--- Generazione immagini classe {j} ---")
    classe_dir = os.path.join(output_dir, f"classe_{j}")
    os.makedirs(classe_dir, exist_ok=True)
    
    # Invece di iterare un seed alla volta, iteriamo a "blocchi" (batch)
    for i in range(0, 2, BATCH_SIZE):
        # Prendiamo i seed per questo batch (es. i primi 10 seed)
        batch_seeds = seeds[i : i + BATCH_SIZE]
        actual_batch_size = len(batch_seeds)
        
        # Creiamo una lista di generatori (uno per ogni immagine del batch) per la riproducibilità
        generators = [torch.Generator(device="cuda").manual_seed(s) for s in batch_seeds]
        
        # Diciamo alla pipeline di generare N volte la stessa classe j
        class_labels = [j] * actual_batch_size
        
        # Generiamo il batch intero in un solo colpo!
        output = pipe(
            class_labels=class_labels, 
            num_inference_steps=NUM_STEPS, 
            generator=generators
        )
        
        # Salviamo le immagini appena generate
        for k, image in enumerate(output.images):
            seed_corrente = batch_seeds[k]
            filename = os.path.join(classe_dir, f"dit_baseline_seed_{seed_corrente}.png")
            image.save(filename)
            
        print(f"[{min(i + BATCH_SIZE, 50)}/50] Immagini salvate...")

print("\nFase 1 completata con successo! Ground Truth generato a velocità turbo.")