import json
import torch
from diffusers import DiTPipeline

BATCH_SIZE = 1
NUM_CLASSES = 1
NUM_STEPS = 25
seeds = range(50)

activation_max_values = {}
last_ts = 1000

def trace_timestep_hook(module, input, output):
    # 'input' è una tupla dei parametri passati al layer
    # In DiT, il primo elemento di input al timestep_embedder è proprio t
    timestep = input[0]
    global last_ts
    last_ts = str(timestep[0].item())
    print(f"--- [HOOK] Il layer sta processando t: {timestep} ---")



# 2. Definiamo la nostra "microspia" (La funzione Hook)
def get_activation_hook(layer_name):
    """
    Questo hook viene eseguito in automatico da PyTorch ogni volta che 
    i dati passano attraverso il layer a cui è attaccato.
    """
    def hook(module, input_tensor, output_tensor):
        # input_tensor è una tupla. Il primo elemento (input_tensor[0]) 
        # contiene esattamente le ATTIVAZIONI che stanno per entrare nel layer Linear.
        attivazioni = input_tensor[0]
        
        # Calcoliamo il valore massimo assoluto di questo blocco di dati
        current_max = attivazioni.abs().max().item()
        
        if str(last_ts) not in activation_max_values:
            activation_max_values[last_ts] = {}  
            activation_max_values[last_ts][layer_name] = current_max
        elif layer_name not in activation_max_values[last_ts] or current_max > activation_max_values[last_ts][layer_name]:
            activation_max_values[last_ts][layer_name] = current_max
        
        print(last_ts,layer_name,activation_max_values[last_ts][layer_name])
            
    return hook

# 1. Creazione della cartella per la Baseline
output_dir = "baseline_fp16"

# 2. Caricamento del modello DiT in FP16
model_id = "facebook/DiT-XL-2-256"
print(f"Caricamento del modello {model_id} in FP16...")


pipe = DiTPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
)

for i, block in enumerate(pipe.transformer.transformer_blocks):
    h_ts = block.norm1.emb.time_proj.register_forward_hook(trace_timestep_hook)
    block.attn1.to_q.register_forward_hook(get_activation_hook('Attention_Q'))
    block.attn1.to_k.register_forward_hook(get_activation_hook('Attention_K'))
    block.attn1.to_v.register_forward_hook(get_activation_hook('Attention_V'))
    block.ff.net[0].proj.register_forward_hook(get_activation_hook('FeedForward_Up'))


# Esegui la pipeline
image = pipe(class_labels=[0], num_inference_steps=NUM_STEPS).images[0]

with open("max_values.json","w") as F:
    json.dump(activation_max_values,F,indent=4)
