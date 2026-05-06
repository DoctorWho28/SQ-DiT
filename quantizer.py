import torch
from diffusers import DiTPipeline
import torch.nn as nn

LAYER_SHALLOW = 4
LAYER_INT = 20
LAYER_DEEP = 4
WINDOW_SIZE = 2
WINDOW_STEP = 1
GAMMA = 0.5


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "facebook/DiT-XL-2-256"
pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=False).to(device)

timesteps = torch.arange(0, 1001, 40, device=device)
timesteps = [t.unsqueeze(0) for t in timesteps]

def quantize_tensor(tensor, bits=4):
    zmin = tensor.min()
    zmax = tensor.max()
    qmax = 2**bits - 1
    alpha = (zmax-zmin)/qmax
    beta = torch.round(zmin/alpha)

    quantized = torch.round(tensor / alpha) - beta
    quantized = quantized.clamp(0, qmax)
    dequantized = (quantized + beta) * alpha
    return dequantized

def calculate_window_index():
    window_list = []
    curr_window = []

    for i in range(LAYER_SHALLOW):
        curr_window.append(i)
        window_list.append(curr_window.copy())

    i = LAYER_SHALLOW-1
    curr_window = [i + x for x in range(WINDOW_SIZE)]

    for i in range(LAYER_SHALLOW+1, LAYER_SHALLOW + LAYER_INT+2):
        window_list.append(curr_window.copy())
        for _ in range(WINDOW_STEP):
            curr_window.pop(0)
            curr_window.append(i)

    i = LAYER_SHALLOW + LAYER_INT
    curr_window = [i + x for x in range(LAYER_DEEP)]

    for i in range(i, LAYER_SHALLOW + LAYER_INT + LAYER_DEEP):
        window_list.append(curr_window.copy())
        curr_window.pop(0)

    return window_list

target_modules = []
for i, module in enumerate(pipe.transformer.transformer_blocks):
    target_modules.append((i,module))

block_names = []
for name, mod in pipe.transformer.transformer_blocks[0].named_modules():
    if isinstance(mod, nn.Linear):
        block_names.append(name)


window_list = calculate_window_index()