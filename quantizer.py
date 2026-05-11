import argparse
import torch
from diffusers import DiTPipeline

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="facebook/DiT-XL-2-256", help="Model name (default: facebook/DiT-XL-2-256)")
    
    parser.add_argument("-e", "--epoch", type=int, default=1, help="Epoch number (default: 1)")
    parser.add_argument("-nc", "--class_n", type=int, default=1, help="Class number (default: 1)")
    parser.add_argument("-b", "--bits", type=int, default=4, help="Bits of quantization (default: 4)")

    args = parser.parse_args()
    model_id = args.model
    epoch_num = args.epoch
    class_num = args.class_n

    print(f"Model: {args.model}")
    print(f"Epochs: {args.epoch}")
    print(f"Class number: {args.class_n}")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=False).to(device)

    '''
    Calculate the bucket:
    timesteps = calculate_buckets(pipe)
    '''
    # Temporaneo
    timesteps = torch.arange(0, 1001, 40, device=device)
    timesteps = [t.unsqueeze(0) for t in timesteps]

    '''
    Calculate layer shallow and deep:
    layer_shallow, layer_int, layer_deep = divide_layer(pipe,timesteps)
    '''
    # Temporaneo
    LAYER_SHALLOW, LAYER_INT, LAYER_DEEP = 4, 20, 4
    
    # Possibili input in cmd
    WINDOW_SIZE = 2
    WINDOW_STEP = 1
    GAMMA = 0.5

    '''
    Apply SliderQuant (modifies the pipe):
    pipe = apply_sliderquant(pipe,device,timesteps,layer_shallow,layer_int,layer_deep,window_size,window_step,gamma,epoch_num,class_num,bits)
    '''




    '''
    Apply TQ-DiT (modifies the pipe):
    pipe = apply_tq(pipe,timesteps)
    '''


    # Save the quantized model
    pipe.save_pretrained(f"output/{model_id}", safe_serialization=True)