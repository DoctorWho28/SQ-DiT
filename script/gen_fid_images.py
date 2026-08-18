import os
import argparse
import json
import torch
import time
from tqdm import tqdm
from inference import load_quantized_pipeline
from diffusers import DiTPipeline
import contextlib

SAVE_STEP = 1000


@contextlib.contextmanager
def track_info():
    info = {}
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    yield info

    info["time"] = time.time() - start_time
    
    if use_cuda:
        info["vram"] = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        info["vram"] = 0.0


def save_json_info(total_info: list[dict], model_id: str, calc_final: bool = False, batch_size: int = None, image_num: int = None):
    json_path = f"json/{model_id}.json"

    # Creiamo la cartella se non esiste, in caso il modello contenga uno slash (es: facebook/DiT...)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if not os.path.exists(json_path):
        # Se anche cercando indietro non esiste, inizializziamo un dizionario vuoto
        if os.path.exists("../" + json_path):
            json_path = "../" + json_path
            with open(json_path, "r") as J:
                json_file = json.load(J)
        else:
            json_file = {}
    else:
        with open(json_path, "r") as J:
            json_file = json.load(J)

    if "generation" not in json_file:
        json_file["generation"] = {
            "mean_time": 0,
            "vram_max": 0,
            "batch_size": batch_size,
            "image_num_per_class": image_num,
            "single_values": []}

    if batch_size is not None:
        json_file["generation"]["batch_size"] = batch_size
    if image_num is not None:
        json_file["generation"]["image_num_per_class"] = image_num

    json_file["generation"]["single_values"] += total_info 

    if calc_final:
        json_file["generation"]["mean_time"] = sum([x["time"] for x in json_file["generation"]["single_values"]]) / len(json_file["generation"]["single_values"])
        json_file["generation"]["vram_max"] = max([x["vram"] for x in json_file["generation"]["single_values"]])


    with open(json_path, "w") as J:
        json.dump(json_file,J,indent=4)



def generate_fid_images(pipe: DiTPipeline,
    model_id: str,
    inference_step: int,
    batch_size: int,
    seed: int,
    device: str,
    images_per_class: int
):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    output_dir = f"FID_images/{model_id}"
    total_classes = 1000
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting generation of {total_classes * images_per_class} images in '{output_dir}'...")
    print(f"Batch size: {batch_size}. Sfruttamento totale della GPU attivato!")
    print("Note: The script supports RESUME. If interrupted, it will resume from where it stopped.")

    # 1. Raccogliamo tutte le immagini mancanti
    pending_tasks = []
    for class_id in range(total_classes):
        for i in range(images_per_class):
            if not os.path.exists(os.path.join(output_dir, f"class_{class_id:03d}_img_{i:02d}.png")):
                pending_tasks.append((class_id, i))

    if not pending_tasks:
        print("\nTutte le immagini sono già state generate!")
        return []

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    total_info = []
    image_before_save = SAVE_STEP
    
    with tqdm(total=total_classes * images_per_class, desc="FID Generation") as pbar:
        # Avanza la barra per le immagini che saltiamo (già esistenti)
        pbar.update((total_classes * images_per_class) - len(pending_tasks))
        
        # 2. Processiamo tutto a blocchi pieni da 'batch_size'
        for i in range(0, len(pending_tasks), batch_size):
            chunk = pending_tasks[i : i + batch_size]
            class_labels = [task[0] for task in chunk]
            
            with track_info() as info:
                output = pipe(
                    class_labels=class_labels,
                    generator=generator,
                    num_inference_steps=inference_step
                )

            total_info.append(info)

            # Salvataggio json periodico
            image_before_save -= len(chunk)
            if image_before_save <= 0:
                save_json_info(total_info, model_id, batch_size=batch_size, image_num=images_per_class)
                image_before_save = SAVE_STEP
                total_info = []

            # Salvataggio immagini su disco
            for idx, img in enumerate(output.images):
                class_id, img_idx = chunk[idx]
                img_name = f"class_{class_id:03d}_img_{img_idx:02d}.png"
                img.save(os.path.join(output_dir, img_name))
                
            pbar.update(len(chunk))

    print(f"\nGeneration of {total_classes * images_per_class} images completed successfully!")
    return total_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_id", type=str ,required=True, help="Model id (required)")
    parser.add_argument("-i", "--inference_step", type=int ,required=False, default=20, help="Inference step for non quantized models")
    parser.add_argument("-n", "--image_num", type=int ,required=False, default=50, help="Number of images per class")
    parser.add_argument("-bs", "--batch_size", type=int ,required=False, default=1, help="Batch size")
    parser.add_argument("-s", "--seed", type=int ,required=False, default=42, help="Seed for image generation")


    args = parser.parse_args()
    model_id = args.model_id
    image_num = args.image_num
    inference_step = args.inference_step
    batch_size = args.batch_size
    seed = args.seed


    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_dir = f"output/{model_id}"
    
    if not os.path.exists(model_dir):
        model_dir = "../" + model_dir


    print(f"Loading pipeline for {model_id}...")
    if os.path.exists(model_dir):
        assert(os.path.exists(f"{model_dir}/quantization_config.json")),"The model need to have a quantization_config.json file"
        pipe, inference_step = load_quantized_pipeline(model_dir, device)
    else:
        pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        
    pipe.set_progress_bar_config(disable=True)
    
    total_info = generate_fid_images(
        pipe,
        model_id,
        inference_step,
        batch_size,     
        seed, 
        device,
        image_num
    )


    save_json_info(total_info, model_id, calc_final=True, batch_size=batch_size, image_num=image_num)