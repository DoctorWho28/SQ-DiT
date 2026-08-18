import os
import argparse
import json
import torch
from tqdm import tqdm
from inference import load_quantized_pipeline
from diffusers import DiTPipeline




def save_json_info(model_id: str, batch_size: int, images_per_class: int, calc_final: bool = False):
    json_path = f"json/{model_id}.json"

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if not os.path.exists(json_path):
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
            "vram_max_total": torch.cuda.max_memory_allocated() / (1024**3),
            "batch_size": batch_size,
            "image_num_per_class": images_per_class}


    with open(json_path, "w") as J:
        json.dump(json_file,J,indent=4)



def generate_fid_images(pipe: DiTPipeline, model_id: str, inference_step: int, batch_size: int, seed: int, device: str, images_per_class: int):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    output_dir = f"FID_images/{model_id}"
    total_classes = 1000
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting generation of {total_classes * images_per_class} images in '{output_dir}'")
    print(f"Batch size: {batch_size}")

    pending_tasks = []
    for class_id in range(total_classes):
        for i in range(images_per_class):
            if not os.path.exists(os.path.join(output_dir, f"class_{class_id:03d}_img_{i:02d}.png")):
                pending_tasks.append((class_id, i))

    if not pending_tasks:
        print("\nAll images already generated!")
        return []

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    
    with tqdm(total=total_classes * images_per_class, desc="FID Generation") as pbar:
        pbar.update((total_classes * images_per_class) - len(pending_tasks))
        
        for i in range(0, len(pending_tasks), batch_size):
            chunk = pending_tasks[i : i + batch_size]
            class_labels = [task[0] for task in chunk]
            
            output = pipe(
                class_labels=class_labels,
                generator=generator,
                num_inference_steps=inference_step
            )

            # Save images
            for idx, img in enumerate(output.images):
                class_id, img_idx = chunk[idx]
                img_name = f"class_{class_id:03d}_img_{img_idx:02d}.png"
                img.save(os.path.join(output_dir, img_name))
                
            pbar.update(len(chunk))

    print(f"\nGeneration of {total_classes * images_per_class} images completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_id", type=str ,required=True, help="Model id (required)")
    parser.add_argument("-i", "--inference_step", type=int ,required=False, default=20, help="Inference step for non quantized models")
    parser.add_argument("-n", "--images_per_class", type=int ,required=False, default=50, help="Number of images per class")
    parser.add_argument("-bs", "--batch_size", type=int ,required=False, default=1, help="Batch size")
    parser.add_argument("-s", "--seed", type=int ,required=False, default=42, help="Seed for image generation")


    args = parser.parse_args()
    model_id = args.model_id
    images_per_class = args.images_per_class
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
    
    generate_fid_images(
        pipe,
        model_id,
        inference_step,
        batch_size,     
        seed, 
        device,
        images_per_class
    )


    save_json_info(model_id, batch_size, images_per_class)