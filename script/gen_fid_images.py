import os
import argparse
import json
import torch
import time
from tqdm import tqdm
from inference import load_quantized_pipeline
from diffusers import DiTPipeline

def generate_fid_images(output_dir: str,
    pipe: DiTPipeline,
    inference_step: int,
    batch_size: int,
    seed: int,
    device: str,
    images_per_class: int
):
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        
    generator = torch.Generator(device=device)
    total_classes = 1000
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting generation of {total_classes * images_per_class} images in '{output_dir}'...")
    print(f"Batch size: {batch_size}. This operation will take several hours.")
    print("Note: The script supports RESUME. If interrupted, it will resume from where it stopped by skipping already generated images.")

    total_time = 0
    image_generated = 0

    
    with tqdm(total=total_classes * images_per_class, desc="FID Generation") as pbar:
        for class_id in range(total_classes):
            
            existing_images = 0
            for i in range(images_per_class):
                if os.path.exists(os.path.join(output_dir, f"class_{class_id:03d}_img_{i:02d}.png")):
                    existing_images += 1
            
            pbar.update(existing_images)
            images_to_generate = images_per_class - existing_images
            
            if images_to_generate <= 0:
                continue
                
            generator.manual_seed(seed + class_id)
            
            current_idx = existing_images
            while images_to_generate > 0:
                current_batch = min(batch_size, images_to_generate)
                class_labels = [class_id] * current_batch

                start_img_gen = time.time()
                
                output = pipe(
                    class_labels=class_labels,
                    generator=generator,
                    num_inference_steps=inference_step
                )

                total_time += time.time() - start_img_gen
                image_generated += current_batch
                
                for img in output.images:
                    img_name = f"class_{class_id:03d}_img_{current_idx:02d}.png"
                    img.save(os.path.join(output_dir, img_name))
                    current_idx += 1
                    
                images_to_generate -= current_batch
                pbar.update(current_batch)

    print(f"\nGeneration of {total_classes * images_per_class} images completed successfully!")

    return total_time, image_generated

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

    #TEMPORANEO, SOLO PER COMODITA
    model_id = "facebook/DiT-XL-2-256_v6"

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model_dir = f"../output/{model_id}"

    print(f"Loading pipeline for {model_id}...")
    if os.path.exists(model_dir):
        assert(os.path.exists(f"{model_dir}/quantization_config.json")),"The model need to have a quantization_config.json file"
        pipe, inference_step = load_quantized_pipeline(model_dir, device)
    else:
        pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
    
    total_time, img_generated = generate_fid_images(
        "FID_Images/"+model_id,
        pipe,
        inference_step,
        batch_size,     
        seed, 
        device,
        image_num
    )


    # Statistic saving
    json_path = f"../json/{model_id}.json"

    with open(json_path, "r") as J:
        json_file = json.load(J)

    json_file["generation"] = {
        "mean_time": total_time/img_generated,
        "RAM": 1000 #TODO
        }

    with open(json_path, "w") as J:
        json.dump(json_file,J,indent=4)

