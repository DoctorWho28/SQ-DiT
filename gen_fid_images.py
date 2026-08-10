import os
import torch
from tqdm import tqdm
from inference import load_quantized_pipeline
from diffusers import DiTPipeline

def generate_fid_images(
    output_dir: str,
    pipe: DiTPipeline,
    inference_step: int,
    batch_size: int = 4,
    images_per_class: int = 50,
    seed: int = 42
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        
    generator = torch.Generator(device=device)
    total_classes = 1000
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting generation of {total_classes * images_per_class} images in '{output_dir}'...")
    print(f"Batch size: {batch_size}. This operation will take several hours.")
    print("Note: The script supports RESUME. If interrupted, it will resume from where it stopped by skipping already generated images.")
    
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
                
                output = pipe(
                    class_labels=class_labels,
                    generator=generator,
                    num_inference_steps=inference_step
                )
                
                for img in output.images:
                    img_name = f"class_{class_id:03d}_img_{current_idx:02d}.png"
                    img.save(os.path.join(output_dir, img_name))
                    current_idx += 1
                    
                images_to_generate -= current_batch
                pbar.update(current_batch)

    print(f"\nGeneration of {total_classes * images_per_class} images completed successfully!")

if __name__ == "__main__":
    
    quant_dir = "output/facebook/DiT-XL-2-256_v6"
    print(f"Loading pipeline from {quant_dir}...")
    if os.path.exists(os.path.join(quant_dir, "quantization_config.json")):
        pipe, inference_step = load_quantized_pipeline(quant_dir=quant_dir)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiTPipeline.from_pretrained(quant_dir, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        inference_step = 20
    
    generate_fid_images(
        output_dir="FID_Images/"+quant_dir,
        pipe= pipe,
        inference_step=inference_step,
        batch_size=4,       
        images_per_class=4, 
        seed=42
    )

