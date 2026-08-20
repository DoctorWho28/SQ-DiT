import os
from dotenv import load_dotenv
load_dotenv(".env")
import json
import argparse
import torch
import random
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
import pytorch_fid.fid_score
from torchmetrics.image.inception import InceptionScore
import pyiqa
from clip_mmd import logic
from inference import load_quantized_pipeline
import contextlib
import time
from diffusers import DiTPipeline

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

class PatchedImagePathDataset(pytorch_fid.fid_score.ImagePathDataset):
    def __init__(self, files, transforms=None):
        super().__init__(sorted(files), transforms)
        self.transforms = TF.Compose([
            TF.Resize(256),
            TF.CenterCrop(256),
            TF.ToTensor()
        ])

pytorch_fid.fid_score.ImagePathDataset = PatchedImagePathDataset
OriginalDataset = PatchedImagePathDataset


class ImageFolderDataset(Dataset):
    """Dataloader to load image from a folder"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        self.files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                             if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img



def download_imagenet_val(output_dir: str):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 10000:
        print(f"Folder '{output_dir}' already exists and contains the images.")
        return # Esce dalla funzione se abbiamo già scaricato le 10k immagini
        
    print("Downloading ImageNet-1k dataset (Validation set, 10k images)...")
    print("WARNING: You might be required to login to HuggingFace.")
    print("If the download fails due to permissions, run 'huggingface-cli login' in the terminal.")
        
    os.makedirs(output_dir, exist_ok=True)
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Token HuggingFace trovato, autenticazione in corso...")
    
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", token=hf_token, streaming=True)
    
    print(f"Saving images to {output_dir}")
    
    class_counts = {i: 0 for i in range(1000)}
    total_saved = 0
    
    with tqdm(total=10000, desc="Saving Images") as pbar:
        for item in dataset:
            if total_saved >= 10000:
                break
                
            label = item["label"]
            if class_counts[label] < 10:
                img = item["image"].convert("RGB")
                img.save(os.path.join(output_dir, f"val_class_{label:03d}_{class_counts[label]:02d}.png"))
                class_counts[label] += 1
                total_saved += 1
                pbar.update(1)


def compute_fid(path_dataset: str, path_generated: str, batch_size: int, device: str, dims: int = 2048):    
    paths = [path_dataset, path_generated]
    
    try:
        num_workers = 0 if os.name == 'nt' else min(os.cpu_count() or 8, 8)
        
        fid_value = pytorch_fid.fid_score.calculate_fid_given_paths(
            paths=paths,
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=num_workers
        )
        
        print(f"FID SCORE RESULT: {fid_value:.4f}")
        
        return fid_value
        
    except Exception as e:
        print(f"\nAn error occurred during FID calculation: {e}")


def calculate_inception_score(path_generated: str, batch_size: int, device: str):
    
    transform = TF.Compose([
        TF.Resize(256), 
        TF.CenterCrop(256),
        TF.PILToTensor()
    ])
    
    dataset = ImageFolderDataset(path_generated, transform=transform)
    num_workers_is = 0 if os.name == 'nt' else min(os.cpu_count() or 4, 4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_is)
    
    isc = InceptionScore().to(device)
    
    for batch in dataloader:
        isc.update(batch.to(device))
        
    mean, std = isc.compute()
    print(f"Inception Score (IS): {mean.item():.4f} ± {std.item():.4f}")
    return mean.item(), std.item()


def calculate_sfid(path_dataset: str, path_generated: str, device: str):
    sfid_metric = pyiqa.create_metric('sfid', device=device)
    
    sfid_score_tensor = sfid_metric(path_dataset, path_generated)
    sfid_value = sfid_score_tensor.item()
    
    print(f"spatial FID (sFID) SCORE: {sfid_value:.4f}")
    return sfid_value


def calculate_cmmd(path_dataset: str, path_generated: str, device: str):
    
    try:
        # Passiamo semplicemente "cuda" o "cpu" alla libreria CMMD
        prep = logic.CMMD(device=device) 
        
        cmmd_score = prep.execute(path_dataset, path_generated)
        print(f"CMMD SCORE: {cmmd_score:.4f}")
        return float(cmmd_score)
        
    except Exception as e:
        print(f"\nAn error occurred during CMMD calculation: {e}")
        return None

def calculate_generation_values(quant_dir: str, model_id: str, device: str):
    config_path = os.path.join(quant_dir, "quantization_config.json")
    if os.path.exists(config_path):
        pipe, inference_step = load_quantized_pipeline(quant_dir, device)
    else:
        pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        inference_step = 20

    total_info = []
    
    for i in tqdm(range(100), desc="Generating image"):
        seed = random.randint(0, 1000000)
        generator = torch.Generator(device=device).manual_seed(seed)

        with track_info() as info:
            output = pipe(class_labels=[0], generator=generator, num_inference_steps=inference_step)

        total_info.append(info)

    vram_max = max(x["vram"] for x in total_info)
    time_mean = sum(x["time"] for x in total_info) / len(total_info)
    return vram_max, time_mean
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_id", type=str ,required=True, help="Model id (required)")
    parser.add_argument("-pd", "--path_dataset", type=str, required=True, help="Path to dataset ImageNet (also where the download is placed)")
    parser.add_argument("-pg", "--path_generated", type=str, required=True, help="Path to generated imaged")
    parser.add_argument("-d", "--download", action="store_true", help="Download the images if activated")
    parser.add_argument("-bs", "--batch_size", type=int, required=False, default=1, help="Batch size")


    args = parser.parse_args()
    model_id = args.model_id
    path_dataset = args.path_dataset
    path_generated = args.path_generated
    download = args.download
    batch_size = args.batch_size

    quant_dir = f"output/{model_id}"

    json_path = f"json/{model_id}.json"
    if not os.path.exists(json_path):
        json_path = "../" + json_path

    assert(os.path.exists(json_path)),f"The model doesn't have a json to save the scores at {json_path}"

    if download:
        download_imagenet_val(path_dataset)

    assert(os.path.exists(path_dataset)),f"The path of dataset doesn't exists"
    assert(os.path.exists(path_generated)),f"The path of generated images desn't exists"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Rendi PyTorch deterministico
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    with open(json_path, "r") as J:
        json_file = json.load(J)
        
    if "metrics" not in json_file:
        json_file["metrics"] = {}
        
    metrics = json_file["metrics"]
    
    def save_json():
        with open(json_path, "w") as J:
            json.dump(json_file, J, indent=4)

    # 1. FID
    if "FID" not in metrics or metrics["FID"] is None:
        print("\n--- FID ---")
        metrics["FID"] = compute_fid(path_dataset, path_generated, batch_size, device)
        save_json()
    else:
        print(f"\n--- FID: skipped ({metrics['FID']:.4f}) ---")

    # 2. Inception Score
    if "IS (mean)" not in metrics or metrics["IS (mean)"] is None:
        print("\n--- Inception Score (IS) ---")
        is_mean, is_std = calculate_inception_score(path_generated, batch_size, device)
        metrics["IS (mean)"] = is_mean
        metrics["IS (std)"] = is_std
        save_json()
    else:
        print(f"\n--- Inception Score: skipped ({metrics['IS (mean)']:.4f} +- {metrics['IS (std)']:.4f}) ---")

    # 3. sFID
    if "sFID" not in metrics or metrics["sFID"] is None:
        print("\n--- sFID ---")
        metrics["sFID"] = calculate_sfid(path_dataset, path_generated, device)
        save_json()
    else:
        print(f"\n--- sFID: skipped ({metrics['sFID']:.4f}) ---")

    # 4. CMMD
    if "CMMD" not in metrics or metrics["CMMD"] is None:
        print("\n--- CMMD ---")
        metrics["CMMD"] = calculate_cmmd(path_dataset, path_generated, device)
        save_json()
    else:
        print(f"\n--- CMMD: skipped ({metrics['CMMD']:.4f}) ---")

    # 5. Generation values
    if "vram_max" not in metrics or metrics["vram_max"] is None:
        print("\n--- Generation values ---")
        vram_max, time_mean = calculate_generation_values(quant_dir, model_id, device)
        metrics["vram_max"] = vram_max
        metrics["time_mean"] = time_mean
        save_json()
    else:
        print(f"\n--- Generation values: skipped (VRAM: {metrics['vram_max']:.4f}, time: {metrics['time_mean']:.4f}) ---")
