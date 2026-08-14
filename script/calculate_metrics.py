import os
import json
import argparse
import torch
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy import linalg
from datasets import load_dataset
import pytorch_fid.fid_score
from pytorch_fid.inception import InceptionV3
from torchmetrics.image.inception import InceptionScore
import pyiqa

OriginalDataset = pytorch_fid.fid_score.ImagePathDataset

class ResizingDataset(OriginalDataset):
    def __init__(self, files, transforms=None):
        super().__init__(files, transforms=transforms)
        self.custom_transforms = TF.Compose([
            TF.Resize(256, interpolation=TF.InterpolationMode.BICUBIC),
            TF.CenterCrop(256),
            TF.ToTensor()
        ])

    def __getitem__(self, i):
        from PIL import Image
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        return self.custom_transforms(img)

class ImageFolderDataset(Dataset):
    """Semplice dataloader per caricare le immagini da una cartella."""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


pytorch_fid.fid_score.ImagePathDataset = ResizingDataset

def download_imagenet_val(output_dir: str):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 50000:
        print(f"Folder '{output_dir}' already exists and contains the images.")
        
    print("Downloading ImageNet-1k dataset (Validation set, 50k images)...")
    print("WARNING: You might be required to login to HuggingFace.")
    print("If the download fails due to permissions, run 'huggingface-cli login' in the terminal.")
        
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")
    
    print(f"Saving images to {output_dir}")
    for i, item in enumerate(tqdm(dataset, desc="Saving Images")):
        img = item["image"].convert("RGB")
        img.save(os.path.join(output_dir, f"val_{i:05d}.png"))


def compute_fid(path_dataset: str, path_generated: str, batch_size: int, device: str, dims: int = 2048):

    if not path_dataset or not os.path.exists(path_dataset):
        print(f"CRITICAL ERROR: Dataset images path does not exist: {path_dataset}")
        return
        
    if not os.path.exists(path_generated):
        print(f"CRITICAL ERROR: Generated images path does not exist: {path_generated}")
        return
        
    print(f"\nStarting FID calculation...")
    print(f"Dataset 1 (Real / Statistics): {path_dataset}")
    print(f"Dataset 2 (Generated Images):   {path_generated}")
    
    paths = [path_dataset, path_generated]
    
    try:
        num_workers = min(os.cpu_count() or 8, 8)
        
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
    print(f"\n--- Calculating Inception Score (IS) ---")
    print(f"Loading images from: {path_generated}")
    
    # torchmetrics IS wants tensor uint8 in range [0, 255]
    transform = TF.Compose([
        TF.Resize(256), 
        TF.CenterCrop(256),
        TF.PILToTensor()
    ])
    
    dataset = ImageFolderDataset(path_generated, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 4, 4))
    
    isc = InceptionScore().to(device)
    
    for batch in tqdm(dataloader, desc="Calculating IS"):
        isc.update(batch.to(device))
        
    mean, std = isc.compute()
    print(f"Inception Score (IS): {mean.item():.4f} ± {std.item():.4f}")
    return mean.item(), std.item()


def calculate_sfid(path_dataset: str, path_generated: str, batch_size: int, device: str):
    print(f"\n--- Calculating spatial FID (sFID) ---")
    print(f"Real images: {path_dataset}")
    print(f"Generated images: {path_generated}")
    
    # Inizializza la metrica sFID di pyiqa (scarica i pesi se necessario)
    sfid_metric = pyiqa.create_metric('sfid', device=device)
    
    # pyiqa accetta direttamente i path delle cartelle
    print("Extracting features and computing sFID...")
    sfid_score_tensor = sfid_metric(path_dataset, path_generated)
    sfid_value = sfid_score_tensor.item()
    
    print(f"spatial FID (sFID) SCORE: {sfid_value:.4f}")
    return sfid_value






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

    json_path = f"json/{model_id}.json"
    if not os.path.exists(json_path):
        json_path = "../" + json_path

    assert(os.path.exists(json_path)),f"The model doesn't have a json to save the scores at {json_path}"

    if download:
        download_imagenet_val(path_dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    fid_score = compute_fid(path_dataset, path_generated, batch_size, device)
    is_mean, is_std = calculate_inception_score(path_generated, batch_size, device)
    sfid_score = calculate_sfid(path_dataset, path_generated, batch_size, device)
    
    # Statistic saving
    with open(json_path, "r") as J:
        json_file = json.load(J)

    json_file["metrics"] = {
        "FID": fid_score,
        "sFID": sfid_score,
        "IS (mean)": is_mean,
        "IS (std)": is_std
        }

    with open(json_path, "w") as J:
        json.dump(json_file,J,indent=4)
