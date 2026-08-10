import os
import sys
import torch
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    print("ERROR: 'pytorch-fid' library is not installed.")
    print("Please, open the terminal and install it by running:")
    print("pip install pytorch-fid")
    sys.exit(1)

import torchvision.transforms as TF
import pytorch_fid.fid_score

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

pytorch_fid.fid_score.ImagePathDataset = ResizingDataset

def download_imagenet_val(output_dir="imagenet_val_cache"):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= 50000:
        print(f"Folder '{output_dir}' already exists and contains the images.")
        return output_dir
        
    print("Downloading ImageNet-1k dataset (Validation set, 50k images)...")
    print("WARNING: You might be required to login to HuggingFace.")
    print("If the download fails due to permissions, run 'huggingface-cli login' in the terminal.")
    try:
        from datasets import load_dataset
        import tqdm
    except ImportError:
        print("ERROR: Install 'datasets' and 'tqdm' to download automatically:")
        print("pip install datasets tqdm")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")
    
    print("Saving images to disk...")
    for i, item in enumerate(tqdm.tqdm(dataset, desc="Saving Images")):
        img = item["image"].convert("RGB")
        img.save(os.path.join(output_dir, f"val_{i:05d}.png"))
        
    return output_dir


def compute_fid(path_real: str, path_generated: str, batch_size: int = 50, device: str = "cuda", dims: int = 2048):
    if not path_real or not os.path.exists(path_real):
        print(f"Real data path '{path_real}' is invalid or not provided.")
        print("Starting automatic download of ImageNet Validation Set from HuggingFace...")
        path_real = download_imagenet_val("imagenet_val_cache")
        print(f"Real images path automatically set to: {path_real}")
        
    if not os.path.exists(path_generated):
        print(f"CRITICAL ERROR: Generated images path does not exist: {path_generated}")
        return
        
    print(f"\nStarting FID calculation...")
    print(f"Dataset 1 (Real / Statistics): {path_real}")
    print(f"Dataset 2 (Generated Images):   {path_generated}")
    print(f"Extracting features using InceptionV3 (this may take several minutes)...\n")
    
    paths = [path_real, path_generated]
    
    try:
        num_workers = min(os.cpu_count() or 8, 8) 
        
        fid_value = calculate_fid_given_paths(
            paths=paths,
            batch_size=batch_size,
            device=device,
            dims=dims,
            num_workers=num_workers
        )
        
        print("\n" + "★"*50)
        print(f"🏆 FID SCORE RESULT: {fid_value:.4f}")
        print("★"*50 + "\n")
        
        return fid_value
        
    except Exception as e:
        print(f"\nAn error occurred during FID calculation: {e}")

if __name__ == "__main__":
    PATH_REAL = "imagenet/imagenet-val-flat"
    PATH_GENERATED = "fid_50k_quantized_v6"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    compute_fid(
        path_real=PATH_REAL,
        path_generated=PATH_REAL,
        batch_size=50,
        device=device
    )
