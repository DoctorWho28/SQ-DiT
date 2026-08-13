import os
import sys
import torch
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy import linalg

# Richiede pytorch-fid per estrarre le feature dal modello Inception
try:
    from pytorch_fid.inception import InceptionV3
except ImportError:
    print("ERROR: 'pytorch-fid' is not installed.")
    print("Install it with: pip install pytorch-fid")
    sys.exit(1)

# Richiede torchmetrics per calcolare l'Inception Score
try:
    from torchmetrics.image.inception import InceptionScore
except ImportError:
    print("ERROR: 'torchmetrics' is required for Inception Score.")
    print("Install it with: pip install torchmetrics")
    sys.exit(1)


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


def calculate_inception_score(generated_path, batch_size=32, device="cuda"):
    """
    Calcola l'Inception Score (IS) usando torchmetrics.
    Valuta sia la qualità che la diversità delle immagini generate.
    Non richiede immagini reali, solo quelle generate.
    """
    print(f"\n--- Calculating Inception Score (IS) ---")
    print(f"Loading images from: {generated_path}")
    
    # torchmetrics IS richiede tensori uint8 nel range [0, 255]
    transform = TF.Compose([
        TF.Resize(256), 
        TF.CenterCrop(256),
        TF.PILToTensor()
    ])
    
    dataset = ImageFolderDataset(generated_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 4, 4))
    
    isc = InceptionScore().to(device)
    
    for batch in tqdm(dataloader, desc="Calculating IS"):
        isc.update(batch.to(device))
        
    mean, std = isc.compute()
    print("\n" + "★"*50)
    print(f"🏆 Inception Score (IS): {mean.item():.4f} ± {std.item():.4f}")
    print("★"*50 + "\n")
    return mean.item(), std.item()


def compute_sfid_statistics(path, batch_size=32, device="cuda"):
    """
    Calcola la media e la matrice di covarianza per lo spatial FID (sFID).
    Per evitare errori di Out Of Memory, calcoliamo le statistiche in modo 
    iterativo accumulando la somma e la somma dei prodotti (Outer Product).
    """
    transform = TF.Compose([
        TF.Resize(256),
        TF.CenterCrop(256),
        TF.ToTensor() # Restituisce float nel range [0, 1] richiesto da InceptionV3
    ])
    
    dataset = ImageFolderDataset(path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count() or 4, 4))
    
    # Il blocco 2 di pytorch_fid.inception corrisponde a Mixed_7c.
    # Restituisce le feature spaziali prima del pooling: shape (B, 2048, 8, 8)
    model = InceptionV3([2]).to(device)
    model.eval()
    
    # Accumulatori in float64 per stabilità numerica (le feature sono 2048)
    sum_x = torch.zeros(2048, dtype=torch.float64, device=device)
    sum_xx = torch.zeros((2048, 2048), dtype=torch.float64, device=device)
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting sFID stats for {os.path.basename(path)}"):
            batch = batch.to(device)
            feat = model(batch)[0] 
            
            # Trasformiamo da (B, 2048, H, W) a (B*H*W, 2048)
            feat = feat.permute(0, 2, 3, 1).reshape(-1, 2048).to(torch.float64)
            
            sum_x += feat.sum(dim=0)
            sum_xx += feat.T @ feat
            num_samples += feat.shape[0]
            
    # Calcolo della media
    mu = (sum_x / num_samples).cpu().numpy()
    
    # Calcolo della covarianza iterativa: E[X * X^T] - E[X] * E[X]^T
    sigma = (sum_xx / (num_samples - 1)) - (torch.outer(sum_x, sum_x) / (num_samples * (num_samples - 1)))
    sigma = sigma.cpu().numpy()
    
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calcola la distanza di Fréchet basata su mu e sigma."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_sfid(path_real, path_generated, batch_size=32, device="cuda"):
    """
    Calcola lo spatial Fréchet Inception Distance (sFID).
    Rispetto al FID normale che prende il vettore aggregato (2048x1x1),
    lo sFID prende in considerazione le mappe spaziali intermedie (8x8x2048), 
    rendendolo molto più sensibile alla struttura spaziale delle immagini.
    """
    print(f"\n--- Calculating spatial FID (sFID) ---")
    print(f"Real images: {path_real}")
    print(f"Generated images: {path_generated}")
    
    mu_real, sigma_real = compute_sfid_statistics(path_real, batch_size, device)
    mu_gen, sigma_gen = compute_sfid_statistics(path_generated, batch_size, device)
    
    sfid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print("\n" + "★"*50)
    print(f"🏆 spatial FID (sFID) SCORE: {sfid_value:.4f}")
    print("★"*50 + "\n")
    return sfid_value

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURAZIONE DEI PERCORSI DA MODIFICARE
    # =========================================================================
    
    # 1. PERCORSO IMMAGINI REALI
    PATH_REAL = "imagenet/imagenet-val-flat"
    
    # 2. PERCORSO IMMAGINI GENERATE
    PATH_GENERATED = "fid_50k_quantized_v6"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    
    try:
        # 1. Calcolo dell'Inception Score (IS)
        calculate_inception_score(PATH_REAL, batch_size=batch_size, device=device)
        
        # 2. Calcolo dello spatial FID (sFID)
        calculate_sfid(PATH_REAL, PATH_REAL, batch_size=batch_size, device=device)
        
    except FileNotFoundError as e:
        print(f"\nERRORE: Impossibile trovare la cartella. {e}")
        print("Assicurati di aver impostato correttamente PATH_REAL e PATH_GENERATED.")
