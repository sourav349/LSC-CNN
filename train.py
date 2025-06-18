import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#from model import LSC_CNN
#from dataset import PairedIRDataset

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    psnr_vals, ssim_vals = [], []
    for i in range(pred_np.shape[0]):
        psnr_vals.append(peak_signal_noise_ratio(target_np[i], pred_np[i], data_range=1.0))
        ssim_vals.append(structural_similarity(target_np[i], pred_np[i], data_range=1.0))
    return np.mean(psnr_vals), np.mean(ssim_vals)

# -----------------------------
# ✅ Updated Paths
# -----------------------------
train_clean_dir = "/home/himanshu/Desktop/NUC_DATA/newfirstoriginal"
train_noisy_dir = "/home/himanshu/Desktop/NUC_DATA/sigmareduced"
val_clean_dir   = "/home/himanshu/Desktop/NUC_DATA/newfirstoriginal"
val_noisy_dir   = "/home/himanshu/Desktop/NUC_DATA/newfirstnoise"
checkpoint_dir  = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Hyperparameters
batch_size = 10      # 16
num_epochs = 25
lr = 1e-3
crop_size = 50

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSC_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Load data
train_dataset = PairedIRDataset(train_clean_dir, train_noisy_dir, crop_size)
val_dataset   = PairedIRDataset(val_clean_dir, val_noisy_dir, crop_size)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Training
best_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        loss = loss_fn(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, psnr_total, ssim_total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            val_loss += loss_fn(pred, clean).item() * noisy.size(0)
            psnr, ssim = compute_metrics(pred, clean)
            psnr_total += psnr * noisy.size(0)
            ssim_total += ssim * noisy.size(0)

    val_loss /= len(val_loader.dataset)
    psnr_total /= len(val_loader.dataset)
    ssim_total /= len(val_loader.dataset)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f"{checkpoint_dir}/best_modelone.pth")
        print(f"✅ Best model saved at epoch {epoch+1} — Val Loss: {val_loss:.6f}")

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {psnr_total:.2f} | SSIM: {ssim_total:.4f}")

torch.save(model.state_dict(), f"{checkpoint_dir}/final_modelone.pth")
print(f"✅ Final model saved to {checkpoint_dir}/final_modelone.pth")