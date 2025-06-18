# dataset.py
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch

class PairedIRDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, crop_size=50):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.filenames = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(('.jpg', '.png', '.bmp'))
        ])
        self.crop_size = crop_size
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        clean = Image.open(os.path.join(self.clean_dir, fname)).convert("L")
        noisy = Image.open(os.path.join(self.noisy_dir, fname.replace("Original", ""))).convert("L")

        clean = self.transform(clean)
        noisy = self.transform(noisy)
          _, h, w = clean.shape
        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        clean = clean[:, top:top+self.crop_size, left:left+self.crop_size]
        noisy = noisy[:, top:top+self.crop_size, left:left+self.crop_size]

        return noisy, clean