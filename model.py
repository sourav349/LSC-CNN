import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Residual Block with 128 Filters
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

# -----------------------------
# LSC_CNN Model
# -----------------------------
class LSC_CNN(nn.Module):
     def __init__(self):
        super().__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 128, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # Intermediate convolution layers
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Residual blocks with 128 channels
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(20)])

        # Upsampling and final refinement layers
        self.conv21 = nn.Conv2d(1 + 128, 64, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        inp = x  # Save the original input

        # Encoding
        x = self.relu1(self.conv1(x))
         x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # Residual feature extraction
        x = self.res_blocks(x)

        # Upsampling to match input size
        up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Match size with original input
        h = min(inp.shape[2], up.shape[2])
        w = min(inp.shape[3], up.shape[3])
        inp = inp[:, :, :h, :w]
        up = up[:, :, :h, :w]

        # Element-wise multiplication and concatenation
        enhanced = up * inp
        concat = torch.cat((inp, enhanced), dim=1)

        # Refinement and noise prediction
        x = self.relu21(self.conv21(concat))
        noise = self.conv22(x)

        # Final clean image
        return inp - noise