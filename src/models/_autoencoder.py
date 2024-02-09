import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.encode = nn.Sequential(
            # 32x32
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(hidden_dim * 2)
            # Add more layers or residual blocks if needed
        )

    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dim):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            # noise: 2048 => (batch_size, 128, 4, 4)
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            ResidualBlock(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Use Tanh for generating images, or another suitable activation
            # Add more layers or residual blocks if needed
        )

    def forward(self, x):
        return self.decode(x)
