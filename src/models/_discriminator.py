import torch
import torch.nn as nn
import timm

from ._generator import StackedDecodingModule


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Base model for feature extraction
        self.base_model = timm.create_model(
            # "efficientnet_b0", pretrained=False, num_classes=1
            # "efficientnet_b1", pretrained=False, num_classes=1
            "edgenext_xx_small",
            pretrained=False,
            num_classes=1,
            # "eca_nfnet_l0", pretrained=False, num_classes=1
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.sigmoid(x)
        return x


class DiscriminatorCustom(nn.Module):
    def __init__(self):
        super().__init__()

        self.pipeline = nn.Sequential(
            StackedDecodingModule(3, 32, norm_layer=nn.BatchNorm2d),
            StackedDecodingModule(32, 64, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            StackedDecodingModule(64, 128, norm_layer=nn.BatchNorm2d),
            StackedDecodingModule(128, 256, norm_layer=nn.BatchNorm2d),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            StackedDecodingModule(256, 512, norm_layer=nn.BatchNorm2d),
            StackedDecodingModule(512, 256, norm_layer=nn.BatchNorm2d),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
        )

    def forward(self, x):
        x = self.pipeline(x)
        x = torch.sigmoid(x)
        return x
