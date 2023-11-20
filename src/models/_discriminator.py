import torch
import torch.nn as nn
import timm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Base model for feature extraction
        self.base_model = timm.create_model(
            # "efficientnet_b0", pretrained=False, num_classes=1
            # "efficientnet_b1", pretrained=False, num_classes=1
            "edgenext_xx_small", pretrained=False, num_classes=1
            # "eca_nfnet_l0", pretrained=False, num_classes=1
        )

    def forward(self, x):
        x = self.base_model(x)
        x = torch.sigmoid(x)
        return x
