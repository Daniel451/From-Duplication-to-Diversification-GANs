import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from functools import partial
from loguru import logger


def _resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class CustomLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer=None,
        activation_function=nn.LeakyReLU,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            # options:encoding, decoding,keep the same
            # decide the output channel based on your goal
            # decoding might more sense since we want diversity in the output
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        match norm_layer:
            case None:
                self.norm_layer = nn.Identity()
            case nn.InstanceNorm2d | nn.BatchNorm2d:
                self.norm_layer = norm_layer(out_channels)
            case nn.LocalResponseNorm:
                self.norm_layer = norm_layer(size=2)
            case nn.utils.spectral_norm:
                self.conv = nn.utils.spectral_norm(self.conv)
                self.norm_layer = nn.Identity()
                

        self.sequential = nn.Sequential(
            # TODO: Adaptive Instance Normalization
            self.conv,
            self.norm_layer,
            activation_function(),
        )

    def forward(self, x):
        # print("x.shape", x.shape)
        out = self.sequential(x)
        # print("out.shape", out.shape)
        return out


class DecodingModule(nn.Module):
    """Decoding module that grows to twice its input channel size"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer=nn.InstanceNorm2d,
        activation_function=nn.LeakyReLU,
    ):
        super().__init__()

        Layer = partial(
            CustomLayer, norm_layer=norm_layer, activation_function=activation_function
        )

        self.sequential = nn.Sequential(
            Layer(in_channels=in_channels + 1, out_channels=int(in_channels * 1.5)),
            Layer(
                in_channels=int(in_channels * 1.5), out_channels=int(in_channels * 1.5)
            ),
            Layer(in_channels=int(in_channels * 1.5), out_channels=out_channels),
        )

        self.shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        local_noise = torch.rand((b, 1, h, w)).to(x.device)

        sequential_in = torch.cat((x, local_noise), dim=1)

        # print("x.shape", x.shape)
        # feed input to sequential module => compute output features
        sequential_out = self.sequential(sequential_in)
        # print("sequential.shape", sequential_out.shape)
        # transform input to match feature match size of output layer of sequential module
        transformed_input = self.shortcut(x)
        # print("transformed_input.shape", transformed_input.shape)

        # residual output
        return transformed_input + sequential_out


class StackedDecodingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer=nn.InstanceNorm2d,
        activation_function=nn.LeakyReLU,
    ):
        super().__init__()

        self.decode1 = DecodingModule(
            in_channels=in_channels,
            out_channels=int(in_channels / 2),
            norm_layer=norm_layer,
            activation_function=activation_function,
        )
        self.decode2 = DecodingModule(
            in_channels=int(in_channels / 2),
            out_channels=out_channels,
            norm_layer=norm_layer,
            activation_function=activation_function,
        )
        self.shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        out1 = self.decode1(x)
        out2 = self.decode2(out1)
        transformed_input = self.shortcut(x)

        residual = (transformed_input + out2) / 2

        return residual


class AdaIN(nn.Module):
    def __init__(self, style_dim, content_dim):
        super().__init__()
        # Define layers to learn affine transformation parameters
        # Adjust style_dim to be style_dim * height * width
        self.style_scale_transform = nn.Linear(style_dim * 2 * 2, content_dim)
        self.style_shift_transform = nn.Linear(style_dim * 2 * 2, content_dim)

    def forward(self, content, style):
        # Flatten the style tensor
        batch_size = style.size(0)
        style = style.view(batch_size, -1)

        style_scale = self.style_scale_transform(style).unsqueeze(-1).unsqueeze(-1)
        style_shift = self.style_shift_transform(style).unsqueeze(-1).unsqueeze(-1)
        normalized_content = F.instance_norm(content)
        return normalized_content * style_scale + style_shift


class FiLM(nn.Module):
    def __init__(self, noise_dim, num_features):
        super().__init__()
        # Adjust the input dimension of the linear layers
        self.scale_transform = nn.Linear(noise_dim * 2 * 2, num_features)
        self.shift_transform = nn.Linear(noise_dim * 2 * 2, num_features)

    def forward(self, features, noise):
        # Flatten the noise tensor
        batch_size = noise.size(0)
        noise = noise.view(
            batch_size, -1
        )  # Reshape noise to [batch_size, noise_dim*2*2]

        scale = self.scale_transform(noise).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift_transform(noise).unsqueeze(-1).unsqueeze(-1)
        return scale * features + shift


class Generator(nn.Module):
    def __init__(self, device):
        super().__init__()

        # CustomDecodeModule = partial(
        #     DecodingModule,
        #     # norm_layer=None,
        #     # norm_layer=nn.InstanceNorm2d,
        #     norm_layer=nn.BatchNorm2d,
        #     # norm_layer=nn.LocalResponseNorm,
        #     activation_function=nn.LeakyReLU,
        # )
        CustomStackedDecodeModule = partial(
            StackedDecodingModule,
            # norm_layer=None,
            # norm_layer=nn.InstanceNorm2d,
            norm_layer=nn.BatchNorm2d,
            # norm_layer=nn.LocalResponseNorm,
            activation_function=nn.LeakyReLU,
        )

        # feature extractor for processing input images
        self.feature_extractor = timm.create_model(
            # "efficientnet_b0",
            "edgenext_xx_small",
            pretrained=True,
            features_only=True,
            # TODO: test diffrent output indices for feature extraction
            # out_indices=[3], # efficientnet b0
            out_indices=[2],  # edgenext_xx_small
        ).to(device)
        self.adain = AdaIN(style_dim=56, content_dim=88).to(device)
        # self.film = FiLM(noise_dim=56, num_features=88).to(device)

        self.noise_transform = nn.Sequential(
            CustomStackedDecodeModule(in_channels=56, out_channels=56),
            CustomStackedDecodeModule(in_channels=56, out_channels=56),
            CustomStackedDecodeModule(in_channels=56, out_channels=88),
        )

        # generative module
        self.generative = nn.Sequential(
            # TODO: input resolution: ???
            # TODO: figure out if upsampling or downsampling is needed (e.g.) timm output is too large or too small
            # *LAZY* conv2d layer which automatically calculates number of in_channels
            # from merged and outputs the specified channel
            nn.LazyConv2d(out_channels=256, kernel_size=1, padding=0),
            # 2x2
            CustomStackedDecodeModule(in_channels=256, out_channels=128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 4x4
            CustomStackedDecodeModule(in_channels=128, out_channels=64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 8x8
            CustomStackedDecodeModule(in_channels=64, out_channels=32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 16x16
            CustomStackedDecodeModule(in_channels=32, out_channels=16),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 32x32
            CustomStackedDecodeModule(in_channels=16, out_channels=8),
            nn.Conv2d(
                in_channels=8,
                out_channels=3,
                kernel_size=3,
                padding=1,
            ),
        )

    def get_generative_parameters(self):
        """Returns parameters of the generative module"""
        # return self.generative.parameters()
        return [*self.generative.parameters(), *self.noise_transform.parameters()]

    def forward(self, img, noise):
        # extract features from image
        features = self.feature_extractor(img)[0]
        # TODO: test the need of subnetwork for noise processing
        transformed_noise = self.noise_transform(noise)

        # merge noise with features
        # => noise and features need to have *same* dimensions if concatenated
        # => merging at dim=1 means concat at channel dim => (b, c, h, w)
        # TODO: check out adaptive instance normalization
        #
        merged = torch.cat((features, transformed_noise), dim=1)

        # compute output image
        output_img = self.generative(merged)
        # sigmoid_output_img = torch.sigmoid(output_img)
        transformed_output = torch.tanh(output_img)

        return transformed_output

