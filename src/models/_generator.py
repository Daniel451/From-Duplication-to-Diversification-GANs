import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from functools import partial


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

        match norm_layer:
            case None:
                self.norm_layer = nn.Identity()
            case nn.InstanceNorm2d | nn.BatchNorm2d:
                self.norm_layer = norm_layer(out_channels)
            case nn.LocalResponseNorm:
                self.norm_layer = norm_layer(size=2)

        self.sequential = nn.Sequential(
            # 1st layer
            nn.Conv2d(
                in_channels=in_channels,
                # options:encoding, decoding,keep the same
                # decide the output channel based on your goal
                # decoding might more sense since we want diversity in the output
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            # TODO: Adaptive Instance Normalization
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
        # self.feature_extractor = timm.create_model(
        #     # "efficientnet_b0",
        #     "edgenext_xx_small",
        #     pretrained=True,
        #     features_only=True,
        #     # TODO: test diffrent output indices for feature extraction
        #     # out_indices=[3], # efficientnet b0
        #     out_indices=[2],  # edgenext_xx_small
        # ).to(device)

        # generative module
        self.generative = nn.Sequential(
            # TODO: input resolution: ???
            # TODO: figure out if upsampling or downsampling is needed (e.g.) timm output is too large or too small
            # *LAZY* conv2d layer which automatically calculates number of in_channels
            # from merged and outputs the specified channel
            nn.LazyConv2d(out_channels=128, kernel_size=1, padding=0),
            # 2x2
            CustomStackedDecodeModule(in_channels=128, out_channels=64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 4x4
            CustomStackedDecodeModule(in_channels=64, out_channels=32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 8x8
            CustomStackedDecodeModule(in_channels=32, out_channels=16),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 16x16
            CustomStackedDecodeModule(in_channels=16, out_channels=16),
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
        return self.generative.parameters()

    def forward(self, img, noise):
        # extract features from image
        # features = self.feature_extractor(img)[0]
        # TODO: test the need of subnetwork for noise processing

        # merge noise with features
        # => noise and features need to have *same* dimensions if concatenated
        # => merging at dim=1 means concat at channel dim => (b, c, h, w)
        # TODO: check out adaptive instance normalization
        # merged = torch.cat((features, noise), dim=1)

        # compute output image
        output_img = self.generative(noise)
        # sigmoid_output_img = torch.sigmoid(output_img)
        transformed_output = torch.tanh(output_img)

        return transformed_output


class Generator2(nn.Module):
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

        # generative module
        # TODO: input resolution: ???
        # TODO: figure out if upsampling or downsampling is needed (e.g.) timm output is too large or too small
        # *LAZY* conv2d layer which automatically calculates number of in_channels
        # from merged and outputs the specified channel
        self.gen_conv1 = nn.LazyConv2d(out_channels=128, kernel_size=1, padding=0)
        # 2x2
        self.gen_sdm1 = CustomStackedDecodeModule(in_channels=128, out_channels=64)
        self.gen_shortcut1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
        self.gen_shortcut1_norm = nn.BatchNorm2d(64)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 4x4
        self.gen_sdm2 = CustomStackedDecodeModule(in_channels=64, out_channels=32)
        self.gen_shortcut2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.gen_shortcut2_norm = nn.BatchNorm2d(32)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 8x8
        self.gen_sdm3 = CustomStackedDecodeModule(in_channels=32, out_channels=16)
        self.gen_shortcut3 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.gen_shortcut3_norm = nn.BatchNorm2d(16)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 16x16
        self.gen_sdm4 = CustomStackedDecodeModule(in_channels=16, out_channels=16)
        self.gen_shortcut4 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.gen_shortcut4_norm = nn.BatchNorm2d(16)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 32x32
        self.gen_sdm5 = CustomStackedDecodeModule(in_channels=16, out_channels=8)
        self.gen_shortcut5 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)
        self.gen_shortcut5_norm = nn.BatchNorm2d(8)
        # output
        self.gen_output_conv = nn.Conv2d(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )

        self.generative = nn.ModuleList(
            [
                self.gen_conv1,
                self.gen_sdm1,
                self.gen_shortcut1,
                self.up1,
                self.gen_sdm2,
                self.gen_shortcut2,
                self.up2,
                self.gen_sdm3,
                self.gen_shortcut3,
                self.up3,
                self.gen_sdm4,
                self.gen_shortcut4,
                self.up4,
                self.gen_sdm5,
                self.gen_shortcut5,
                self.gen_output_conv,
            ]
        )

    def get_generative_parameters(self):
        """Returns parameters of the generative module"""
        return self.generative.parameters()

    def forward(self, img, noise):
        # extract features from image
        features = self.feature_extractor(img)[0]
        # TODO: test the need of subnetwork for noise processing

        # merge noise with features
        # => noise and features need to have *same* dimensions if concatenated
        # => merging at dim=1 means concat at channel dim => (b, c, h, w)
        # TODO: check out adaptive instance normalization
        merged = torch.cat((features, noise), dim=1)

        # generative part
        x = self.gen_conv1(merged)

        # 2x2
        x = self.gen_sdm1(x) + torch.tanh(
            self.gen_shortcut1_norm(self.gen_shortcut1(_resize(img, size=(2, 2))))
        )
        x = self.up1(x)

        # 4x4
        x = self.gen_sdm2(x) + torch.tanh(
            self.gen_shortcut2_norm(self.gen_shortcut2(_resize(img, size=(4, 4))))
        )
        x = self.up2(x)

        # 8x8
        x = self.gen_sdm3(x) + torch.tanh(
            self.gen_shortcut3_norm(self.gen_shortcut3(_resize(img, size=(8, 8))))
        )
        x = self.up3(x)

        # 16x16
        x = self.gen_sdm4(x) + torch.tanh(
            self.gen_shortcut4_norm(self.gen_shortcut4(_resize(img, size=(16, 16))))
        )
        x = self.up4(x)

        # 32x32
        x = self.gen_sdm5(x)  # + self.gen_shortcut5(_resize(img, size=(32, 32)))

        output_img = self.gen_output_conv(x)

        # sigmoid_output_img = torch.sigmoid(output_img)
        transformed_output = torch.tanh(output_img)

        return transformed_output


class Generator3(nn.Module):
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

        # generative module
        # TODO: input resolution: ???
        # TODO: figure out if upsampling or downsampling is needed (e.g.) timm output is too large or too small
        # *LAZY* conv2d layer which automatically calculates number of in_channels
        # from merged and outputs the specified channel
        self.gen_conv1 = nn.LazyConv2d(out_channels=128, kernel_size=1, padding=0)
        # 2x2
        self.gen_sdm1a = CustomStackedDecodeModule(in_channels=128, out_channels=3)
        self.gen_sdm1b = CustomStackedDecodeModule(in_channels=3, out_channels=64)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 4x4
        self.gen_sdm2a = CustomStackedDecodeModule(in_channels=64, out_channels=3)
        self.gen_sdm2b = CustomStackedDecodeModule(in_channels=3, out_channels=32)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 8x8
        self.gen_sdm3a = CustomStackedDecodeModule(in_channels=32, out_channels=3)
        self.gen_sdm3b = CustomStackedDecodeModule(in_channels=3, out_channels=16)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 16x16
        self.gen_sdm4a = CustomStackedDecodeModule(in_channels=16, out_channels=3)
        self.gen_sdm4b = CustomStackedDecodeModule(in_channels=3, out_channels=16)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        # 32x32
        self.gen_sdm5 = CustomStackedDecodeModule(in_channels=16, out_channels=8)
        # output
        self.gen_output_conv = nn.Conv2d(
            in_channels=8,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )

        self.generative = nn.ModuleList(
            [
                self.gen_conv1,
                self.gen_sdm1a,
                self.gen_sdm1b,
                self.up1,
                self.gen_sdm2a,
                self.gen_sdm2b,
                self.up2,
                self.gen_sdm3a,
                self.gen_sdm3b,
                self.up3,
                self.gen_sdm4a,
                self.gen_sdm4b,
                self.up4,
                self.gen_sdm5,
                self.gen_output_conv,
            ]
        )

    def get_generative_parameters(self):
        """Returns parameters of the generative module"""
        return self.generative.parameters()

    def forward(self, img, noise):
        # extract features from image
        features = self.feature_extractor(img)[0]
        # TODO: test the need of subnetwork for noise processing

        # merge noise with features
        # => noise and features need to have *same* dimensions if concatenated
        # => merging at dim=1 means concat at channel dim => (b, c, h, w)
        # TODO: check out adaptive instance normalization
        merged = torch.cat((features, noise), dim=1)

        # generative part
        x = self.gen_conv1(merged)

        # 2x2
        x = self.gen_sdm1a(x) + _resize(img, size=(2, 2))
        x = self.gen_sdm1b(x)
        x = self.up1(x)

        # 4x4
        x = self.gen_sdm2a(x) + _resize(img, size=(4, 4))
        x = self.gen_sdm2b(x)
        x = self.up2(x)

        # 8x8
        x = self.gen_sdm3a(x) + _resize(img, size=(8, 8))
        x = self.gen_sdm3b(x)
        x = self.up3(x)

        # 16x16
        x = self.gen_sdm4a(x) + _resize(img, size=(16, 16))
        x = self.gen_sdm4b(x)
        x = self.up4(x)

        # 32x32
        x = self.gen_sdm5(x)

        output_img = self.gen_output_conv(x)

        # sigmoid_output_img = torch.sigmoid(output_img)
        transformed_output = torch.tanh(output_img)

        return transformed_output
