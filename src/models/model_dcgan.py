import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from loguru import logger


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        logger.debug(f"init conv weights '{classname}'")
        nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_uniform_(m.weight, mode="fan_out")
        # nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        logger.debug(f"init linear weights '{classname}'")
        nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_uniform_(m.weight, mode="fan_out")
        # nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        logger.debug(f"init batchnorm 2d weights '{classname}'")
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        logger.debug(f"init batchnorm 1d weights '{classname}'")
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


class Block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels / 2),
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.b1 = nn.BatchNorm2d(num_features=int(in_channels / 2))

        self.c2 = nn.Conv2d(
            in_channels=int(in_channels / 2),
            out_channels=int(in_channels / 2),
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.b2 = nn.BatchNorm2d(num_features=int(in_channels / 2))

        self.c3 = nn.Conv2d(
            in_channels=int(in_channels / 2),
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.b3 = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        features_in = x

        x = self.c1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)

        x = self.c2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)

        x = self.c3(x)
        x = self.b3(x)
        x = F.leaky_relu(x)

        x = features_in + x

        return x


class Generator(nn.Module):
    def __init__(self, g_input_dim):
        super(Generator, self).__init__()
        # Adjust the output size of the first linear layer
        self.fc1 = nn.Linear(in_features=g_input_dim, out_features=8 * 8 * 128)
        self.fc2 = nn.Linear(in_features=10, out_features=8 * 8 * 128)
        self.fc3 = nn.Linear(in_features=8 * 8 * 256, out_features=8 * 8 * 128)

        self.block1 = Block(in_channels=128)
        self.block2 = Block(in_channels=128)
        self.block3 = Block(in_channels=128)

        # Adjust the output channels to 3 for RGB
        self.final_conv = nn.Conv2d(
            in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, label):
        # process random noise
        fc1 = F.leaky_relu(self.fc1(x), 0.2)

        # process class information
        fc2 = F.leaky_relu(self.fc2(label), 0.2)

        # concat data
        fc_concat = torch.cat((fc1, fc2), dim=1)
        fc3 = F.leaky_relu(self.fc3(fc_concat))

        # transform to 2D space
        x = fc3.view(-1, 128, 8, 8)

        # apply convolutions and upsample
        x = self.block1(x)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # Upsample to 16x16

        x = self.block2(x)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # Upsample to 32x32

        x = self.block3(x)

        # final layer to adjust to RGB
        x = self.final_conv(x)

        # last activation
        x_logits = x
        x = torch.tanh(x_logits)

        return x, x_logits


class Discriminator(nn.Module):
    def __init__(self, norm, weight_norm, activation):
        super(Discriminator, self).__init__()

        # First convolutional layer
        self.step1 = nn.Sequential(
            weight_norm(
                nn.Conv2d(
                    in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
                )
            ),
            norm(num_features=64),
            activation(),
        )

        # Second convolutional layer
        self.step2 = nn.Sequential(
            weight_norm(
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
                )
            ),
            norm(num_features=128),
            activation(),
        )

        # Third convolutional layer
        self.step3 = nn.Sequential(
            weight_norm(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ),
            norm(num_features=256),
            activation(),
        )

        # Fourth convolutional layer
        self.step4 = nn.Sequential(
            weight_norm(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ),
            norm(num_features=512),
            activation(),
        )

        # Linear layer for label processing
        self.layer_label = nn.Sequential(
            weight_norm(nn.Linear(in_features=10, out_features=512)),
            norm(num_features=512),
            activation(),
        )

        # Calculate the size of the flattened feature maps
        self.fc_num_features = 512 * 2 * 2  # Assuming the final feature map size is 2x2

        # Fully connected layers
        self.fc1 = nn.Sequential(
            weight_norm(nn.Linear(in_features=self.fc_num_features, out_features=512)),
            norm(num_features=512),
            activation(),
        )

        self.fc2 = nn.Linear(
            in_features=1024, out_features=1
        )  # 512 for image features + 512 for label features

    def forward(self, x, label):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)

        # Flatten the output for the fully connected layer
        x = x.view(-1, self.fc_num_features)

        # Process image information
        x = self.fc1(x)

        # Process label information
        label = self.layer_label(label)

        # Concatenate image and label information
        x = torch.cat((x, label), dim=1)

        # Final layer
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
