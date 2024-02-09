import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class ConvEncoder(nn.Module):
    def __init__(self, embed_size, image_size=64, image_channels=3):
        super(ConvEncoder, self).__init__()
        self.embed_size = embed_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                image_channels, 64, kernel_size=4, stride=2, padding=1
            ),  # Output: [64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # Output: [128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(
                128, 256, kernel_size=4, stride=2, padding=1
            ),  # Output: [256, 8, 8]
            nn.ReLU(),
        )

        # Calculate the size of the features after convolutional layers
        conv_output_size = 256 * (image_size // 8) * (image_size // 8)
        self.fc = nn.Linear(conv_output_size, embed_size)

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, embed_size, image_size=64, image_channels=3):
        super(ConvDecoder, self).__init__()
        self.embed_size = embed_size
        self.image_size = image_size

        # Transformer Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model=embed_size, nhead=8)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=1)

        self.fc = nn.Linear(embed_size, 256 * (image_size // 8) * (image_size // 8))

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, memory):
        x = self.transformer_decoder(x, memory)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(
            -1, 256, self.image_size // 8, self.image_size // 8
        )  # Reshape to match the conv layers input
        x = self.conv_layers(x)
        return x


class ConvTransformerAutoencoder(nn.Module):
    def __init__(self, embed_size, image_size=64, image_channels=3):
        super(ConvTransformerAutoencoder, self).__init__()
        self.encoder = ConvEncoder(embed_size, image_size, image_channels)
        self.decoder = ConvDecoder(embed_size, image_size, image_channels)

    def forward(self, x):
        memory = self.encoder(x)
        x = self.decoder(memory, memory)
        return x


class TransformerGenerator(nn.Module):
    def __init__(self, embed_size=256, image_size=32):
        super().__init__()
        self.conv_transformer_autoencoder = ConvTransformerAutoencoder(
            embed_size, image_size
        )

    def forward(self, x):
        x = self.conv_transformer_autoencoder(x)
        return x
