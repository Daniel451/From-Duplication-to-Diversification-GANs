import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class ConvEncoder(nn.Module):
    def __init__(self, embed_size, image_size=64, image_channels=3):
        super(ConvEncoder, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(image_size * image_size * image_channels, embed_size)

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, embed_size, image_size=64, image_channels=3):
        super(ConvDecoder, self).__init__()
        self.embed_size = embed_size
        self.image_size = image_size
        self.image_channels = image_channels

        # Transformer Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model=embed_size, nhead=8)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=1)

        self.fc = nn.Linear(embed_size, image_size * image_size * image_channels)

    def forward(self, x, memory):
        x = self.transformer_decoder(x, memory)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(
            -1, self.image_channels, self.image_size, self.image_size
        )  # Reshape to match the conv layers input
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
        x = torch.tanh(x)
        return x


class TransformerDiscriminator(nn.Module):
    def __init__(self, embed_size=256, image_size=32):
        super().__init__()
        self.conv_transformer_autoencoder = ConvTransformerAutoencoder(
            embed_size, image_size
        )
        self.conv_encoder = ConvEncoder(embed_size, image_size)
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, x):
        x = self.conv_transformer_autoencoder(x)
        x = self.conv_encoder(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 1)
        return x