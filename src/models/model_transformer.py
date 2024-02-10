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

        self.transform1 = nn.Conv2d(3, 96, kernel_size=1, stride=1, padding="same")
        self.mod1a = MultiConvModule(96, 48)
        self.mod1b = MultiConvModule(48, 24)
        self.mod1c = MultiConvModule(24, 3)

        self.transform2 = nn.Conv2d(3, 96, kernel_size=1, stride=1, padding="same")
        self.mod2a = MultiConvModule(96, 48)
        self.mod2b = MultiConvModule(48, 24)
        self.mod2c = MultiConvModule(24, 3)

        self.transform3 = nn.Conv2d(3, 96, kernel_size=1, stride=1, padding="same")
        self.mod3a = MultiConvModule(96, 48)
        self.mod3b = MultiConvModule(48, 24)
        self.mod3c = MultiConvModule(24, 3)
        
        self.final = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding="same")

    def forward(self, x):
        x = self.conv_transformer_autoencoder(x)
        transformer_output = x

        x = self.transform1(x)
        x = self.mod1a(x)
        x = self.mod1b(x)
        x = self.mod1c(x)
        mod1_out = transformer_output + x

        x = self.transform2(mod1_out)
        x = self.mod2a(x)
        x = self.mod2b(x)
        x = self.mod2c(x)
        mod2_out = mod1_out + x

        x = self.transform3(mod2_out)
        x = self.mod3a(x)
        x = self.mod3b(x)
        x = self.mod3c(x)
        mod3_out = mod2_out + x
        
        x = self.final(mod3_out)
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


class MultiConvModule(nn.Module):
    
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__()

        # 1x1 input processing
        self.conv_input = nn.Conv2d(input_channels, input_channels//6, kernel_size=1, stride=1, padding="same")
        self.bn_input = nn.BatchNorm2d(input_channels//6)

        # 7x7, 5x5, 3x3, 1x1, maxpool, avgpool
        self.conv7x7 = nn.Conv2d(input_channels//6, input_channels//6, kernel_size=7, stride=1, padding="same")
        self.bn7x7 = nn.BatchNorm2d(input_channels//6)

        self.conv5x5 = nn.Conv2d(input_channels//6, input_channels//6, kernel_size=5, stride=1, padding="same")
        self.bn5x5 = nn.BatchNorm2d(input_channels//6)
        
        self.conv3x3 = nn.Conv2d(input_channels//6, input_channels//6, kernel_size=3, stride=1, padding="same")
        self.bn3x3 = nn.BatchNorm2d(input_channels//6)
        
        self.conv1x1 = nn.Conv2d(input_channels//6, input_channels//6, kernel_size=1, stride=1, padding="same")
        self.bn1x1 = nn.BatchNorm2d(input_channels//6)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bn_maxpool = nn.BatchNorm2d(input_channels//6)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.bn_avgpool = nn.BatchNorm2d(input_channels//6)

        # 1x1 output processing
        self.conv_output = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding="same")
        self.bn_output = nn.BatchNorm2d(output_channels)


    def forward(self, x):
        # input processing
        x_input = self.conv_input(x)
        x_input = self.bn_input(x_input)
        x_input = F.softsign(x_input)

        # modules
        part1 = self.conv7x7(x_input)
        part1 = self.bn7x7(part1)
        part1 = F.softsign(part1)

        part2 = self.conv5x5(x_input)
        part2 = self.bn5x5(part2)
        part2 = F.softsign(part2)
        
        part3 = self.conv3x3(x_input)
        part3 = self.bn3x3(part3)
        part3 = F.softsign(part3)
        
        part4 = self.conv1x1(x_input)
        part4 = self.bn1x1(part4)
        part4 = F.softsign(part4)
        
        part5 = self.maxpool(x_input)
        part5 = self.bn_maxpool(part5)
        part5 = F.softsign(part5)
        
        part6 = self.avgpool(x_input)
        part6 = self.bn_avgpool(part6)
        part6 = F.softsign(part6)

        # output processing
        x_output = torch.cat([part1, part2, part3, part4, part5, part6], dim=1)
        x_output = self.conv_output(x_output + x)
        x_output = self.bn_output(x_output)
        x_output = F.softsign(x_output)

        return x_output
        

