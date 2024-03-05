import torch
import torch.nn.functional as F
from src.models import DCGenerator, DCDiscriminator
import torch.nn as nn


def main():
    # Define parameters
    z_dim = 100  # Dimension of the noise vector
    num_samples = 2  # Number of samples to generate
    num_classes = 10  # Number of classes in CIFAR-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Generator and Discriminator
    G = DCGenerator(z_dim).to(device)
    # config D - weight norm
    D_weight_norm = nn.utils.spectral_norm

    # config D - norm
    D_norm = nn.Identity

    # config D - activation
    D_activation = nn.LeakyReLU

    # initialize D
    D = DCDiscriminator(
        norm=D_norm, weight_norm=D_weight_norm, activation=D_activation
    ).to(device)

    G.eval()  # Set the generator to evaluation mode
    D.eval()  # Set the discriminator to evaluation mode

    # Create random noise as input for G
    dummy_noise = torch.randn(num_samples, z_dim).to(device)

    # Generate random labels and convert them to one-hot encoding
    labels_fake = torch.randint(low=0, high=num_classes, size=(num_samples,)).to(device)
    labels_fake_onehot = (
        F.one_hot(labels_fake, num_classes=num_classes).float().to(device)
    )

    # Generate fake images
    gen_output = G(dummy_noise, labels_fake_onehot)[0]
    print("Generator output shape:", gen_output.shape)

    # Test Discriminator on fake images
    disc_output_fake = D(gen_output, labels_fake_onehot)
    print("Discriminator output shape:", disc_output_fake.shape)


if __name__ == "__main__":
    main()
