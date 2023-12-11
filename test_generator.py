import torch
from src.models import Generator
from src.models import Discriminator


def main():
    # Test Generator
    G = Generator("cpu")
    dummy_img = torch.rand(size=(2, 3, 32, 32))

    dummy_noise = torch.rand(size=(2, 56, 8, 8))
    gen_output = G(dummy_img, dummy_noise)

    print("Generator output shape:", gen_output.shape)


if __name__ == "__main__":
    main()
