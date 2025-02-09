import torch
import os
from torchvision.utils import save_image
from src.models import VanillaGenerator as Generator


def load_generator(checkpoint_path, device):
    generator = Generator()
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator


def generate_images(generator, num_images, device, batch_size=10, latent_dim=100):
    save_dir = "CIFAR10_class4_vanilla_gan_generated"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(0, num_images, batch_size):
        # Generate noise vectors as input for the generator
        noise = torch.randn(batch_size, latent_dim, device=device)

        with torch.no_grad():
            generated_images = generator(noise)

        for j, image in enumerate(generated_images):
            save_path = os.path.join(save_dir, f"image_{i+j+1}.png")
            save_image(image, save_path, normalize=True)

    print(f"Generated {num_images} images in '{save_dir}'")


if __name__ == "__main__":
    device = torch.device("cpu")  # Use 'cuda' if GPU is available
    checkpoint_path = (
        "model_checkpoints/iwt11afk/generator_175.pt"  # Update with the actual path
    )
    generator = load_generator(checkpoint_path, device)
    generate_images(generator, 5000, device)  # Generate 5000 images
