import torch
import os
from torchvision.utils import save_image
from src.models import Generator2 as Generator
from src.data import (
    get_single_cifar10_dataloader as get_cifar10_dataloader,
)  # Assuming this is the correct import based on your data loader


def load_generator(checkpoint_path, device):
    generator = Generator(device=device)  # Initialize the generator
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    return generator


def generate_images(generator, dataloader, device, num_images):
    save_dir = "CIFAR10_class4_customgan_generated"
    os.makedirs(save_dir, exist_ok=True)

    generated_count = 0
    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        noise = torch.rand(batch_size, 56, 2, 2, device=device)

        with torch.no_grad():
            generated_images = generator(images, noise)

        for image in generated_images:
            if generated_count >= num_images:
                break
            save_path = os.path.join(save_dir, f"image_{generated_count+1}.png")
            save_image(image, save_path)
            generated_count += 1

        if generated_count >= num_images:
            break

    print(f"Generated {generated_count} images in '{save_dir}'")


if __name__ == "__main__":
    device = torch.device("cpu")  # Use 'cuda' if GPU is available
    checkpoint_path = (
        "saved_models/grtwq2vc/generator_200.pt"  # Update with actual path
    )
    generator = load_generator(checkpoint_path, device)

    # Load the CIFAR-10 dataloader for class 4
    dataloader = get_cifar10_dataloader(target_class=4, batch_size=10, num_workers=8)[
        0
    ]  # Adjust batch_size and num_workers as needed

    generate_images(
        generator, dataloader, device, num_images=5000
    )  # Generate 5000 images
