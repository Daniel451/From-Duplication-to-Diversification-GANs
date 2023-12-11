import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.data import get_cifar10_dataloader as get_dataloader
from models import DCGenerator, DCDiscriminator
import timm
from torchvision.utils import make_grid
import uuid


class GAN(pl.LightningModule):
    def __init__(self, z_dim=100, lr=1e-4, start_c_after=15):
        super().__init__()
        self.save_hyperparameters()
        self.z_dim = z_dim
        self.lr = lr
        self.start_c_after = start_c_after

        # Model Definitions
        self.C = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=10, in_chans=1
        )
        self.G = DCGenerator(g_input_dim=z_dim)
        self.D = DCDiscriminator()

        # Loss Functions
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()

    def forward(self, z, labels_onehot):
        return self.G(z, labels_onehot)

    def training_step(self, batch, batch_idx, optimizer_idx):
        img, labels_real = batch
        img = img.to(self.device)
        labels_real = labels_real.to(self.device)
        labels_real_onehot = (
            F.one_hot(labels_real, num_classes=10).float().to(self.device)
        )

        # Create noise vector and labels for generator
        z = torch.randn(img.size(0), self.z_dim).to(self.device)
        labels_fake = torch.randint(low=0, high=9, size=(img.size(0),)).to(self.device)
        labels_fake_onehot = (
            F.one_hot(labels_fake, num_classes=10).float().to(self.device)
        )

        if optimizer_idx == 0:  # Generator
            generated_imgs = self(z, labels_fake_onehot)
            D_out = self.D(generated_imgs, labels_fake_onehot)
            G_loss = self.criterion(D_out, torch.ones(img.size(0), 1).to(self.device))

            # Log generated images
            if batch_idx % 50 == 0:
                grid = make_grid((generated_imgs + 1) / 2)
                self.logger.experiment.log(
                    {
                        "generated_images": [
                            wandb.Image(grid, caption="Generated Images")
                        ]
                    }
                )

            return G_loss

        if optimizer_idx == 1:  # Discriminator
            # Real images
            D_real = self.D(img, labels_real_onehot)
            D_real_loss = self.criterion(
                D_real, torch.ones(img.size(0), 1).to(self.device)
            )

            # Fake images
            with torch.no_grad():
                fake_imgs = self(z, labels_fake_onehot).detach()
            D_fake = self.D(fake_imgs, labels_fake_onehot)
            D_fake_loss = self.criterion(
                D_fake, torch.zeros(img.size(0), 1).to(self.device)
            )

            D_loss = (D_real_loss + D_fake_loss) / 2
            return D_loss

    def validation_step(self, batch, batch_idx):
        # Implement validation logic if needed
        pass

    def configure_optimizers(self):
        lr = self.lr
        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=100, num_workers=16, dataset_size=48000):
        super().__init__()
        # Load training data using the get_dataloader function
        train_loader, _, _, _ = get_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            dataset_size=self.dataset_size,
        )
        return train_loader

    def val_dataloader(self):
        # Implement if you have validation data
        pass


def main(
    root_path: str = typer.Option("."),
    epochs: int = typer.Option(100),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    z_dim: int = typer.Option(100),
    start_c_after: int = typer.Option(15),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    dataset_size: int = typer.Option(48000),
):
    # Initialize WandB
    wandb_logger = WandbLogger(name="GAN_Experiment", project="GAN_Project")

    # Initialize our model
    model = GAN(z_dim=z_dim, lr=lr, start_c_after=start_c_after)

    # Data Module
    data_module = CustomDataModule(
        batch_size=batch_size, num_workers=num_workers, dataset_size=dataset_size
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    typer.run(main)
