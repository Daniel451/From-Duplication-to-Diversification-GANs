import torch
import torchvision
import pytorch_lightning as pl
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from src.models import Generator, Discriminator
from src.data import get_cifar10_dataloader


class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator(self.device).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.criterion = torch.nn.BCELoss()
        self.sample_val_images = None

        self.automatic_optimization = False

    def on_epoch_start(self):
        if self.sample_val_images is None:
            self.sample_val_images = next(iter(self.train_dataloader()))[0].to(
                self.device
            )

    def training_step(self, batch, batch_idx):
        images, _ = batch
        images = images.to(self.device)
        batch_size = images.size(0)
        noise = torch.rand(size=(batch_size, 56, 2, 2)).to(self.device)

        # Move the valid and fake tensors to the same device as the model
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Discriminator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        real_loss = self.criterion(self.discriminator(images), valid)
        fake_loss = self.criterion(
            self.discriminator(self.generator(images, noise)), fake
        )
        loss_d = (real_loss + fake_loss) / 2
        self.manual_backward(loss_d)
        self.opt_d.step()

        # Generator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        gen_imgs = self.generator(images, noise)

        loss_g = self.criterion(self.discriminator(gen_imgs), valid)
        self.manual_backward(loss_g)
        self.opt_g.step()

        if batch_idx % 50 == 0:
            with torch.no_grad():
                # log losses
                self.logger.experiment.log(
                    {
                        "losses/d_fake": fake_loss,
                        "losses/d_real": real_loss,
                        "losses/d": loss_d,
                        "losses/g": loss_g,
                    }
                )

        # Log generated images
        if batch_idx % 250 == 0:
            with torch.no_grad():
                # Log generated images
                img_grid = torchvision.utils.make_grid(gen_imgs, normalize=True)
                self.logger.experiment.log(
                    {
                        "images/generated": [
                            wandb.Image(img_grid, caption="Generated Images")
                        ]
                    }
                )
                # Log real images
                img_grid_real = torchvision.utils.make_grid(images, normalize=True)
                self.logger.experiment.log(
                    {
                        "images/real": [
                            wandb.Image(img_grid_real, caption="Generated Images")
                        ]
                    }
                )

        return {
            "loss": loss_g,
            "log": {"loss_generator": loss_g},
        }

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator.get_generative_parameters(), lr=0.001, betas=(0.5, 0.9)
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.9)
        )
        # Get both optimizers
        self.opt_g = optimizer_g
        self.opt_d = optimizer_d
        return optimizer_d, optimizer_g

    def train_dataloader(self):
        logger.info("Loading training data...")
        return get_cifar10_dataloader(batch_size=128, num_workers=8)[0]


current_time = datetime.now()
session_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Weights & Biases setup for online-only logging
wandb.init(
    project="GAN-CIFAR10",
    name="Basic-GAN-train-" + session_name,
    settings=wandb.Settings(mode="online"),
)

wandb_logger = WandbLogger()
gpus = 1 if torch.cuda.is_available() else 0
# start training
logger.info("Starting training...")
torch.set_float32_matmul_precision("medium")  # or 'high' based on your precision needs
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1, logger=wandb_logger)
gan = GAN()
trainer.fit(gan)
wandb.finish()
logger.info("Finished training!")
