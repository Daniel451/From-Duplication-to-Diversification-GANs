import torch
import torchvision
import pytorch_lightning as pl
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from src.models import Generator, Discriminator
from src.data import get_cifar10_dataloader


# TODO: try different weight init methods
def init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()

        # create generator
        self.generator = Generator(self.device).to(self.device)
        # generator dummy call => init lazy layers
        dummy_noise = torch.rand(size=(2, 56, 2, 2)).to(self.device)
        dummy_images = torch.rand(size=(2, 3, 32, 32)).to(self.device)
        self.generator(dummy_images, dummy_noise)
        # initialize weights
        for layer in self.generator.generative.modules():
            layer.apply(init_weights)

        # create discriminator
        self.discriminator = Discriminator().to(self.device)

        # exponential moving average losses for G and D
        self.g_ema = 0
        self.d_ema = 0
        self.d_ema_g_ema_diff = 0

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
        # valid = torch.ones(batch_size, 1).to(self.device)
        # fake = torch.zeros(batch_size, 1).to(self.device)

        # soft labels
        # TODO: try out more or less randomness
        valid = torch.rand((batch_size, 1), device=self.device) * 0.1 + 0.9
        fake = torch.rand((batch_size, 1), device=self.device) * 0.1

        # Discriminator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        real_loss = self.criterion(self.discriminator(images), valid)
        fake_loss = self.criterion(
            self.discriminator(self.generator(images, noise)), fake
        )
        loss_d = (real_loss + fake_loss) / 2
        if self.d_ema_g_ema_diff > -0.15:
            self.manual_backward(loss_d)
            self.opt_d.step()

        # Update exponential moving average loss for D
        self.d_ema = self.d_ema * 0.9 + loss_d.detach().item() * 0.1

        # Generator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()
        gen_imgs = self.generator(images, noise)

        # TODO: try out no soft-labels for generator (only for discriminator)
        loss_g_div = self.criterion(self.discriminator(gen_imgs), valid)
        gen_images_id = self.generator(images, torch.zeros_like(noise))
        loss_g_id = torch.mean((gen_images_id - images) ** 2)
        loss_g = loss_g_div + loss_g_id * 2
        if self.d_ema_g_ema_diff < 0.15:
            self.manual_backward(loss_g)
            self.opt_g.step()

        # Update exponential moving average loss for G
        self.g_ema = self.g_ema * 0.9 + loss_g_div.detach().item() * 0.1

        self.d_ema_g_ema_diff = self.d_ema - (self.g_ema / 2)

        if batch_idx % 50 == 0:
            with torch.no_grad():
                # log losses
                self.logger.experiment.log(
                    {
                        "losses/d_fake": fake_loss,
                        "losses/d_real": real_loss,
                        "losses/d_ema-g_ema": self.d_ema_g_ema_diff,
                        "losses/d_ema": self.d_ema,
                        "losses/g_ema": self.g_ema,
                        "losses/d": loss_d,
                        "losses/g_div": loss_g_div,
                        "losses/g_id": loss_g_id,
                        "losses/g": loss_g,
                    }
                )

        # Log generated images
        if batch_idx % 250 == 0:
            with torch.no_grad():
                # Log generated images
                img_grid = torchvision.utils.make_grid(gen_imgs, normalize=True)
                img_grid_id = torchvision.utils.make_grid(gen_images_id, normalize=True)
                self.logger.experiment.log(
                    {
                        "images/generated": [
                            wandb.Image(img_grid, caption="Generated Images")
                        ],
                        "images/generated_id": [
                            wandb.Image(
                                img_grid_id, caption="Generated Identity Images"
                            )
                        ],
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
            self.generator.get_generative_parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        # TODO: try out different learning rates for discriminator
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        # Get both optimizers
        self.opt_g = optimizer_g
        self.opt_d = optimizer_d
        return optimizer_d, optimizer_g

    def train_dataloader(self):
        logger.info("Loading training data...")
        return get_cifar10_dataloader(target_class=4, batch_size=128, num_workers=8)[0]


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
trainer = pl.Trainer(max_epochs=500, accelerator="gpu", devices=1, logger=wandb_logger)
gan = GAN()
trainer.fit(gan)
wandb.finish()
logger.info("Finished training!")
