import torch
import torchvision
import pytorch_lightning as pl
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from src.models import Generator, Discriminator
from src.data import get_cifar10_dataloader


class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.noise = torch.rand(size=(2, 112, 14, 14)).to(self.device)
        self.generator = Generator().to(self.device)
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

        # Move the valid and fake tensors to the same device as the model
        valid = torch.ones(batch_size, 1).to(self.device)
        fake = torch.zeros(batch_size, 1).to(self.device)

        # Discriminator update
        self.opt_d.zero_grad()
        real_loss = self.criterion(self.discriminator(images), valid)
        fake_loss = self.criterion(
            self.discriminator(self.generator(images, self.noise)), fake
        )
        loss_d = (real_loss + fake_loss) / 2
        self.manual_backward(loss_d)
        self.opt_d.step()

        # Generator update
        self.opt_g.zero_grad()
        gen_imgs = self.generator(images, self.noise)
        loss_g = self.criterion(self.discriminator(gen_imgs), valid)
        self.manual_backward(loss_g)
        self.opt_g.step()

        # Log generated images
        if batch_idx % 100 == 0:
            with torch.no_grad():
                img_grid = torchvision.utils.make_grid(gen_imgs, normalize=True)
                self.logger.experiment.log(
                    {
                        "generated_images": [
                            wandb.Image(img_grid, caption="Generated Images")
                        ]
                    }
                )

        return {
            "loss": loss_g,
            "log": {"loss_generator": loss_g},
        }

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        # Get both optimizers
        self.opt_g = optimizer_g
        self.opt_d = optimizer_d
        return optimizer_d, optimizer_g

    def train_dataloader(self):
        logger.info("Loading training data...")
        return get_cifar10_dataloader(batch_size=64, num_workers=2)[0]


# Weights & Biases setup for online-only logging
wandb.init(
    project="GAN-CIFAR10", name="GAN-run", settings=wandb.Settings(mode="online")
)

wandb_logger = WandbLogger()
gpus = 1 if torch.cuda.is_available() else 0
# start training
logger.info("Starting training...")
torch.set_float32_matmul_precision("medium")  # or 'high' based on your precision needs
trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=1, logger=wandb_logger)
gan = GAN()
trainer.fit(gan)
wandb.finish()
logger.info("Finished training!")
