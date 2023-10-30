import torch
import torchvision
import pytorch_lightning as pl
import wandb
import loguru as logger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from src.models import Generator, Discriminator
from src.data import get_cifar10_dataloader

seed_everything(42)  # For reproducibility


class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.criterion = torch.nn.BCELoss()
        self.sample_val_images = None

        self.fixed_noise = torch.rand(size=(2, 112, 14, 14))

    def on_epoch_start(self):
        if self.sample_val_images is None:
            self.sample_val_images = next(iter(self.train_dataloader()))[0]

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch
        batch_size = images.size(0)
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Discriminator update

        if optimizer_idx == 0:
            real_loss = self.criterion(self.discriminator(images), valid)
            fake_loss = self.criterion(
                self.discriminator(self.generator(self.fixed_noise)), fake
            )
            loss_d = (real_loss + fake_loss) / 2
            return {"loss": loss_d, "log": {"loss_discriminator": loss_d}}

        # Generator update
        if optimizer_idx == 1:
            gen_imgs = self.generator(self.fixed_noise)
            loss_g = self.criterion(self.discriminator(gen_imgs), valid)

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

            return {"loss": loss_g, "log": {"loss_generator": loss_g}}

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        return [optimizer_d, optimizer_g], []

    def train_dataloader(self):
        logger.info("Loading training data...")
        return get_cifar10_dataloader(batch_size=64, num_workers=2)[0]


# Weights & Biases setup for online-only logging
wandb.init(
    project="GAN-CIFAR10", name="GAN-run", settings=wandb.Settings(mode="online")
)

logger = WandbLogger()
gpus = 1 if torch.cuda.is_available() else 0
# start training
logger.info("Starting training...")
trainer = pl.Trainer(max_epochs=10, gpus=gpus, logger=logger)
gan = GAN()
trainer.fit(gan)
logger.info("Finished training!")