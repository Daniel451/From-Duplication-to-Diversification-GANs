import torch
import torchvision
import pytorch_lightning as pl
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from src.models import Generator, Discriminator
from src.data import get_single_cifar10_dataloader as get_cifar10_dataloader
from pytorch_msssim import SSIM
import torch.nn.functional as F
from functools import partial

sweep_config = {
    "method": "bayes",
    "metric": {"name": "loss_g", "goal": "minimize"},
    "parameters": {
        "lr_gen": {"values": [0.00005, 0.0001, 0.0002, 0.0005]},
        "lr_disc": {"values": [0.00005, 0.0001, 0.0002, 0.0005]},
        "weight_init": {"values": ["normal", "xavier", "kaiming"]},
        "loss_type": {"values": ["BCE", "LSGAN", "Hinge"]},
        "optimizer_type": {"values": ["adam", "sgd", "rmsprop"]},
        "batch_size": {"values": [32, 64, 128]},
    },
}


def init_weights(m, weight_init):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        if weight_init == "normal":
            torch.nn.init.normal_(m.weight)
        elif weight_init == "xavier":
            torch.nn.init.xavier_uniform_(m.weight)
        elif weight_init == "kaiming":
            torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.0)  # Gamma initialized to 1
        torch.nn.init.constant_(m.bias, 0.0)  # Beta initialized to 0


class GAN(pl.LightningModule):
    def __init__(
        self,
        lr_gen,
        lr_disc,
        optimizer_type,
        weight_init,
        loss_type,
        batch_size,
    ):
        super(GAN, self).__init__()
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.optimizer_type = optimizer_type
        self.weight_init = weight_init
        self.loss_type = loss_type
        self.batch_size = batch_size

        # create generator
        self.generator = Generator(self.device).to(self.device)
        # generator dummy call => init lazy layers
        dummy_noise = torch.rand(size=(2, 56, 2, 2)).to(self.device)
        dummy_images = torch.rand(size=(2, 3, 32, 32)).to(self.device)
        self.generator(dummy_images, dummy_noise)
        # initialize weights
        for layer in self.generator.generative.modules():
            layer.apply(partial(init_weights, weight_init))

        # create discriminator
        self.discriminator = Discriminator().to(self.device)

        # exponential moving average losses for G and D
        self.g_ema = 0
        self.d_ema = 0
        self.d_ema_g_ema_diff = 0

        # self.criterion = torch.nn.BCELoss()
        self.criterion = self.create_criterion()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.sample_val_images = None

        self.automatic_optimization = False
        self.best_loss = float("inf")
        self.best_model_state = None

    def create_criterion(self):
        if self.loss_type == "BCE":
            return torch.nn.BCELoss()
        elif self.loss_type == "LSGAN":
            return torch.nn.MSELoss()
        elif self.loss_type == "Hinge":
            return torch.nn.HingeEmbeddingLoss()
        # TODO implement WASG Loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def configure_optimizers(self):
        # Define default optimizers in case none of the conditions match
        optimizer_g = torch.optim.Adam(
            self.generator.get_generative_parameters(),
            lr=self.lr_gen,
            betas=(0.5, 0.999),
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_disc, betas=(0.5, 0.999)
        )

        # Check for specific optimizer types and update accordingly
        if self.optimizer_type == "sgd":
            optimizer_g = torch.optim.SGD(
                self.generator.get_generative_parameters(), lr=self.lr_gen, momentum=0.9
            )
            optimizer_d = torch.optim.SGD(
                self.discriminator.parameters(), lr=self.lr_disc, momentum=0.9
            )
        elif self.optimizer_type == "rmsprop":
            optimizer_g = torch.optim.RMSprop(
                self.generator.get_generative_parameters(), lr=self.lr_gen
            )
            optimizer_d = torch.optim.RMSprop(
                self.discriminator.parameters(), lr=self.lr_disc
            )

        self.opt_g = optimizer_g
        self.opt_d = optimizer_d
        return optimizer_d, optimizer_g

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
        # Generate fake images
        fake_images = self.generator(images, noise)
        fake_images_uint8 = (fake_images * 255).to(torch.uint8)
        images_uint8 = (images * 255).to(torch.uint8)

        # Soft labels
        valid = torch.rand((batch_size, 1), device=self.device) * 0.1 + 0.9
        fake = torch.rand((batch_size, 1), device=self.device) * 0.1

        # Discriminator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()

        # Discriminator's prediction
        real_pred = self.discriminator(images)
        fake_pred = self.discriminator(fake_images.detach())

        # Loss for real and fake images
        if self.loss_type in ["BCE", "LSGAN"]:
            real_loss = self.criterion(real_pred, valid)
            fake_loss = self.criterion(fake_pred, fake)
        elif self.loss_type == "Hinge":
            real_loss = torch.mean(torch.relu(1.0 - real_pred))
            fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
        else:
            raise ValueError(f"Unsupported loss type: {self.hparams.loss_type}")
        loss_d = (real_loss + fake_loss) / 2
        if self.d_ema_g_ema_diff > -0.15:
            self.manual_backward(loss_d)
            self.opt_d.step()

        # Update exponential moving average loss for D
        self.d_ema = self.d_ema * 0.9 + loss_d.detach().item() * 0.1

        # Generator update
        self.opt_g.zero_grad()
        self.opt_d.zero_grad()

        gen_pred = self.discriminator(fake_images)

        # TODO: try out no soft-labels for generator (only for discriminator)

        # loss_g_div = self.criterion(self.discriminator(gen_imgs), valid)
        if self.loss_type == "BCE":
            loss_g_div = self.criterion(gen_pred, valid)
        elif self.loss_type == "LSGAN":
            loss_g_div = self.criterion(gen_pred, valid)
        elif self.loss_type == "Hinge":
            loss_g_div = -torch.mean(gen_pred)
        gen_images_id = self.generator(images, torch.zeros_like(noise))
        loss_g_id_ssim = 1 - self.ssim(gen_images_id, images)
        loss_g_id_mse = torch.mean((gen_images_id - images) ** 2) * 2
        loss_g_id = loss_g_id_ssim + loss_g_id_mse
        loss_g = loss_g_div + loss_g_id
        if self.d_ema_g_ema_diff < 0.15:
            self.manual_backward(loss_g)
            self.opt_g.step()

        # Update exponential moving average loss for G
        self.g_ema = self.g_ema * 0.9 + loss_g_div.detach().item() * 0.1

        self.d_ema_g_ema_diff = self.d_ema - (self.g_ema / 2)

        if loss_g < self.best_loss:
            self.best_loss = loss_g
            self.best_model_state = self.generator.state_dict()

        if batch_idx % 50 == 0:
            with torch.no_grad():
                self.logger.experiment.log(
                    {
                        "losses/d_fake": fake_loss,
                        "losses/d_real": real_loss,
                        "losses/d_ema-g_ema": self.d_ema_g_ema_diff,
                        "losses/d_ema": self.d_ema,
                        "losses/g_ema": self.g_ema,
                        "losses/d": loss_d,
                        "losses/g_div": loss_g_div,
                        "losses/g_id_ssim": loss_g_id_ssim,
                        "losses/g_id_mse": loss_g_id_mse,
                        "losses/g_id": loss_g_id,
                        "losses/g": loss_g,
                    }
                )

        # Log generated images
        if batch_idx % 250 == 0:
            with torch.no_grad():
                # Log generated images
                img_grid = torchvision.utils.make_grid(fake_images, normalize=True)
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

    def train_dataloader(self):
        logger.info("Loading training data...")
        return get_cifar10_dataloader(self.batch_size, num_workers=8)[0]


def train(config=None):
    # Generate a unique session name
    current_time = datetime.now()
    session_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    with wandb.init(
        project="GAN-CIFAR10",
        name="Sweep-GAN-train-FILM-edgenet" + session_name,
        settings=wandb.Settings(mode="online"),
        config=config,
    ):
        config = wandb.config

        model = GAN(
            lr_gen=config.lr_gen,
            lr_disc=config.lr_disc,
            optimizer_type=config.optimizer_type,
            weight_init=config.weight_init,
            loss_type=config.loss_type,
            batch_size=config.batch_size,
        )
        wandb_logger = WandbLogger()
        gpus = 1 if torch.cuda.is_available() else 0
        logger.info("Starting training...")
        torch.set_float32_matmul_precision("medium")

        trainer = pl.Trainer(
            max_epochs=300, accelerator="gpu", devices=gpus, logger=wandb_logger
        )
        trainer.fit(model)


sweep_id = wandb.sweep(sweep_config, project="GAN-CIFAR10")

wandb.agent(sweep_id, train, count=10)  # Adjust count as needed


logger.info("Finished training!")
