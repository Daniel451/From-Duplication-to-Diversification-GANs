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


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # Calculate critic's scores on interpolated images
    mixed_scores = critic(interpolates)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolates,
        outputs=mixed_scores,
        grad_outputs=torch.ones(mixed_scores.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate the norm of the gradients
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


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

        self.critic_iterations = 5
        self.lambda_gp = 5
        self.clip_gradients = True

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

        """
            -----------------------------
            Critic (Discriminator) update
            -----------------------------
        """
        for _ in range(self.critic_iterations):
            self.opt_d.zero_grad()

            imgs_fake = self.generator(images, noise).detach()

            # critic loss
            loss_critic_real = self.discriminator(images).mean()
            loss_critic_fake = self.discriminator(imgs_fake).mean()
            loss_critic = loss_critic_fake - loss_critic_real
            
            # gradient penalty
            gp = compute_gradient_penalty(self.discriminator, images, imgs_fake, self.device)
            loss_critic = loss_critic + self.lambda_gp * gp

            # backward pass
            self.manual_backward(loss_critic)

            # optionally, add gradient clipping here
            if self.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

            # update critic
            self.opt_d.step()
            
        
        """
            ----------------
            Generator update
            ----------------
        """
        self.opt_g.zero_grad()
        gen_imgs = self.generator(images, noise)

        loss_g = -self.discriminator(gen_imgs).mean()

        self.manual_backward(loss_g)
        self.opt_g.step()

        if batch_idx % 50 == 0:
            with torch.no_grad():
                # log losses
                self.logger.experiment.log(
                    {
                        "losses/d_fake": loss_critic_fake.detach().cpu(),
                        "losses/d_real": loss_critic_real.detach().cpu(),
                        "losses/d": loss_critic.detach().cpu(),
                        "losses/g": loss_g.detach().cpu(),
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
            self.generator.get_generative_parameters(), lr=0.0001, betas=(0.5, 0.999)
        )
        # TODO: try out different learning rates for discriminator
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999)
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
# run = wandb.init(
#     project="GAN-CIFAR10",
#     name="Basic-GAN-train-" + session_name,
#     settings=wandb.Settings(mode="online"),
# )
wandb_logger = WandbLogger(
    project="GAN-CIFAR10",
    name="Basic-GAN-train-" + session_name,
    settings=wandb.Settings(mode="online"),
    tags=["wgan-gp", "cifar10"],
    group="wgan-gp",
)

gpus = 1 if torch.cuda.is_available() else 0
# start training
logger.info("Starting training...")
torch.set_float32_matmul_precision("medium")  # or 'high' based on your precision needs
trainer = pl.Trainer(max_epochs=200, accelerator="gpu", devices=1, logger=wandb_logger)
gan = GAN()
trainer.fit(gan)
wandb.finish()
logger.info("Finished training!")
