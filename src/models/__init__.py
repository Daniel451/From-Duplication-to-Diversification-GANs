from ._discriminator import Discriminator
from ._generator import Generator, Generator2, Generator3
from .model_dcgan import Generator as DCGenerator
from .model_dcgan import Discriminator as DCDiscriminator
from .model_dcgan import weights_init_normal
from .vanilla_gan import (
    Generator as VanillaGenerator,
    Discriminator as VanillaDiscriminator,
)
