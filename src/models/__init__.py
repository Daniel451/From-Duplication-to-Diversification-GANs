from ._discriminator import Discriminator, DiscriminatorCustom
from ._generator import Generator
from ._generator2 import Generator as Generator2
from .model_dcgan import Generator as DCGenerator
from .model_dcgan import Discriminator as DCDiscriminator
from .model_dcgan import weights_init_normal
from .vanilla_gan import (
    Generator as VanillaGenerator,
    Discriminator as VanillaDiscriminator,
)
