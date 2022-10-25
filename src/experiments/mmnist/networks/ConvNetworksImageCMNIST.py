import torch
import torch.nn as nn

from models import Encoder, Decoder
from utils.utils import Flatten, Unflatten


class EncoderImg(Encoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(
        self,
        class_dim: int,
        style_dim: int,
        num_modalities: int,
        force_partition_latent_space_mode: str = None,
        factorized_representation: bool = False,
    ):
        super(EncoderImg, self).__init__()

        self._class_dim = class_dim
        self._num_modalities = num_modalities
        self._style_dim = style_dim
        self._factorized_representation = factorized_representation

        force_partition_latent_space_mode_choices = ["concatenate", "variance", None]
        if (
            force_partition_latent_space_mode
            not in force_partition_latent_space_mode_choices
        ):
            raise ValueError(
                f"Unknown latent space partitioning mode '{force_partition_latent_space_mode}'"
            )

        if force_partition_latent_space_mode == "concatenate":
            self._class_dim = self._class_dim // self._num_modalities

        self._shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),  # -> (2048)
            nn.Linear(
                2048, self._style_dim + self._class_dim
            ),  # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self._class_mu = nn.Linear(self._style_dim + self._class_dim, self._class_dim)
        self._class_logvar = nn.Linear(
            self._style_dim + self._class_dim, self._class_dim
        )
        # optional style branch
        if self._factorized_representation:
            self._style_mu = nn.Linear(
                self._style_dim + self._class_dim, self._style_dim
            )
            self._style_logvar = nn.Linear(
                self._style_dim + self._class_dim, self._style_dim
            )

    def forward(self, x):
        h = self._shared_encoder(x)
        if self._factorized_representation:
            return (
                self._style_mu(h),
                self._style_logvar(h),
                self._class_mu(h),
                self._class_logvar(h),
            )
        else:
            return None, None, self._class_mu(h), self._class_logvar(h)


class DecoderImg(Decoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(
        self, class_dim: int, style_dim: int, factorized_representation: bool = False
    ):
        super(DecoderImg, self).__init__()

        self._class_dim = class_dim
        self._style_dim = style_dim
        self._factorized_representation = factorized_representation

        self._decoder = nn.Sequential(
            nn.Linear(self._style_dim + self._class_dim, 2048),  # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),  # -> (128, 4, 4)
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (3, 28, 28)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self._factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self._decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat, torch.tensor(0.75).to(
            z.device
        )  # NOTE: consider learning scale param, too
