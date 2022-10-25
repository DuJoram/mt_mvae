from typing import Tuple, Optional

import torch
from torch import nn

from models import Encoder, Decoder


class MdSpritesConvImageEncoder(Encoder):
    def __init__(
        self,
        class_dim: int,
        style_dim: int,
        input_channels: int,
        num_modalities: int,
        force_partition_latent_space_mode: str = None,
        factorized_representation: bool = False,
    ):
        super(MdSpritesConvImageEncoder, self).__init__()

        self._class_dim = class_dim
        self._style_dim = style_dim
        self._input_channels = input_channels
        self._num_modalities = num_modalities
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

        shared_features = self._style_dim + self._class_dim

        # Input shape: (B, 1|3, 64, 64)
        self._shared_encoder = nn.Sequential(
            nn.Conv2d(  # -> (B, 16, 32, 32)
                in_channels=input_channels,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(  # -> (B, 32, 16, 16)
                in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(  # -> (B, 64, 8, 8)
                in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(  # -> (B, 128, 4, 4)
                in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),  # -> (B, 128*4*4) = (B, 2048)
            nn.Linear(in_features=2048, out_features=shared_features),
            nn.ReLU(),
        )

        self._class_mean = nn.Linear(
            in_features=shared_features, out_features=self._class_dim
        )

        self._class_log_var = nn.Linear(
            in_features=shared_features, out_features=self._class_dim
        )

        if self._factorized_representation:
            self._style_mean = nn.Linear(
                in_features=shared_features, out_features=self._style_dim
            )
            self._style_log_var = nn.Linear(
                in_features=shared_features, out_features=self._style_dim
            )

    def forward(self, input_batch) -> Tuple[Optional[torch.Tensor], ...]:
        shared_features = self._shared_encoder(input_batch)
        class_mean = self._class_mean(shared_features)
        class_log_var = self._class_log_var(shared_features)
        style_mean = None
        style_log_var = None
        if self._factorized_representation:
            style_mean = self._style_mean(shared_features)
            style_log_var = self._style_log_var(shared_features)

        return style_mean, style_log_var, class_mean, class_log_var


class MdSpritesConvImageDecoder(Decoder):
    def __init__(
        self,
        class_dim: int,
        style_dim: int,
        output_channels: int = 1,
        factorized_representation: bool = False,
        predict_variance: bool = False,
        predict_logits: bool = False,
    ):
        super(MdSpritesConvImageDecoder, self).__init__()

        self._class_dim = class_dim
        self._style_dim = style_dim
        self._factorized_representation = factorized_representation
        self._predict_variance = predict_variance

        self._decoder = nn.Sequential(
            nn.Linear(  # (B, ???) -> (B, 2048)
                in_features=self._style_dim + self._class_dim, out_features=2048
            ),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(128, 4, 4)),
            nn.ConvTranspose2d(
                in_channels=128,  # -> (B, 64, 8, 8)
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,  # -> (B, 32, 16, 16)
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,  # -> (B, 16, 32, 32)
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,  # -> (B, 1|3, 64, 64)
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.Identity() if predict_logits else nn.Sigmoid()
        )

    def forward(
        self, style_latents, class_latents
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if self._factorized_representation:
            latents = torch.cat((style_latents, class_latents), dim=1)
        else:
            latents = class_latents

        decoded_mean = self._decoder(latents)

        if self._predict_variance:
            return decoded_mean, torch.tensor(0.75, device=latents.device)

        return (decoded_mean,)
