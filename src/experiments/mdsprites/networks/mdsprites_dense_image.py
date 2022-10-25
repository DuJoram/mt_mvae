from typing import Tuple, Optional

import torch
from torch import nn

from models import Encoder, Decoder


class MdSpritesDenseImageEncoder(Encoder):
    def __init__(
        self,
        class_dim: int,
        style_dim: int,
        num_modalities: int,
        input_channels: int = 1,
        num_hidden_layers: int = 2,
        force_partition_latent_space_mode: str = None,
        factorized_representation: bool = False,
    ):
        super(MdSpritesDenseImageEncoder, self).__init__()

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
        hidden_layers = list()
        for idx in range(max(0, num_hidden_layers - 1)):
            hidden_layers.extend(
                [
                    nn.Linear(
                        in_features=1200,
                        out_features=1200,
                    ),
                    nn.ReLU(),
                ]
            )

        self._shared_encoder = nn.Sequential(
            nn.Flatten(),  # --> (B, (3|1) * 64 * 64,
            nn.Linear(
                in_features=input_channels * 64 * 64,
                out_features=1200,
            ),
            nn.ReLU(),
            *hidden_layers,
        )

        self._class_latent_params = nn.Sequential(
            nn.Linear(in_features=1200, out_features=2 * self._class_dim)
        )

        if self._factorized_representation:
            self._style_latent_params = nn.Sequential(
                nn.Linear(in_features=1200, out_features=2 * self._style_dim)
            )

    def forward(self, input_batch) -> Tuple[Optional[torch.Tensor], ...]:
        shared_features = self._shared_encoder(input_batch)
        class_params = self._class_latent_params(shared_features)
        class_mean = class_params[:, : self._class_dim]
        class_log_var = class_params[:, self._class_dim :]

        style_mean = None
        style_log_var = None
        if self._factorized_representation:
            style_params = self._style_latent_params(shared_features)
            style_mean = style_params[:, self._style_dim]
            style_log_var = style_params[:, self._style_dim :]

        return style_mean, style_log_var, class_mean, class_log_var


class MdSpritesDenseImageDecoder(Decoder):
    def __init__(
        self,
        class_dim: int,
        style_dim: int,
        output_channels: int = 1,
        num_hidden_layers: int = 3,
        factorized_representation: bool = False,
        predict_variance: bool = False,
        predict_logits: bool = False
    ):
        super(MdSpritesDenseImageDecoder, self).__init__()

        self._class_dim = class_dim
        self._style_dim = style_dim
        self._factorized_representation = factorized_representation
        self._predict_variance = predict_variance

        hidden_layers = list()
        for idx in range(max(0, num_hidden_layers - 1)):
            hidden_layers.extend(
                [
                    nn.Linear(
                        in_features=1200,
                        out_features=1200,
                    ),
                    nn.Tanh(),
                ]
            )
        self._decoder = nn.Sequential(
            nn.Linear(in_features=self._style_dim + self._class_dim, out_features=1200),
            nn.Tanh(),
            *hidden_layers,
            nn.Linear(in_features=1200, out_features=output_channels * 64 * 64),
            nn.Identity() if predict_logits else nn.Sigmoid(),
            nn.Unflatten(dim=-1, unflattened_size=(output_channels, 64, 64)),
        )


    def forward(
        self,
        style_latents: Optional[torch.Tensor],
        class_latents: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if self._factorized_representation:
            latents = torch.cat((style_latents, class_latents), dim=1)
        else:
            latents = class_latents

        decoded_mean = self._decoder(latents)

        if self._predict_variance:
            return decoded_mean, torch.tensor(0.75, device=latents.device)

        return (decoded_mean,)
