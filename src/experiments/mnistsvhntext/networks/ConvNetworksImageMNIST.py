import typing

import torch
import torch.nn as nn

dataSize = torch.Size([1, 28, 28])

from models import Encoder, Decoder


class EncoderImg(Encoder):
    def __init__(
        self,
        class_dim: int,
        style_mnist_dim: int,
        num_hidden_layers: int,
        num_mods: int,
        force_partition_latent_space_mode: typing.Optional[str] = None,
        factorized_representation: bool = False,
    ):
        super(EncoderImg, self).__init__()
        self._hidden_dim = 400

        force_partition_latent_space_mode_choices = ["concatenate", "variance", None]
        if (
            force_partition_latent_space_mode
            not in force_partition_latent_space_mode_choices
        ):
            raise ValueError(
                f"Unsupported latent space partitioning mode '{force_partition_latent_space_mode}'"
            )

        self._class_dim = class_dim
        if force_partition_latent_space_mode == "concatenate":
            self._class_dim = self._class_dim // num_mods

        self._style_mnist_dim = style_mnist_dim
        self._num_hidden_layers = num_hidden_layers
        self._num_mods = num_mods
        self._factorized_representation = factorized_representation

        modules = []
        modules.append(nn.Sequential(nn.Linear(784, self._hidden_dim), nn.ReLU(True)))
        modules.extend(
            [
                nn.Sequential(
                    nn.Linear(self._hidden_dim, self._hidden_dim), nn.ReLU(True)
                )
                for _ in range(self._num_hidden_layers - 1)
            ]
        )
        self._enc = nn.Sequential(*modules)
        self._relu = nn.ReLU()
        if self._factorized_representation:
            # style
            self._style_mu = nn.Linear(
                in_features=self._hidden_dim,
                out_features=self._style_mnist_dim,
                bias=True,
            )
            self._style_logvar = nn.Linear(
                in_features=self._hidden_dim,
                out_features=self._style_mnist_dim,
                bias=True,
            )
            # class
            self._class_mu = nn.Linear(
                in_features=self._hidden_dim, out_features=self._class_dim, bias=True
            )
            self._class_logvar = nn.Linear(
                in_features=self._hidden_dim, out_features=self._class_dim, bias=True
            )
        else:
            # non-factorized
            self._hidden_mu = nn.Linear(
                in_features=self._hidden_dim, out_features=self._class_dim, bias=True
            )
            self._hidden_logvar = nn.Linear(
                in_features=self._hidden_dim, out_features=self._class_dim, bias=True
            )

    def forward(self, x):
        h = x.view(*x.size()[:-3], -1)
        h = self._enc(h)
        h = h.view(h.size(0), -1)
        if self._factorized_representation:
            style_latent_space_mu = self._style_mu(h)
            style_latent_space_logvar = self._style_logvar(h)
            class_latent_space_mu = self._class_mu(h)
            class_latent_space_logvar = self._class_logvar(h)
            style_latent_space_mu = style_latent_space_mu.view(
                style_latent_space_mu.size(0), -1
            )
            style_latent_space_logvar = style_latent_space_logvar.view(
                style_latent_space_logvar.size(0), -1
            )
            class_latent_space_mu = class_latent_space_mu.view(
                class_latent_space_mu.size(0), -1
            )
            class_latent_space_logvar = class_latent_space_logvar.view(
                class_latent_space_logvar.size(0), -1
            )
            return (
                style_latent_space_mu,
                style_latent_space_logvar,
                class_latent_space_mu,
                class_latent_space_logvar,
            )
        else:
            latent_space_mu = self._hidden_mu(h)
            latent_space_logvar = self._hidden_logvar(h)
            latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
            latent_space_logvar = latent_space_logvar.view(
                latent_space_logvar.size(0), -1
            )
            return None, None, latent_space_mu, latent_space_logvar


class DecoderImg(Decoder):
    def __init__(
        self,
        class_dim: int,
        style_mnist_dim: int,
        num_hidden_layers: int,
        factorized_representation: bool = False,
    ):
        super(DecoderImg, self).__init__()

        self._class_dim = class_dim
        self._style_mnist_dim = style_mnist_dim
        self._num_hidden_layers = num_hidden_layers
        self._factorized_representation = factorized_representation

        self._hidden_dim = 400
        modules = []
        if self._factorized_representation:
            modules.append(
                nn.Sequential(
                    nn.Linear(self._style_mnist_dim + self._class_dim, self.hidden_dim),
                    nn.ReLU(True),
                )
            )
        else:
            modules.append(
                nn.Sequential(
                    nn.Linear(self._class_dim, self._hidden_dim), nn.ReLU(True)
                )
            )

        modules.extend(
            [
                nn.Sequential(
                    nn.Linear(self._hidden_dim, self._hidden_dim), nn.ReLU(True)
                )
                for _ in range(self._num_hidden_layers - 1)
            ]
        )
        self._dec = nn.Sequential(*modules)
        self._fc3 = nn.Linear(self._hidden_dim, 784)
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()

    def forward(self, style_latent_space, class_latent_space):

        if self._factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self._dec(z)
        x_hat = self._fc3(x_hat)
        x_hat = self._sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *dataSize)
        return x_hat, torch.tensor(0.75, device=z.device)
