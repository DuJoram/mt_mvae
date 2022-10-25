import torch
from torch import nn


class ExpertClassifier(nn.Module):
    def __init__(self, latent_size: int, n_experts: int, n_layers: int = 10):
        super(ExpertClassifier, self).__init__()

        self._latent_size = latent_size
        self._n_experts = n_experts

        modules = list()
        in_features = 2 * latent_size
        for idx in range(n_layers - 1):
            out_features = max(in_features // 2, n_experts)
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.Dropout(0.3))
            modules.append(nn.ReLU())
            in_features = out_features

        modules.append(nn.Linear(in_features, n_experts))

        self._model = nn.Sequential(*modules)

    def forward(self, mean: torch.Tensor, logvar: torch.Tensor):
        X = torch.concat([mean, logvar], dim=-1)
        out = self._model(X)
        if not self.training:
            out = torch.sigmoid(out)
        return out
