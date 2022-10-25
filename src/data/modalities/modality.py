from abc import ABC, abstractmethod
from typing import Union

import torch

import utils.utils
from utils import utils


class Modality(ABC):
    def __init__(
        self,
        name: str,
        likelihood: Union[str, torch.distributions.Distribution],
    ):
        self._name = name
        if isinstance(likelihood, torch.distributions.Distribution):
            self._likelihood = likelihood
        else:
            self._likelihood = utils.get_likelihood(likelihood)

    @abstractmethod
    def save_data(self, d, fn, args):
        pass

    @abstractmethod
    def plot_data(self, d):
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def likelihood(self) -> torch.distributions.Distribution:
        return self._likelihood

    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target).sum()
        mean_val_logprob = log_prob / norm_value
        # mean_val_logprob = out_dist.log_prob(target).mean()
        return mean_val_logprob
