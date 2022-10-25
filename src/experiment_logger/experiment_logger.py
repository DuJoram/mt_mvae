import abc
from argparse import Namespace
from typing import Dict, List, Union

import torch

from eval_metrics.disentanglement import DisentanglementMetricHigginsResult
from eval_metrics.latent_partitioning import ExpertClassifierStatistics


class ExperimentLogger(abc.ABC):
    def __init__(self):
        super(ExperimentLogger, self).__init__()

    @abc.abstractmethod
    def write_flags(self, flags: Namespace):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_expert_classifier_statistics(
        self, stats: Dict[str, ExpertClassifierStatistics]
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_lr_eval(self, lr_eval: Dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_coherence_logs(self, gen_eval: Dict, step: int = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_lhood_logs(self, lhoods: Dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_prd_scores(self, prd_scores: torch.Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_plots(self, plots: Dict, epoch: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_avg_loss(
        self,
        prefix: str,
        loss_weighted: torch.Tensor,
        loss: torch.Tensor,
        loss_tcmvae: torch.Tensor,
        loss_tcmvae_unweighted: torch.Tensor,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_kld_decomposition(self, name: str, kld_decomposition: Dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_training_logs(
        self,
        results: Dict,
        loss: Union[torch.Tensor, float],
        loss_unweighted: Union[torch.Tensor, float],
        loss_tcmvae: Union[torch.Tensor, float],
        loss_tcmvae_unweighted: Union[torch.Tensor, float],
        log_probs: torch.Tensor,
        klds: torch.Tensor,
        kld_decompostions: Dict,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_testing_logs(
        self,
        results: Dict,
        loss: Union[torch.Tensor, float],
        loss_unweighted: Union[torch.Tensor, float],
        loss_tcmvae: Union[torch.Tensor, float],
        loss_tcmvae_unweighted: Union[torch.Tensor, float],
        log_probs: torch.Tensor,
        klds: torch.Tensor,
        kld_decompostions: Dict,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_disentanglement_metric(
        self,
        result: DisentanglementMetricHigginsResult,
        attribute_names: List[str],
        subset_name: str,
        step: int,
        prefix: str = "test",
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def write_total_correlations(
        self, result: Dict[str, torch.Tensor], step: int, prefix: str = "train"
    ):
        raise NotImplementedError()
