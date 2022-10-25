from argparse import Namespace
from typing import Dict, List, Union

import torch
import wandb
from PIL import Image

from eval_metrics.disentanglement import DisentanglementMetricHigginsResult
from eval_metrics.latent_partitioning import ExpertClassifierStatistics
from utils import utils
from . import ExperimentLogger


class WAndBLogger(ExperimentLogger):
    def __init__(
        self,
        logdir: str = None,
        project="mvae",
        flags: Namespace = None,
        name: str = None,
        group: str = None,
        mode: str = None,
        job_type: str = "train",
    ):
        super(WAndBLogger, self).__init__()

        self._step = 0
        self._commit = True

        self._config_exclude_keys = [
            "mm_mvae_save",
            "load_daved",
            "use_classifier",
            "calc_nll",
            "calc_prd",
            "save_figure",
        ]

        if flags is not None:
            flags = wandb.helper.parse_config(flags, exclude=self._config_exclude_keys)

        wandb.init(
            project=project,
            save_code=False,
            name=name,
            group=group,
            mode=mode,
            config=flags,
            dir=logdir,
            job_type=job_type,
        )

    def write_flags(self, flags: Namespace):
        if flags is not None:
            flags_wandb = wandb.helper.parse_config(
                flags, exclude=self._config_exclude_keys
            )
        wandb.config.update(flags_wandb)
        utils.save_and_log_flags(flags)

    def write_expert_classifier_statistics(
        self, stats: Dict[str, ExpertClassifierStatistics]
    ):
        log = {
            f"Expert Classifier/{stat}": {
                "accuracy": stats[stat].accuracy,
                "precision": stats[stat].precision,
                "recall": stats[stat].recall,
                "f1": stats[stat].f1,
            }
            for stat in stats
        }
        wandb.log(log, step=self._step, commit=False)

    def write_lr_eval(self, lr_eval: Dict):
        log = {f"Latent Representation/{key}": lr_eval[key] for key in sorted(lr_eval)}
        wandb.log(log, step=self._step, commit=False)

    def write_coherence_logs(self, gen_eval: Dict, step: int = None):
        if step is None:
            step = self._step
        log = dict()
        for latent_key in sorted(gen_eval["cond"]):
            for sampled_key in sorted(gen_eval["cond"][latent_key]):
                log_key = f"Generation/{latent_key}"
                if log_key not in log:
                    log[log_key] = dict()

                log[log_key][sampled_key] = gen_eval["cond"][latent_key][sampled_key]

        log["Generation/loo"] = gen_eval["loo"]
        log["Generation/Random"] = gen_eval["random"]
        wandb.log(log, step=step, commit=False)

    def write_lhood_logs(self, lhoods: Dict):
        log = dict()
        for key in lhoods:
            log[f"Likelihoods/{key}"] = lhoods[key]
        wandb.log(log, step=self._step, commit=False)

    def write_prd_scores(self, prd_scores: Dict):
        log = dict()
        for key in prd_scores:
            log[f"PRD/{key}"] = prd_scores[key]
        wandb.log(log, step=self._step, commit=False)

    def write_plots(self, plots: Dict, epoch: int):
        log = dict()
        for plot_key in plots:
            sub_plots = plots[plot_key]
            for plot_name in sub_plots:
                plot = Image.fromarray(sub_plots[plot_name])
                caption = f"{plot_key}_{plot_name}"
                log[caption] = wandb.Image(plot, caption=caption)

        log["epoch"] = epoch

        wandb.log(log, step=self._step, commit=False)

    def write_avg_loss(
        self,
        prefix: str,
        loss_weighted: Union[torch.Tensor, float],
        loss: Union[torch.Tensor, float],
        loss_tcmvae_weighted: Union[torch.Tensor, float],
        loss_tcmvae_unweighted: Union[torch.Tensor, float],
    ):
        log = {
            f"{prefix}/Loss Average Batches Unweighted": {
                "loss_unweighted": loss.data.item(),
                "loss_tcmvae_unweighted": loss.data.item(),
            },
            f"{prefix}/Loss Average Batches Weighted": {
                "loss": loss_weighted.data.item(),
                "loss_tcmvae": loss_weighted.data.item(),
            },
        }
        wandb.log(log, step=self._step)

    def write_kld_decomposition(self, name: str, kld_decomposition: Dict):
        log = {f"{name}/kld_decomposition": kld_decomposition}
        wandb.log(log, step=self._step)

    def _collect_logs(
        self,
        results: Dict,
        loss: Union[torch.Tensor, float],
        loss_unweighted: Union[torch.Tensor, float],
        loss_tcmvae: Union[torch.Tensor, float],
        loss_tcmvae_unweighted: Union[torch.Tensor, float],
        log_probs: torch.Tensor,
        klds: torch.Tensor,
        kld_decompositions: Dict,
    ) -> Dict:
        modalities = results["latents"]["modalities"]
        means = dict()
        log_vars = dict()
        for modality in modalities:
            if modalities[modality][0] is not None:
                means[modality] = modalities[modality][0].mean().item()
            if modalities[modality][1] is not None:
                log_vars[modality] = modalities[modality][1].mean().item()

        log = {
            "loss": loss.data.item(),
            "loss_unweighted": loss_unweighted.data.item(),
            "LogProb": log_probs,
            "klds": klds,
            "group_divergence": results["joint_divergence"],
            "means": means,
            "log_vars": log_vars,
        }

        if not(loss_tcmvae is None or loss_tcmvae_unweighted is None or kld_decompositions is None):
            log.update({
                "loss_tcmvae": loss_tcmvae.data.item(),
                "loss_tcmvae_unweighted": loss_tcmvae_unweighted.data.item(),
                "kld_decompositions": kld_decompositions,
            })

        return log

    def write_training_logs(
        self,
        results: Dict,
        loss: Union[torch.Tensor, float],
        loss_unweighted: Union[torch.Tensor, float],
        loss_tcmvae: Union[torch.Tensor, float],
        loss_tcmvae_unweighted: Union[torch.Tensor, float],
        log_probs: torch.Tensor,
        klds: torch.Tensor,
        kld_decompositions: Dict,
    ):

        log = self._collect_logs(
            results,
            loss,
            loss_unweighted,
            loss_tcmvae,
            loss_tcmvae_unweighted,
            log_probs,
            klds,
            kld_decompositions,
        )
        train_log = dict()
        for key in log:
            train_log[f"train/{key}"] = log[key]

        wandb.log(train_log, step=self._step)
        self._step += 1

    def write_testing_logs(
        self,
        results: Dict,
        loss: Union[torch.Tensor, float],
        loss_unweighted: Union[torch.Tensor, float],
        loss_tcmvae: Union[torch.Tensor, float],
        loss_tcmvae_unweighted: Union[torch.Tensor, float],
        log_probs: torch.Tensor,
        klds: torch.Tensor,
        kld_decompositions: Dict,
    ):
        log = self._collect_logs(
            results,
            loss,
            loss_unweighted,
            loss_tcmvae,
            loss_tcmvae_unweighted,
            log_probs,
            klds,
            kld_decompositions,
        )
        test_log = dict()
        for key in log:
            test_log[f"test/{key}"] = log[key]

        wandb.log(test_log, step=self._step)
        self._step += 1

    def write_disentanglement_metric(
        self,
        result: DisentanglementMetricHigginsResult,
        attribute_names: List[str],
        subset_name: str,
        step: int,
        prefix: str = "test",
    ):
        prefix = f"disentanglement/higgins/{prefix}/{subset_name}/"
        disent_log = {
            f"{prefix}joint": {
                "disentanglement": result.disentanglement,
                "confusion_matrix_table": wandb.Table(
                    data=result.confusion_matrix,
                    columns=attribute_names,
                    rows=attribute_names,
                ),
                "confusion_matrix_image": wandb.Image(
                    result.confusion_matrix_display.figure_
                ),
                "confusion_matrix_plot": result.confusion_matrix_display.figure_,
            },
        }

        for attribute_name, attribute_disentanglement in zip(
            attribute_names, result.per_attribute_disentanglement
        ):
            disent_log[f"{prefix}{attribute_name}"] = {
                "disentanglement": attribute_disentanglement
            }

        wandb.log(disent_log, step=step)

    def write_total_correlations(
        self, result: Dict[str, torch.Tensor], step: int, prefix: str = "train"
    ):
        if step is None:
            step = self._step

        prefix = f"total_correlation/{prefix}/"
        disent_log = {f"total_correlation/{prefix}": result}
        wandb.log(disent_log, step=step)

    def close(self):
        wandb.finish()
