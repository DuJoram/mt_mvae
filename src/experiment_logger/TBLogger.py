from argparse import Namespace
from typing import Dict, List

import torch.utils.tensorboard.writer
from tensorboardX import SummaryWriter

from eval_metrics.disentanglement import DisentanglementMetricHigginsResult
from utils import utils
from . import ExperimentLogger


class TBLogger(ExperimentLogger):
    def __init__(self, name, dir_logs):
        super(TBLogger, self).__init__()

        self.name = name
        self.writer = SummaryWriter(dir_logs)
        self.training_prefix = "train"
        self.testing_prefix = "test"
        self.step = 0

    def write_flags(self, flags: Namespace):
        str_flags = utils.save_and_log_flags(flags)
        self.writer.add_text("FLAGS", str_flags, 0)

    def write_log_probs(self, name, log_probs):
        self.writer.add_scalars("%s/LogProb" % name, log_probs, self.step)

    def write_klds(self, name, klds):
        self.writer.add_scalars("%s/KLD" % name, klds, self.step)

    def write_group_div(self, name, group_div):
        self.writer.add_scalars(
            "%s/group_divergence" % name, {"group_div": group_div.item()}, self.step
        )

    def write_latent_distr(self, name, latents):
        l_mods = latents["modalities"]
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.writer.add_scalars(
                    "%s/mu" % name, {key: l_mods[key][0].mean().item()}, self.step
                )
            if not l_mods[key][1] is None:
                self.writer.add_scalars(
                    "%s/logvar" % name, {key: l_mods[key][1].mean().item()}, self.step
                )

    def write_expert_classifier_statistics(self, stats):
        for expert in sorted(stats):
            expert_stats = stats[expert]
            self.writer.add_scalars(
                "Expert Classifier/%/accuracy" % (expert),
                expert_stats.accuracy,
                self.step,
            )
            self.writer.add_scalars(
                "Expert Classifier/%/precision" % (expert),
                expert_stats.precision,
                self.step,
            )
            self.writer.add_scalars(
                "Expert Classifier/%/recall" % (expert), expert_stats.recall, self.step
            )
            self.writer.add_scalars(
                "Expert Classifier/%/f1" % (expert), expert_stats.f1, self.step
            )

    def write_lr_eval(self, lr_eval):
        for s, l_key in enumerate(sorted(lr_eval.keys())):
            self.writer.add_scalars(
                "Latent Representation/%s" % (l_key), lr_eval[l_key], self.step
            )

    def write_coherence_logs(self, gen_eval, step: int = None):
        if step is None:
            step = self.step

        for j, l_key in enumerate(sorted(gen_eval["cond"].keys())):
            for k, s_key in enumerate(gen_eval["cond"][l_key].keys()):
                self.writer.add_scalars(
                    "Generation/%s/%s" % (l_key, s_key),
                    gen_eval["cond"][l_key][s_key],
                    step,
                )

        self.writer.add_scalars("Generation/loo", gen_eval["loo"], step)
        self.writer.add_scalars("Generation/Random", gen_eval["random"], step)

    def write_lhood_logs(self, lhoods):
        for k, key in enumerate(sorted(lhoods.keys())):
            self.writer.add_scalars("Likelihoods/%s" % (key), lhoods[key], self.step)

    def write_prd_scores(self, prd_scores):
        self.writer.add_scalars("PRD", prd_scores, self.step)

    def write_plots(self, plots, epoch):
        for k, p_key in enumerate(plots.keys()):
            ps = plots[p_key]
            for l, name in enumerate(ps.keys()):
                fig = ps[name]
                self.writer.add_image(p_key + "_" + name, fig, epoch, dataformats="HWC")

    def write_avg_loss(self, name, loss, loss_unweighted, loss_tcmvae, loss_tcmvae_unweighted):
        if loss_tcmvae is None or loss_tcmvae_unweighted is None:
            self.writer.add_scalars(
                "%s/Loss_avg_batches_w" % name, {"loss": loss.data.item()}, self.step
            )
            self.writer.add_scalars(
                "%s/Loss_avg_batches_unw" % name,
                {"loss_unweighted": loss_unweighted.data.item()},
                self.step,
                )
        else:
            self.writer.add_scalars(
                "%s/Loss_avg_batches_w" % name, {"loss": loss.data.item(), "loss_tcmvae": loss_tcmvae.item()}, self.step
            )
            self.writer.add_scalars(
                "%s/Loss_avg_batches_unw" % name,
                {"loss_unweighted": loss_unweighted.data.item(), "loss_tcmvae_unweighted": loss_tcmvae_unweighted.item()},
                self.step,
            )

    def write_kld_decomposition(self, name, kld_decompositions):
        for subset, components in kld_decompositions["class"].items():
            self.writer.add_scalars(f"{name}/kld_decomposition/class/{subset}", components, self.step)

        for subset, modalities in kld_decompositions["style"].items():
            for modality, components in modalities.values():
                self.writer.add_scalars(f"{name}/kld_decomposition/style/{subset}/{modality}", components, self.step)


    def add_basic_logs(self, name, results, loss, loss_unweighted, loss_tcmvae, loss_tcmvae_unweighted, log_probs, klds, kld_decompositions):
        self.writer.add_scalars("%s/Loss" % name, {"loss": loss.data.item()}, self.step)
        self.writer.add_scalars(
            "%s/Loss_unweighted" % name,
            {"value": loss_unweighted.data.item()},
            self.step,
        )
        if not(loss_tcmvae is None or loss_tcmvae_unweighted is None):
            self.writer.add_scalars(f"{name}/Loss_tcmvae", {"loss": loss_tcmvae.item()}, self.step)
            self.writer.add_scalars(f"{name}/Loss_tcmvae_unweighted", {"loss_unweighted": loss_tcmvae_unweighted.item()}, self.step)
        self.write_log_probs(name, log_probs)
        self.write_klds(name, klds)
        self.write_group_div(name, results["joint_divergence"])
        self.write_latent_distr(name, results["latents"])
        if kld_decompositions is not None:
            self.write_kld_decomposition(name, kld_decompositions)

    def write_training_logs(self, results, loss, loss_unweighted, loss_tcmvae, loss_tcmvae_unweighted, log_probs, klds, kld_decompositions):
        self.add_basic_logs(
            self.training_prefix, results, loss, loss_unweighted, loss_tcmvae, loss_tcmvae_unweighted, log_probs, klds, kld_decompositions
        )
        self.step += 1

    def write_testing_logs(self, results, loss, loss_unweighted, loss_tcmvae, loss_tcmvae_unweighted, log_probs, klds, kld_decompositions):
        self.add_basic_logs(
            self.testing_prefix, results, loss, loss_unweighted, loss_tcmvae, loss_tcmvae_unweighted, log_probs, klds, kld_decompositions
        )
        self.step += 1

    def write_disentanglement_metric(
        self,
        result: DisentanglementMetricHigginsResult,
        attribute_names: List[str],
        subset_name: str,
        step: int,
        prefix: str = "test",
    ):
        tag_prefix = f"disentanglement/higgins/{prefix}/{subset_name}/"
        self.writer.add_scalar(
            tag_prefix + "joint/disentanglement",
            result.disentanglement,
            global_step=step,
        )
        self.writer.add_figure(
            tag_prefix + "joint/confusion_matrix_plot",
            result.confusion_matrix_display.figure_,
            global_step=step,
        )

        for attribute_name, attribute_disentanglement in zip(
            attribute_names, result.per_attribute_disentanglement
        ):
            self.writer.add_scalar(
                tag_prefix + "joint/disentanglement_" + attribute_name,
                attribute_disentanglement,
                global_step=step,
            )

    def write_total_correlations(
        self, result: Dict[str, torch.Tensor], step: int, prefix: str = "train"
    ):
        if step is None:
            step = self.step

        for subset_name, tc in result.items():
            self.writer.add_scalar(
                f"total_correlation/{prefix}/{subset_name}",
                tc,
                global_step=step,
            )

    def close(self):
        pass
