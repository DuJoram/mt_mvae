from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import poe
from divergence_measures.mm_div import poe_stable
from data.modalities.modality import Modality
from utils import utils
from .decoder import Decoder
from .encoder import Encoder


class MultimodalVAE(nn.Module):
    def __init__(
        self, flags, encoders: List[Encoder], decoders: List[Decoder], modalities: List[Modality], subsets: Dict[str, List[Modality]]
    ):
        super(MultimodalVAE, self).__init__()
        self.flags = flags
        self._num_modalities = len(modalities)
        self._modalities = modalities
        self._subsets = subsets
        self._subsets_names_modalities = list(subsets.items())
        self._set_fusion_functions()

        self._class_dim: int = flags.class_dim
        self._factorized_representation = flags.factorized_representation
        self._device = flags.device

        self._epoch: int = 0
        self._iteration: int = 0
        self._num_iterations: int = 0

        self._modality_to_index = {modality.name: modality_idx for modality_idx, modality in enumerate(self._modalities)}
        self._modality_concat_slice_size = self._class_dim // self._num_modalities

        self._encoders = torch.nn.ModuleList(encoders)
        self._decoders = torch.nn.ModuleList(decoders)

    @property
    def num_modalities(self) -> int:
        return self._num_modalities

    @property
    def modalities(self) -> List[Modality]:
        return self._modalities

    @property
    def subsets(self) -> Dict[str, List[Modality]]:
        return self._subsets

    def forward(self, input_batch):
        latents = self.inference(input_batch)
        results = dict()
        results["latents"] = latents
        results["group_distr"] = latents["joint"]
        class_latents = utils.reparameterize(*latents["joint"])
        results["class_latent_samples"] = class_latents
        if self._factorized_representation:
            results["style_latent_samples"] = dict()

        divergence = self.calc_joint_divergence(
            latents["mus"], latents["logvars"], latents["weights"]
        )

        for divergence_name, divergence in divergence.items():
            results[divergence_name] = divergence

        encoded_modalities = latents["modalities"]
        results_reconstructions = dict()

        for modality, decoder in zip(self._modalities, self._decoders):
            if modality.name in input_batch:
                modality_input_batch = input_batch[modality.name]
                if modality_input_batch is not None:
                    style_mean, style_log_var = encoded_modalities[
                        f"{modality.name}_style"
                    ]
                    if self._factorized_representation:
                        style_latents = utils.reparameterize(style_mean, style_log_var)
                        results["style_latent_samples"][modality.name] = style_latents
                    else:
                        style_latents = None

                    results_reconstructions[modality.name] = modality.likelihood(
                        *decoder(style_latents, class_latents)
                    )
        results["rec"] = results_reconstructions
        return results

    def encode(
        self, input_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        encoded_modalities: Dict[str, List[Optional[torch.Tensor]]] = dict()
        for modality, encoder in zip(self._modalities, self._encoders):
            if modality.name in input_batch:
                modality_input_batch = input_batch[modality.name]
                if modality_input_batch is not None:
                    encoded = encoder(modality_input_batch)
                    encoded_style = encoded[:2]
                    encoded_class = encoded[2:]
                else:
                    encoded_style = [None, None]
                    encoded_class = [None, None]

                encoded_modalities[modality.name] = encoded_class
                encoded_modalities[f"{modality.name}_style"] = encoded_style

        return encoded_modalities

    def get_random_styles(self, num_samples: int) -> Dict[str, torch.Tensor]:
        styles: Dict[str, torch.Tensor] = dict()
        for modality in self._modalities:
            if self._factorized_representation:
                z_style = torch.randn(num_samples, self._style_dim, device=self._device)
            else:
                z_style = None
            styles[modality.name] = z_style
        return styles

    def get_random_style_dists(self, num_samples):
        style_parameters: Dict[str, List[torch.Tensor, torch.Tensor]] = dict()
        for modality in self._modalities:
            style_mean = torch.zeros(num_samples, self._style_dim, device=self._device)
            style_log_var = torch.zeros(
                num_samples, self._style_dim, device=self._device
            )
            style_parameters[modality.name] = [style_mean, style_log_var]

        return style_parameters

    def generate_sufficient_statistics_from_latents(
        self, latents
    ) -> Dict[str, torch.distributions.Distribution]:
        latents_style = latents["style"]
        latents_content = latents["content"]
        conditionally_generated: Dict[str, torch.distributions.Distribution] = dict()
        for modality, decoder in zip(self._modalities, self._decoders):
            conditionally_generated[modality.name] = modality.likelihood(
                *decoder(latents_style[modality.name], latents_content)
            )

        return conditionally_generated

    def pre_epoch_callback(self, epoch: int, num_iterations: int):
        self._epoch = epoch
        self._num_iterations = num_iterations

    def post_epoch_callback(self, epoch: int):
        pass

    def pre_iteration_callback(self, iteration: int):
        self._iteration = iteration

    def post_iteration_callback(self, iteration: int):
        pass

    def _set_fusion_functions(self):
        weights = utils.reweight_weights(torch.Tensor(self.flags.alpha_modalities))
        self.weights = weights.to(self.flags.device)
        if self.flags.modality_moe:
            self.modality_fusion = self.moe_fusion
            self.fusion_condition = self.fusion_condition_moe
            self.calc_joint_divergence = self.divergence_static_prior
        elif self.flags.modality_jsd:
            self.modality_fusion = self.moe_fusion
            self.fusion_condition = self.fusion_condition_moe
            self.calc_joint_divergence = self.divergence_dynamic_prior
        elif self.flags.modality_poe:
            self.modality_fusion = self.poe_fusion
            self.fusion_condition = self.fusion_condition_poe
            self.calc_joint_divergence = self.divergence_static_prior
        elif self.flags.mopoe:
            self.modality_fusion = self.poe_fusion
            self.fusion_condition = self.fusion_condition_joint
            self.calc_joint_divergence = self.divergence_static_prior

    def divergence_static_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = weights.clone()
        weights = utils.reweight_weights(weights)
        div_measures = calc_group_divergence_moe(
            self.flags, mus, logvars, weights, normalization=self.flags.batch_size
        )
        divs = dict()
        divs["joint_divergence"] = div_measures[0]
        divs["individual_divs"] = div_measures[1]
        divs["dyn_prior"] = None
        return divs

    def divergence_dynamic_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        div_measures = calc_alphaJSD_modalities(
            self.flags, mus, logvars, weights, normalization=self.flags.batch_size
        )
        divs = dict()
        divs["joint_divergence"] = div_measures[0]
        divs["individual_divs"] = div_measures[1]
        divs["dyn_prior"] = div_measures[2]
        return divs

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights
        weights = utils.reweight_weights(weights)
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        mu_moe, logvar_moe = utils.mixture_component_selection(
            self.flags, mus, logvars, weights
        )
        return [mu_moe, logvar_moe]

    def poe_fusion(self, mus, logvars, weights=None):
        if self.flags.modality_poe:
            num_samples = mus[0].shape[0]
            if self.flags.poe_product_prior:
                mus = torch.cat(
                    (
                        mus,
                        torch.zeros(1, num_samples, self.flags.class_dim, device=self.flags.device),
                    ),
                    dim=0,
                )
                logvars = torch.cat(
                    (
                        logvars,
                        torch.zeros(1, num_samples, self.flags.class_dim, device=self.flags.device),
                    ),
                    dim=0,
                )
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        if hasattr(self.flags, "stable_poe") and self.flags.stable_poe:
            mu_poe, logvar_poe = poe_stable(mus, logvars)
        else:
            mu_poe, logvar_poe = poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def fusion_condition_moe(self, subset, input_batch=None):
        if len(subset) == 1:
            return True
        else:
            return False

    def fusion_condition_poe(self, subset, input_batch=None):
        if len(subset) == len(input_batch.keys()):
            return True
        else:
            return False

    def fusion_condition_joint(self, subset, input_batch=None):
        return True

    def inference(self, input_batch, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size
        latents = dict()
        enc_mods = self.encode(input_batch)

        # Multiply variances with expert temperature. This is equivalent to
        # dividing the exponent of the normal density by the temperature.
        if hasattr(self.flags, "expert_temperature"):
            for modality in enc_mods:
                if (
                    modality in self.flags.expert_temperature
                    and enc_mods[modality] is not None
                    and enc_mods[modality][0] is not None
                ):
                    logvar_temped = enc_mods[modality][1] + torch.log(
                        self.flags.expert_temperature[modality]
                    )
                    enc_mods[modality] = enc_mods[modality][0], logvar_temped

        # This fixes the covariance matrices to scaled identity variances.
        # Used for experiment that explores the influence of miscalibrated precisions.
        if (
            hasattr(self.flags, "latent_constant_variances")
            and self.flags.latent_constant_variances is not None
        ):
            const = (
                torch.tensor([self.flags.latent_constant_variances], dtype=torch.float, device=self.flags.device).log()
            )
            for modality in enc_mods:
                if enc_mods[modality][1] is not None:
                    enc_mods[modality] = (
                        enc_mods[modality][0],
                        torch.zeros_like(enc_mods[modality][1], device=self.flags.device)
                        + const,
                    )

        if (
            hasattr(self.flags, "latent_variances_offset")
            and self.flags.latent_variances_offset is not None
        ):
            for modality in enc_mods:
                if enc_mods[modality][1] is not None:
                    offset = self.flags.latent_variances_offset
                    if (
                        offset > 0
                        and self.flags.anneal_latent_variance_by_epoch is not None
                    ):
                        offset = offset * max(
                            0,
                            1
                            - (self._epoch * self._num_iterations + self._iteration)
                            / (
                                self.flags.anneal_latent_variance_by_epoch
                                * self._num_iterations
                            ),
                        )

                    if offset > 0:
                        # This may be numerically problematic (log(exp(...))
                        log_var = enc_mods[modality][1].unsqueeze(0)
                        log_offset = (torch.ones_like(log_var, device=self._device)*offset).log()
                        log_var_offset = torch.logsumexp(torch.cat((log_var, log_offset), dim=0), dim=0)
                        enc_mods[modality] = enc_mods[modality][0], log_var_offset

        if (
            hasattr(self.flags, "force_partition_latent_space_mode")
            and (self.flags.force_partition_latent_space_mode == "variance")
            and (self.flags.partition_latent_space_variance_offset > 0)
        ):
            slice_size = self.flags.class_dim // self.flags.num_mods
            for modality_idx, modality in enumerate(self._modalities):
                if modality.name not in enc_mods:
                    continue
                means, logvars = enc_mods[modality.name]
                if means is not None and logvars is not None:
                    logvars_slice = logvars[:, :(slice_size * modality_idx)].unsqueeze(0)
                    log_offset_slice = (torch.ones_like(logvars_slice, device=self._device) * self.flags.partition_latent_space_variance_offset).log()
                    logvars[:, :(slice_size * modality_idx)] = torch.logsumexp(torch.cat((logvars_slice, log_offset_slice), dim=0), dim=0)

                    logvars_slice = logvars[:, (slice_size * (modality_idx+1)):].unsqueeze(0)
                    log_offset_slice = (torch.ones_like(logvars_slice, device=self._device) * self.flags.partition_latent_space_variance_offset).log()
                    logvars[:, (slice_size * (modality_idx+1)):] = torch.logsumexp(torch.cat((logvars_slice, log_offset_slice), dim=0), dim=0)

                    enc_mods[modality.name] = means, logvars

        if (
            hasattr(self.flags, "poe_variance_clipping")
            and self.flags.poe_variance_clipping[0] != float("inf")
            and self.flags.poe_variance_clipping[1] != float("inf")
        ):
            a, b = self.flags.poe_variance_clipping
            if a == float("inf"):
                a = -a
            for modality in self._modalities:
                if (
                    enc_mods[modality.name][0] is not None
                    and enc_mods[modality.name][1] is not None
                ):
                    means, logvars = enc_mods[modality.name]
                    enc_mods[modality.name] = means, logvars.clip(a, b)

        latents["modalities"] = enc_mods
        mus = torch.tensor([], device=self.flags.device)
        logvars = torch.tensor([], device=self.flags.device)
        distr_subsets = dict()
        for subset_name, subset_modalities in self._subsets_names_modalities:
            if len(subset_modalities) > 0:
                mus_subset = torch.tensor([], device=self.flags.device)
                logvars_subset = torch.tensor([], device=self.flags.device)
                mods_avail = True
                normalizing_temperature = (
                    torch.tensor(
                        [
                            len(subset_modalities)
                            if (
                                hasattr(self.flags, "poe_normalize_experts")
                                and self.flags.poe_normalize_experts
                            )
                            else 1
                        ],
                        dtype=torch.float,
                        device=self.flags.device
                    )
                    .log()
                )
                for modality in subset_modalities:
                    if modality.name in input_batch.keys():
                        if (
                            hasattr(self.flags, "poe_normalize_experts")
                            and self.flags.poe_normalize_experts
                        ):
                            enc_mods[modality.name] = (
                                enc_mods[modality.name][0],
                                enc_mods[modality.name][1] + normalizing_temperature,
                            )

                        batch_size = enc_mods[modality.name][0].shape[0]

                        if (
                            hasattr(self.flags, "force_partition_latent_space_mode")
                            and self.flags.force_partition_latent_space_mode
                            == "concatenate"
                        ):
                            if mus_subset.shape[0] == 0:
                                mus_subset = torch.zeros(
                                    (1, batch_size, self.flags.class_dim), device=self.flags.device
                                )
                            if logvars_subset.shape[0] == 0:
                                logvars_subset = torch.zeros(
                                    (1, batch_size, self.flags.class_dim), device=self.flags.device
                                )

                            modality_index = self._modality_to_index[modality.name]
                            slice_start = (
                                modality_index * self._modality_concat_slice_size
                            )
                            slice_end = slice_start + self._modality_concat_slice_size
                            mus_subset[0, :, slice_start:slice_end] = enc_mods[
                                modality.name
                            ][0]
                            logvars_subset[0, :, slice_start:slice_end] = enc_mods[
                                modality.name
                            ][1]

                        else:
                            mus_subset = torch.cat(
                                (mus_subset, enc_mods[modality.name][0].unsqueeze(0)), dim=0
                            )
                            logvars_subset = torch.cat(
                                (logvars_subset, enc_mods[modality.name][1].unsqueeze(0)),
                                dim=0,
                            )
                    else:
                        mods_avail = False
                if mods_avail:
                    weights_subset = (1 / float(len(mus_subset))) * torch.ones(
                        len(mus_subset), device=self.flags.device)
                    s_mu, s_logvar = self.modality_fusion(
                        mus_subset, logvars_subset, weights_subset
                    )
                    distr_subsets[subset_name] = [s_mu, s_logvar]
                    if self.fusion_condition(subset_modalities, input_batch):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)
        if self.flags.modality_jsd:
            mus = torch.cat(
                (
                    mus,
                    torch.zeros(1, num_samples, self.flags.class_dim, device=self.flags.device),
                ),
                dim=0,
            )
            logvars = torch.cat(
                (
                    logvars,
                    torch.zeros(1, num_samples, self.flags.class_dim, device=self.flags.device),
                ),
                dim=0,
            )
        # weights = (1/float(len(mus)))*torch.ones(len(mus), device=self.flags.device)
        weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0], device=self.flags.device)
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)
        # mus = torch.cat(mus, dim=0);
        # logvars = torch.cat(logvars, dim=0);
        latents["mus"] = mus
        latents["logvars"] = logvars
        latents["weights"] = weights
        latents["joint"] = [joint_mu, joint_logvar]
        latents["subsets"] = distr_subsets
        return latents

    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        mu = torch.zeros(num_samples, self.flags.class_dim, device=self.flags.device)
        logvar = torch.zeros(num_samples, self.flags.class_dim, device=self.flags.device)
        z_class = utils.reparameterize(mu, logvar)
        z_styles = self.get_random_styles(num_samples)
        random_latents = {"content": z_class, "style": z_styles}
        random_samples = self.generate_from_latents(random_latents)
        return random_samples

    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents)
        cond_gen = dict()
        for m, m_key in enumerate(latents["style"].keys()):
            cond_gen_m = suff_stats[m_key].mean
            cond_gen[m_key] = cond_gen_m
        return cond_gen

    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size

        style_latents = self.get_random_styles(num_samples)
        cond_gen_samples = dict()
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key]
            content_rep = utils.reparameterize(mu=mu, logvar=logvar)
            latents = {"content": content_rep, "style": style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents)
        return cond_gen_samples
