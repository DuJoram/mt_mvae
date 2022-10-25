import math
from typing import List, Dict, Tuple

import torch
import tqdm
from torch.utils.data import DataLoader

from models import MultimodalVAE


@torch.no_grad()
def estimate_marginal_and_joint_entropies(
    means: torch.Tensor,
    log_vars: torch.Tensor,
    latent_size: int,
    samples_batch_size: int,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimates
        H[Z]
    and
        H[Z_i]
    for i = 1,...,d
    where d is the latent size.

    Heavily inspired by https://github.com/rtqichen/beta-tcvae
    """

    num_samples = means.shape[0]
    num_latent_samples = 10_000
    perm = torch.randperm(num_samples, device=device)[:num_latent_samples]
    means_selected = means.index_select(0, perm)
    log_vars_selected = log_vars.index_select(0, perm)

    q_dist = torch.distributions.Normal(
        loc=means_selected, scale=log_vars_selected.mul(0.5).exp()
    )

    samples = q_dist.sample()

    marginal_entropies = torch.zeros((latent_size,), device=device)
    joint_entropy = torch.zeros((1,), device=device)

    num_sample_batches = num_latent_samples // samples_batch_size
    for idx in tqdm.trange(num_sample_batches, desc="Estimating log likelihoods"):
        batch_start = idx * samples_batch_size
        batch_end = batch_start + samples_batch_size
        dist = torch.distributions.Normal(
            loc=means.unsqueeze(2).repeat(1, 1, samples_batch_size),
            scale=log_vars.mul(0.5).exp().unsqueeze(2).repeat(1, 1, samples_batch_size),
        )
        log_qz_i = dist.log_prob(
            samples[batch_start:batch_end].T.unsqueeze(0).repeat(num_samples, 1, 1)
        )

        marginal_entropies += (
            math.log(num_samples) - torch.logsumexp(log_qz_i, dim=0)
        ).sum(1)

        log_qz = log_qz_i.sum(1)

        joint_entropy += (math.log(num_samples) - torch.logsumexp(log_qz, dim=0)).sum(0)

    marginal_entropies /= num_latent_samples
    joint_entropy /= num_latent_samples

    return marginal_entropies, joint_entropy


@torch.no_grad()
def estimate_total_correlation(
    data_loader: DataLoader,
    subset_names: List[str],
    mm_vae: MultimodalVAE,
    batch_size: int,
    latent_size: int,
    num_samples: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, float]:
    total_correlations: Dict[str, float] = dict()

    means: Dict[str, torch.Tensor] = dict()
    log_vars: Dict[str, torch.Tensor] = dict()

    subset_names = [subset_name for subset_name in subset_names if len(subset_name) > 0]

    for subset_name in subset_names:

        means[subset_name] = torch.empty(
            (len(data_loader) * batch_size, latent_size), device=device
        )
        log_vars[subset_name] = torch.empty(
            (len(data_loader) * batch_size, latent_size), device=device
        )

    for batch_idx, (batch_data, batch_label) in tqdm.tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Loading latent parameters"
    ):
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        for modality_name, modality_data in batch_data.items():
            batch_data[modality_name] = modality_data.to(device)

        inferred = mm_vae.inference(batch_data)

        for subset_name in subset_names:
            means[subset_name][start_idx:end_idx] = inferred["subsets"][subset_name][0]
            log_vars[subset_name][start_idx:end_idx] = inferred["subsets"][subset_name][
                1
            ]

    for subset_name in tqdm.tqdm(subset_names, desc="Iterating subsets"):
        marginal_entropies, joint_entropy = estimate_marginal_and_joint_entropies(
            means=means[subset_name],
            log_vars=log_vars[subset_name],
            latent_size=latent_size,
            samples_batch_size=num_samples,
            device=device,
            verbose=verbose,
        )

        total_correlations[subset_name] = (
            marginal_entropies.sum() - joint_entropy
        ).item()

    return total_correlations
