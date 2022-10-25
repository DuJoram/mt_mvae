import math
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from divergence_measures.kl_div import calc_kl_divergence
from eval_metrics.coherence import test_generation
from eval_metrics.latent_partitioning import train_experts_classifier, test_expert_classifier
from eval_metrics.likelihood import estimate_likelihoods
from eval_metrics.representation import test_classifier_latent_representation_all_subsets
from eval_metrics.representation import train_classifier_latent_representation_all_subsets
from eval_metrics.sample_quality import calc_prd_score
from experiment_logger import WAndBLogger, TBLogger, ExperimentLogger
from experiments import Experiment
from models.multimodal_vae import MultimodalVAE
from plotting import generate_plots
from utils import utils

# global variables
SEED = None
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


def calc_log_probs(exp, result, batch):
    modalities = exp.modalities
    batch_size = exp.flags.batch_size
    rec_weights = exp.rec_weights

    log_probs = dict()
    weighted_log_prob = 0.0
    for modality in modalities:
        log_probs[modality.name] = -modality.calc_log_prob(
            result['rec'][modality.name],
            batch[0][modality.name],
            batch_size
        )
        weighted_log_prob += exp.rec_weights[modality.name]*log_probs[modality.name]
    return log_probs, weighted_log_prob


def calc_klds(exp, result):
    latents = result['latents']['subsets']
    klds = dict()
    bs = exp.flags.batch_size
    c_dim = exp.flags.class_dim
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar, norm_value=bs)
    return klds;


def calc_klds_style(exp, result):
    latents = result['latents']['modalities'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        if key.endswith('style'):
            mu, logvar = latents[key];
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=exp.flags.batch_size)
    return klds;


def calc_style_kld(exp, klds):
    modalities = exp.modalities;
    style_weights = exp.style_weights;
    weighted_klds = 0.0;
    for modality in modalities:
        weighted_klds += style_weights[modality.name]*klds[modality.name + '_style'];
    return weighted_klds;

def kld_decompositions_tcmvae(
    sampled_latents: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
    batch_size: int,
    dataset_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Code slightly adapted from: https://github.com/rtqichen/beta-tcvae/

    class_dim = means.shape[-1]

    log2pi = math.log(math.sqrt(2*math.pi))
    q_normalizer = 0.5*log_vars + log2pi

    # `m` and n denote batch dimension, i denotes latent element index
    log_qzmi_xm = -0.5*((sampled_latents - means)**2) * (-log_vars).exp() - q_normalizer
    log_qzmi_xn = -0.5*((sampled_latents.unsqueeze(1) - means.unsqueeze(0))**2) * (-log_vars).exp() - q_normalizer
    log_pzmi = -sampled_latents**2/2 - log2pi

    logpz = log_pzmi.view(batch_size, -1).sum(1)
    log_qzm_xm = log_qzmi_xm.view(batch_size, -1).sum(1)

    log_qzmi = (torch.logsumexp(log_qzmi_xn, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
    logqz = (torch.logsumexp(log_qzmi_xn.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))

    index_code_mi = torch.mean(log_qzm_xm - logpz)
    total_correlation = torch.mean(logqz - log_qzmi)
    dim_wise_kld = torch.mean(log_qzmi - logpz)

    return index_code_mi, total_correlation, dim_wise_kld

def get_total_correlation_weight(target_weight: float, current_epoch: int, target_epoch, wait_epochs) -> float:
    if target_epoch < 0:
        return target_weight

    if current_epoch < wait_epochs:
        return 1.

    assert target_epoch > wait_epochs
    return 1 + (target_weight - 1) * min(1, (current_epoch - wait_epochs) / (target_epoch - wait_epochs))


def basic_routine_epoch(exp: Experiment, batch, dataset_size: int, epoch: int = None):
    mm_vae = exp.mm_vae;
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta
    modalities = exp.modalities


    # set up weights
    rec_weight = 1.0;

    batch_data = batch[0];
    for k, modality_name in enumerate(batch_data.keys()):
        batch_data[modality_name] = batch_data[modality_name].to(exp.flags.device);
    results = mm_vae(batch_data);

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch);
    group_divergence = results['joint_divergence'];

    latents = results["latents"]
    kld_decompositions = {"class": dict(), "style": dict()}
    joint_mean, joint_log_var = latents["joint"]
    joint_class_index_code_mi, joint_class_total_correlation, joint_class_dim_wise_kld = kld_decompositions_tcmvae(
        sampled_latents=results["class_latent_samples"],
        means=joint_mean,
        log_vars=joint_log_var,
        batch_size=exp.flags.batch_size,
        dataset_size=dataset_size,
    )

    kld_decompositions["class"][exp.modalities_subsets_names[-1]] = {
        "index_code_mi": joint_class_index_code_mi,
        "total_correlation": joint_class_total_correlation,
        "dim_wise_kld": joint_class_dim_wise_kld
    }

    if exp.flags.factorized_representation:
        kld_decompositions["style"]["joint"] = dict()
        for modality in modalities:
            style_mean, style_log_var = latents["modalities"][modality.name + "_style"]
            style_index_code_mi, style_total_correlation, style_dim_wise_kld = kld_decompositions_tcmvae(
                sampled_latents=results["style_latent_samples"][modality.name],
                means=style_mean,
                log_vars=style_log_var,
                batch_size=exp.flags.batch_size,
                dataset_size=dataset_size,
            )
            kld_decompositions["style"][exp.modalities_subsets_names[-1]][modality.name] = {
                "index_code_mi": style_index_code_mi,
                "total_correlation": style_total_correlation,
                "dim_wise_kld": style_dim_wise_kld,
            }


    klds = calc_klds(exp, results);
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results);

    if (exp.flags.modality_jsd or exp.flags.modality_moe
        or exp.flags.mopoe):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style);
        else:
            kld_style = 0.0;
        kld_content = group_divergence;
        kld_weighted = beta_style * kld_style + beta_content * kld_content;
        total_loss = rec_weight * weighted_log_prob + beta * kld_weighted;
        total_loss_unweighted = weighted_log_prob + kld_content + kld_style
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()};
        recs_joint = dict();
        elbos = dict();
        elbos_unw = {}

        elbos_tcmvae = dict()
        elbos_tcmvae_unweighted = dict()

        subsets_indices = np.array([], dtype=int)

        total_correlation_weight = get_total_correlation_weight(
            target_weight=exp.flags.total_correlation_weight,
            current_epoch=epoch,
            target_epoch=exp.flags.total_correlation_weight_target_epoch,
            wait_epochs=exp.flags.total_correlation_weight_wait_epochs,
        )

        if exp.flags.poe_unimodal_elbos:
            subsets_indices = np.concatenate([subsets_indices, np.arange(exp.num_modalities)+1])

        if exp.flags.poe_num_subset_elbos > 0:
            random_subsets = np.random.choice(
                range(exp.num_modalities + 1, len(exp.modalities_subsets_indices) - 1),
                size=exp.flags.poe_num_subset_elbos,
                replace=False,
            )

            subsets_indices = np.concatenate([subsets_indices, random_subsets])


        for subset_index in subsets_indices:
            subset_key = exp.modalities_subsets_names[subset_index]
            subset = exp.modalities_subsets[subset_key]

            subset_batch = {modality.name: batch_data[modality.name] for modality in subset}

            result_subset = mm_vae(subset_batch)

            content_kl_divergence = klds[subset_key]

            weighted_style_kld = 0.0
            unweighted_style_kld = 0.0

            weighted_style_index_code_mi = 0.0
            unweighted_style_index_code_mi = 0.0

            weighted_style_total_correlation = 0.0
            unweighted_style_total_correlation = 0.0

            weighted_style_dim_wise_kld = 0.0
            unweighted_style_dim_wise_kld = 0.0

            weighted_reconstruction_error = 0.0
            unweighted_reconstruction_error = 0.0

            if exp.flags.tcmvae or exp.flags.elbo_add_tc:
                class_index_code_mi, class_total_correlation, class_dim_wise_klds = kld_decompositions_tcmvae(
                    sampled_latents=result_subset["class_latent_samples"],
                    means=result_subset["latents"]["joint"][0],
                    log_vars=result_subset["latents"]["joint"][1],
                    batch_size=exp.flags.batch_size,
                    dataset_size=dataset_size,
                )

                kld_decompositions["class"][subset_key] = {
                    "index_code_mi": class_index_code_mi,
                    "total_correlation": class_total_correlation,
                    "dim_wise_klds": class_dim_wise_klds,
                }
                kld_decompositions["style"][subset_key] = dict()

            for modality in subset:

                if exp.flags.factorized_representation:
                    style_kld = klds_style[modality.name + '_style'] if exp.flags.factorized_representation else 0.0
                    weighted_style_kld += exp.style_weights[modality.name] * style_kld
                    unweighted_style_kld += style_kld

                    if exp.flags.tcmvae or exp.flags.elbo_add_tc:
                        style_index_code_mi, style_total_correlation, style_dim_wise_kld = kld_decompositions_tcmvae(
                            sampled_latents=result_subset["style_latent_samples"][modality.name],
                            means=result_subset["latents"]["modalities"][modality.name + "_style"][0],
                            log_vars=result_subset["latents"]["modalities"][modality.name + "_style"][1],
                            batch_size=exp.flags.batch_size,
                            dataset_size=dataset_size,
                        )

                        weighted_style_index_code_mi += exp.style_weights[modality.name] * style_index_code_mi
                        weighted_style_total_correlation += total_correlation_weight * exp.style_weights[modality.name] * style_total_correlation
                        weighted_style_dim_wise_kld += exp.style_weights[modality.name] * style_dim_wise_kld
                        unweighted_style_index_code_mi += style_index_code_mi
                        unweighted_style_total_correlation += style_total_correlation
                        unweighted_style_dim_wise_kld += style_dim_wise_kld
                        kld_decompositions["style"][subset_key][modality.name] = {
                            "index_code_mi": style_index_code_mi,
                            "total_correlation": style_total_correlation,
                            "dim_wise_kld": style_dim_wise_kld,
                        }

                log_prob_modality = modality.calc_log_prob(result_subset['rec'][modality.name], batch_data[modality.name], exp.flags.batch_size)

                weighted_reconstruction_error -= exp.rec_weights[modality.name] * log_prob_modality
                unweighted_reconstruction_error -= log_prob_modality

            weighted_kl_divergence = exp.flags.beta_style * weighted_style_kld + exp.flags.beta * content_kl_divergence
            unweighted_kl_divergence = unweighted_style_kld + content_kl_divergence

            if exp.flags.tcmvae or exp.flags.elbo_add_tc:
                weighted_subset_elbo_tcmvae = weighted_reconstruction_error + exp.flags.beta * (
                    exp.flags.index_code_mi_weight * class_index_code_mi +
                    total_correlation_weight * class_total_correlation +
                    exp.flags.dimension_wise_kld_weight * class_dim_wise_klds
                ) + exp.flags.beta_style * (
                    exp.flags.index_code_mi_weight * weighted_style_index_code_mi +
                    total_correlation_weight * weighted_style_total_correlation +
                    exp.flags.dimension_wise_kld_weight * weighted_style_total_correlation
                )

                unweighted_subset_elbo_tcmvae = (
                    unweighted_reconstruction_error +
                    class_index_code_mi + unweighted_style_index_code_mi +
                    class_total_correlation + unweighted_style_total_correlation +
                    class_dim_wise_klds + unweighted_style_total_correlation
                )
                elbos_tcmvae[subset_key] = weighted_subset_elbo_tcmvae
                elbos_tcmvae_unweighted[subset_key] = unweighted_subset_elbo_tcmvae

            weighted_subset_elbo = weighted_reconstruction_error + weighted_kl_divergence
            unweighted_subset_elbo = unweighted_reconstruction_error + unweighted_kl_divergence


            elbos[subset_key] = weighted_subset_elbo
            elbos_unw[subset_key] = unweighted_subset_elbo

        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint);
        elbo_joint_unw = utils.calc_elbo(exp, 'joint', log_probs, klds_joint, weighted=False);

        reconstruction_error = 0.0
        reconstruction_error_unweighted = 0.0
        for modality in exp.modalities:
            reconstruction_error += exp.rec_weights[modality.name] * log_probs[modality.name]
            reconstruction_error_unweighted = log_probs[modality.name]

        if exp.flags.tcmvae or exp.flags.elbo_add_tc:
            joint_style_index_code_mi = 0.0
            joint_style_total_correlation = 0.0
            joint_style_dim_wise_kld = 0.0
            if exp.flags.factorized_representation:
                for values in kld_decompositions["style"][exp.modalities_subsets_names[-1]].values():
                    joint_style_index_code_mi += values["index_code_mi"]
                    joint_style_total_correlation += values["total_correlation"]
                    joint_style_dim_wise_kld += values["dim_wise_kld"]



            joint_class_kld_decomposed = exp.flags.index_code_mi_weight * joint_class_index_code_mi + total_correlation_weight * joint_class_total_correlation + exp.flags.dimension_wise_kld_weight * joint_class_dim_wise_kld
            joint_class_kld_decomposed_unweighted = joint_class_index_code_mi + joint_class_total_correlation + joint_class_dim_wise_kld

            joint_style_kld_decomposed = exp.flags.index_code_mi_weight * joint_style_index_code_mi + total_correlation_weight * joint_style_total_correlation + exp.flags.dimension_wise_kld_weight * joint_style_dim_wise_kld
            joint_style_kld_decomposed_unweighted = joint_style_index_code_mi + joint_style_total_correlation + joint_style_dim_wise_kld

            elbos_tcmvae["joint"] = reconstruction_error + exp.flags.beta * joint_class_kld_decomposed + exp.flags.beta_style * joint_style_kld_decomposed
            elbos_tcmvae_unweighted["joint"] = reconstruction_error_unweighted + joint_class_kld_decomposed_unweighted + joint_style_kld_decomposed_unweighted

        elbos['joint'] = elbo_joint;
        elbos_unw['joint'] = elbo_joint_unw

        if exp.flags.elbo_add_tc:
            elbos['joint'] += exp.flags.beta * total_correlation_weight * joint_class_total_correlation + exp.flags.beta_style * total_correlation_weight * joint_style_total_correlation
            elbos_unw['joint'] += joint_class_total_correlation + joint_style_total_correlation


    out_basic_routine = dict();
    out_basic_routine['results'] = results;
    out_basic_routine['log_probs'] = log_probs;
    if exp.flags.tcmvae or exp.flags.elbo_add_tc:
        out_basic_routine['total_loss_tcmvae'] = sum(elbos_tcmvae.values())
        out_basic_routine['total_loss_tcmvae_unweighted'] = sum(elbos_tcmvae_unweighted.values())
        out_basic_routine['kld_decompositions'] = kld_decompositions
    out_basic_routine['total_loss'] = sum(elbos.values())
    out_basic_routine['total_loss_unweighted'] = sum(elbos_unw.values())
    out_basic_routine['klds'] = klds;
    return out_basic_routine;

def train(epoch, exp, data_loader, tb_logger):
    mm_vae: MultimodalVAE = exp.mm_vae;
    mm_vae.train();
    exp.mm_vae = mm_vae;


    num_batches = float(len(data_loader))
    sum_loss = 0.0
    sum_loss_unweighted = 0.0
    sum_loss_tcmvae = 0.0
    sum_loss_tcmvae_unweighted = 0.0
    total_loss_tcmvae_unweighted = None
    kld_decompositions = None
    total_loss_tcmvae = None
    total_loss_tcmvae_unweighted = None
    mm_vae.pre_epoch_callback(epoch, int(num_batches))
    for iteration, batch in tqdm.tqdm(enumerate(data_loader), total=num_batches, desc="Training", file=sys.stdout, leave=False):
        mm_vae.pre_iteration_callback(iteration)
        basic_routine = basic_routine_epoch(exp, batch, dataset_size=len(exp.dataset_train), epoch=epoch);
        results = basic_routine['results'];
        total_loss = basic_routine['total_loss'];
        total_loss_unweighted = basic_routine['total_loss_unweighted']
        if exp.flags.tcmvae or exp.flags.elbo_add_tc:
            total_loss_tcmvae = basic_routine['total_loss_tcmvae']
            total_loss_tcmvae_unweighted = basic_routine['total_loss_tcmvae_unweighted']
            kld_decompositions = basic_routine['kld_decompositions']
        klds = basic_routine['klds'];
        log_probs = basic_routine['log_probs'];

        # backprop
        exp.optimizer.zero_grad()
        if exp.flags.tcmvae:
            total_loss_tcmvae.backward()
        else:
            total_loss.backward()
        exp.optimizer.step()
        with torch.no_grad():
            tb_logger.write_training_logs(results, total_loss, total_loss_unweighted, total_loss_tcmvae, total_loss_tcmvae_unweighted, log_probs, klds, kld_decompositions);
            sum_loss += total_loss
            sum_loss_unweighted += total_loss_unweighted
            if exp.flags.tcmvae or exp.flags.elbo_add_tc:
                sum_loss_tcmvae += total_loss_tcmvae
                sum_loss_tcmvae_unweighted += total_loss_tcmvae_unweighted
            mm_vae.post_iteration_callback(int(iteration))
    with torch.no_grad():
        mm_vae.post_epoch_callback(epoch)
        avg_loss = sum_loss / num_batches
        avg_loss_unweighted = sum_loss_unweighted/num_batches
        avg_loss_tcmvae = None
        avg_loss_tcmvae_unweighted = None
        if exp.flags.tcmvae or exp.flags.elbo_add_tc:
            avg_loss_tcmvae = sum_loss_tcmvae/num_batches
            avg_loss_tcmvae_unweighted = sum_loss_tcmvae_unweighted/num_batches
        tb_logger.write_avg_loss('train', avg_loss, avg_loss_unweighted, avg_loss_tcmvae, avg_loss_tcmvae_unweighted)


def test(epoch: int, exp: Experiment, train_data_loader: DataLoader, test_data_loader: DataLoader, tb_logger: ExperimentLogger):
    with torch.no_grad():
        mm_vae = exp.mm_vae;
        mm_vae.eval();
        exp.mm_vae = mm_vae;

        # set up weights
        beta_style = exp.flags.beta_style;
        beta_content = exp.flags.beta_content;
        beta = exp.flags.beta;
        rec_weight = 1.0;

        num_batches = float(len(test_data_loader))
        sum_loss = 0.0
        sum_loss_unweighted = 0.0
        sum_loss_tcmvae = 0.0
        sum_loss_tcmvae_unweighted = 0.0
        total_loss_tcmvae = None
        total_loss_tcmvae_unweighted = None
        kld_decompositions = None
        avg_loss_unweighted = None
        avg_loss_tcmvae = None
        avg_loss_tcmvae_unweighted = None
        for iteration, batch in tqdm.tqdm(enumerate(test_data_loader), desc="Testing", total=num_batches, file=sys.stdout, leave=False):
            basic_routine = basic_routine_epoch(exp, batch, len(exp.dataset_test), epoch=epoch);
            results = basic_routine['results'];
            total_loss = basic_routine['total_loss'];
            total_loss_unweighted = basic_routine['total_loss_unweighted']
            if exp.flags.tcmvae or exp.flags.elbo_add_tc:
                total_loss_tcmvae = basic_routine['total_loss_tcmvae'];
                total_loss_tcmvae_unweighted = basic_routine['total_loss_tcmvae_unweighted']
                kld_decompositions = basic_routine['kld_decompositions']
            klds = basic_routine['klds'];
            log_probs = basic_routine['log_probs'];
            tb_logger.write_testing_logs(results, total_loss, total_loss_unweighted, total_loss_tcmvae, total_loss_tcmvae_unweighted, log_probs, klds, kld_decompositions);
            sum_loss += total_loss
            sum_loss_unweighted += total_loss_unweighted
            if exp.flags.tcmvae or exp.flags.elbo_add_tc:
                sum_loss_tcmvae += total_loss_tcmvae
                sum_loss_tcmvae_unweighted += total_loss_tcmvae_unweighted
        avg_loss = sum_loss/num_batches
        avg_loss_unweighted = sum_loss_unweighted/num_batches
        if exp.flags.tcmvae or exp.flags.elbo_add_tc:
            avg_loss_tcmvae = sum_loss/num_batches
            avg_loss_tcmvae_unweighted = sum_loss_unweighted/num_batches
        tb_logger.write_avg_loss('test', avg_loss, avg_loss_unweighted, avg_loss_tcmvae, avg_loss_tcmvae_unweighted)

        if (epoch + 1) % exp.flags.plotting_freq == 0:
            plots = generate_plots(exp, epoch);
            tb_logger.write_plots(plots, epoch);

        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:
            if exp.flags.eval_lr:
                latent_representation_classifiers = train_classifier_latent_representation_all_subsets(exp, train_data_loader);
                latent_representation_eval = test_classifier_latent_representation_all_subsets(epoch, latent_representation_classifiers, exp, test_data_loader);
                tb_logger.write_lr_eval(latent_representation_eval);

            if exp.flags.use_expert_classifier:
                expert_classifier = train_experts_classifier(exp, train_data_loader)
                results = test_expert_classifier(exp, expert_classifier, test_data_loader)
                tb_logger.write_expert_classifier_statistics(results)

            if exp.flags.use_classifier:
                gen_eval = test_generation(epoch, exp, test_data_loader);
                tb_logger.write_coherence_logs(gen_eval);

            if exp.flags.calc_nll:
                lhoods = estimate_likelihoods(exp);
                tb_logger.write_lhood_logs(lhoods);

            if exp.flags.calc_prd and ((epoch + 1) % exp.flags.eval_freq_fid == 0):
                prd_scores = calc_prd_score(exp);
                tb_logger.write_prd_scores(prd_scores)


def run_epochs(exp):
    # initialize summary writer
    # writer = SummaryWriter(exp.flags.dir_logs)
    # tb_logger = TBLogger(exp.flags.str_experiment, exp.flags.dir_logs)
    tb_logger = WAndBLogger(
        logdir=exp.flags.dir_logs,
        name=exp.flags.run_name,
    )
    tb_logger.write_flags(exp.flags)

    num_workers = exp.flags.num_workers

    test_data_loader = DataLoader(
        exp.dataset_test,
        batch_size=exp.flags.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    train_data_loader = DataLoader(
        exp.dataset_train,
        batch_size=exp.flags.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    print('training epochs progress:')
    epoch = 0
    for epoch in tqdm.trange(exp.flags.start_epoch, exp.flags.end_epoch, desc="Epochs", file=sys.stdout):
        # one epoch of training and testing
        train(epoch, exp, train_data_loader, tb_logger);
        test(epoch, exp, train_data_loader, test_data_loader, tb_logger);
        if (epoch+1) % exp.flags.checkpoint_frequency == 0 or (epoch + 1) == exp.flags.end_epoch:
            dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4));
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch);
            torch.save(exp.mm_vae.state_dict(), os.path.join(dir_network_epoch, exp.flags.checkpoint_model_name))

    tb_logger.close()

    dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4));
    if not os.path.exists(dir_network_epoch):
        os.makedirs(dir_network_epoch)
    torch.save(exp.mm_vae.state_dict(), os.path.join(dir_network_epoch, exp.flags.checkpoint_model_name))
