
import numpy as np
import math

from torch.utils.data import DataLoader

from utils.likelihood import get_latent_samples
from utils.likelihood import log_marginal_estimate
from utils.likelihood import log_joint_estimate

LOG2PI = float(np.log(2.0 * math.pi))



#at the moment: only marginals and joint
def calc_log_likelihood_batch(exp, latents, subset_key, subset, batch, num_imp_samples=10):
    flags = exp.flags;
    model = exp.mm_vae;
    mod_weights = exp.style_weights;
    modalities = exp.modalities;
    batch_size = flags.batch_size
    factorized_representation = flags.factorized_representation

    s_dist = latents['subsets'][subset_key]
    n_total_samples = s_dist[0].shape[0]*num_imp_samples;

    if factorized_representation:
        enc_mods = latents['modalities'];
        style = model.get_random_style_dists(batch_size);
        for modality in subset:
            if (enc_mods[modality.name + '_style'][0] is not None
                and enc_mods[modality.name + '_style'][1] is not None):
                style[modality.name] = enc_mods[modality.name + '_style'];
    else:
        style = None;

    l_subset = {'content': s_dist, 'style': style};
    l = get_latent_samples(flags, l_subset, num_imp_samples, modalities);

    l_style_rep = l['style'];
    l_content_rep = l['content'];

    c = {'mu': l_content_rep['mu'].view(n_total_samples, -1),
         'logvar': l_content_rep['logvar'].view(n_total_samples, -1),
         'z': l_content_rep['z'].view(n_total_samples, -1)}
    l_lin_rep = {'content': c,
                 'style': dict()};
    for m, m_key in enumerate(l_style_rep.keys()):
        if factorized_representation:
            s = {'mu': l_style_rep[m_key]['mu'].view(n_total_samples, -1),
                 'logvar': l_style_rep[m_key]['logvar'].view(n_total_samples, -1),
                 'z': l_style_rep[m_key]['z'].view(n_total_samples, -1)}
            l_lin_rep['style'][m_key] = s;
        else:
            l_lin_rep['style'][m_key] = None;

    l_dec = {'content': l_lin_rep['content']['z'],
             'style': dict()};
    for m, m_key in enumerate(l_style_rep.keys()):
        if factorized_representation:
            l_dec['style'][m_key] = l_lin_rep['style'][m_key]['z'];
        else:
            l_dec['style'][m_key] = None;

    gen = model.generate_sufficient_statistics_from_latents(l_dec);

    l_lin_rep_style = l_lin_rep['style'];
    l_lin_rep_content = l_lin_rep['content'];
    ll = dict();
    for modality in modalities:
        # compute marginal log-likelihood
        if modality in subset:
            style_mod = l_lin_rep_style[modality.name];
        else:
            style_mod = None;
        ll_mod = log_marginal_estimate(flags,
                                       num_imp_samples,
                                       gen[modality.name],
                                       batch[modality.name],
                                       style_mod,
                                       l_lin_rep_content)
        ll[modality.name] = ll_mod;

    ll_joint = log_joint_estimate(flags, num_imp_samples, gen, batch,
                                  l_lin_rep_style,
                                  l_lin_rep_content);
    ll['joint'] = ll_joint;
    return ll;


def estimate_likelihoods(exp):
    model = exp.mm_vae
    modalities = exp.modalities
    bs_normal = exp.flags.batch_size
    device = exp.flags.device
    dataset_test = exp.dataset_test
    subsets = exp.subsets


    # FIXME: Use local variable instead
    exp.flags.batch_size = 64;
    d_loader = DataLoader(dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True, pin_memory=True);

    likelihoods = dict()
    for subset_key in subsets:
        if subset_key != '':
            likelihoods[subset_key] = dict();
            for modality in modalities:
                likelihoods[subset_key][modality.name] = [];
            likelihoods[subset_key]['joint'] = [];

    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        for modality in modalities:
            batch_d[modality.name] = batch_d[modality.name].to(device);

        latents = model.inference(batch_d);
        for subset_key, subset in subsets.items():
            if subset_key != '':
                ll_batch = calc_log_likelihood_batch(exp, latents,
                                                     subset_key, subset,
                                                     batch_d,
                                                     num_imp_samples=12)
                for modality_name, log_likelihood_batch in ll_batch.items():
                    likelihoods[subset_key][modality_name].append(log_likelihood_batch.item());

    for subset_key, log_likelihoods_subset in likelihoods.items():
        for modality, log_likelihood_modality in log_likelihoods_subset.items():
            mean_val = np.mean(np.array(log_likelihood_modality))
            likelihoods[subset_key][modality] = mean_val;
    exp.flags.batch_size = bs_normal;
    return likelihoods;



