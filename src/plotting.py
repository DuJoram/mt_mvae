import os

import torch

from utils import utils, plot


def generate_plots(exp, epoch):
    factorized_representation = exp.flags.factorized_representation
    modalities = exp.modalities
    plots = dict();
    if exp.flags.factorized_representation:
        # mnist to mnist: swapping content and style intra modal
        swapping_figs = generate_swapping_plot(exp, epoch)
        plots['swapping'] = swapping_figs;

    for k in range(len(modalities)):
        cond_k = generate_conditional_fig_M(exp, epoch, k+1)
        plots['cond_gen_' + str(k+1).zfill(2)] = cond_k;

    plots['random'] = generate_random_samples_plots(exp, epoch);
    return plots;


def generate_random_samples_plots(exp, epoch):
    model = exp.mm_vae
    modalities = exp.modalities
    plot_image_size = exp.plot_image_size
    dir_random_samples = exp.flags.dir_random_samples
    save_figure = exp.flags.save_figure

    num_samples = 100
    random_samples = model.generate(num_samples)
    random_plots = dict();
    for modality in modalities:
        samples_mod = random_samples[modality.name];
        rec = torch.zeros(plot_image_size,
                          dtype=torch.float32).repeat(num_samples,1,1,1);
        for l in range(0, num_samples):
            rand_plot = modality.plot_data(samples_mod[l]);
            rec[l, :, :, :] = rand_plot;
        random_plots[modality.name] = rec;

    for modality in modalities:
        fn = os.path.join(dir_random_samples, 'random_epoch_' +
                             str(epoch).zfill(4) + '_' + modality.name + '.png');
        mod_plot = random_plots[modality.name]
        p = plot.create_fig(fn, mod_plot, 10, save_figure=save_figure);
        random_plots[modality.name] = p;
    return random_plots;


def generate_swapping_plot(exp, epoch):
    model = exp.mm_vae;
    modalities = exp.modalities;
    plot_image_size = exp.plot_image_size
    samples = exp.test_samples;
    device = exp.flags.device
    dir_swapping = exp.flags.dir_swapping
    save_figure = exp.flags.save_figure

    swap_plots = dict();
    for modality_in in modalities:
        for modality_out in modalities:
            rec = torch.zeros(plot_image_size,
                              dtype=torch.float32, device=device).repeat(121,1,1,1);
            for i in range(len(samples)):
                c_sample_in = modality_in.plot_data(samples[i][modality_in.name]);
                s_sample_out = modality_out.plot_data(samples[i][modality_out.name]);
                rec[i+1, :, :, :] = c_sample_in;
                rec[(i + 1) * 11, :, :, :] = s_sample_out;
            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch_s = {modality_out.name: samples[i][modality_out.name].unsqueeze(0)}
                    i_batch_c = {modality_in.name: samples[i][modality_in.name].unsqueeze(0)}
                    l_style = model.inference(i_batch_s,
                                              num_samples=1)
                    l_content = model.inference(i_batch_c,
                                                num_samples=1)
                    l_s_mod = l_style['modalities'][modality_out.name + '_style'];
                    l_c_mod = l_content['modalities'][modality_in.name];
                    s_emb = utils.reparameterize(l_s_mod[0], l_s_mod[1]);
                    c_emb = utils.reparameterize(l_c_mod[0], l_c_mod[1]);
                    style_emb = {modality_out.name: s_emb}
                    emb_swap = {'content': c_emb, 'style': style_emb};
                    swap_sample = model.generate_from_latents(emb_swap);
                    swap_out = modality_out.plot_data(swap_sample[modality_out.name].squeeze(0));
                    rec[(i+1) * 11 + (j+1), :, :, :] = swap_out;
                    fn_comb = (modality_in.name + '_to_' + modality_out.name + '_epoch_'
                               + str(epoch).zfill(4) + '.png');
                    fn = os.path.join(dir_swapping, fn_comb);
                    swap_plot = plot.create_fig(fn, rec, 11, save_figure=save_figure);
                    swap_plots[modality_in.name + '_' + modality_out.name] = swap_plot;
    return swap_plots;


def generate_conditional_fig_M(exp, epoch, M):
    model = exp.mm_vae
    modalities = exp.modalities
    samples = exp.test_samples
    subsets = exp.subsets
    plot_image_size = exp.plot_image_size
    factorized_representation = exp.flags.factorized_representation
    dir_cond_gen = exp.flags.dir_cond_gen
    save_figure = exp.flags.save_figure

    # get style from random sampling
    random_styles = model.get_random_styles(10);

    cond_plots = dict();
    for subset_key, subset in subsets.items():
        num_mods_subset = len(subset);

        if num_mods_subset == M:
            for modality_out in modalities:
                rec = torch.zeros(plot_image_size, dtype=torch.float32).repeat(100 + M*10, 1, 1, 1)
                for m, sample in enumerate(samples):
                    for n, modality_in in enumerate(subset):
                        c_in = modality_in.plot_data(sample[modality_in.name])
                        rec[m + n*10, :, :, :] = c_in
                cond_plots[subset_key + '__' + modality_out.name] = rec

            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch = dict();
                    for modality in subset:
                        i_batch[modality.name] = samples[j][modality.name].unsqueeze(0);
                    latents = model.inference(i_batch, num_samples=1)
                    c_in = latents['subsets'][subset_key];
                    c_rep = utils.reparameterize(mu=c_in[0], logvar=c_in[1]);

                    style = dict();
                    for modality_out in modalities:
                        if factorized_representation:
                            style[modality_out.name] = random_styles[modality_out.name][i].unsqueeze(0);
                        else:
                            style[modality_out.name] = None;
                    cond_mod_in = {'content': c_rep, 'style': style};
                    cond_gen_samples = model.generate_from_latents(cond_mod_in);

                    for modality_out in modalities:
                        rec = cond_plots[subset_key + '__' + modality_out.name];
                        squeezed = cond_gen_samples[modality_out.name].squeeze(0);
                        p_out = modality_out.plot_data(squeezed);
                        rec[(i+M) * 10 + j, :, :, :] = p_out;
                        cond_plots[subset_key + '__' + modality_out.name] = rec;

    for subset_key_in, subset_in in subsets.items():
        if len(subset_in) == M:
            for modality_out in modalities:
                rec = cond_plots[subset_key_in + '__' + modality_out.name];
                fn_comb = (subset_key_in + '_to_' + modality_out.name + '_epoch_' +
                           str(epoch).zfill(4) + '.png')
                fn_out = os.path.join(dir_cond_gen, fn_comb);
                plot_out = plot.create_fig(fn_out, rec, 10, save_figure=save_figure);
                cond_plots[subset_key_in + '__' + modality_out.name] = plot_out;
    return cond_plots;

