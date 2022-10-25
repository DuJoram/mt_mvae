import os

import torch
import torch.nn as nn

from utils import utils
from models.multimodal_vae import MultimodalVAE


class VAEMMNIST(MultimodalVAE):
    def __init__(self, flags, modalities, subsets):
        super(VAEMMNIST, self).__init__(flags, modalities, subsets)
        # self.num_modalities = len(modalities.keys())
        # self.flags = flags
        # self.modalities = modalities
        # self.subsets = subsets
        print('num modalities: ' + str(self.num_modalities))
        self.encoders = nn.ModuleList([modalities["m%d" % m].encoder.to(flags.device) for m in range(self.num_modalities)])
        self.decoders = nn.ModuleList([modalities["m%d" % m].decoder.to(flags.device) for m in range(self.num_modalities)])
        self.likelihoods = [modalities["m%d" % m].likelihood for m in range(self.num_modalities)]

    # def forward(self, input_batch):
    #     latents = self.inference(input_batch)
    #
    #     results = dict()
    #     results['latents'] = latents
    #
    #     results['group_distr'] = latents['joint']
    #     class_embeddings = utils.reparameterize(latents['joint'][0],
    #                                             latents['joint'][1])
    #     div = self.calc_joint_divergence(latents['mus'],
    #                                      latents['logvars'],
    #                                      latents['weights'])
    #     for k, key in enumerate(div.keys()):
    #         results[key] = div[key]
    #
    #     enc_mods = latents['modalities']
    #     results_rec = dict()
    #     for m in range(self.num_modalities):
    #         if "m%d" % m in input_batch.keys():
    #             x_m = input_batch['m%d' % m]
    #             if x_m is not None:
    #                 style_mu, style_logvar = enc_mods['m%d_style' % m]
    #                 if self.flags.factorized_representation:
    #                     style_embeddings = utils.reparameterize(mu=style_mu, logvar=style_logvar)
    #                 else:
    #                     style_embeddings = None
    #                 rec = self.likelihoods[m](*self.decoders[m](style_embeddings, class_embeddings))
    #                 results_rec['m%d' % m] = rec
    #     results['rec'] = results_rec
    #     return results

    # def encode(self, input_batch):
    #     # latents = dict()
    #     enc_mods = dict()
    #     for m in range(self.num_modalities):
    #         if "m%d" % m in input_batch.keys():
    #             x_m = input_batch['m%d' % m]
    #         else:
    #             x_m = None
    #         latents_style, latents_class = self.encode_m(x_m, m)
    #         enc_mods["m%d" % m] = latents_class
    #         enc_mods["m%d_style" % m] = latents_style
    #     # latents['modalities'] = enc_mods
    #     return enc_mods

    # def encode_m(self, x_m, m):
    #     if x_m is not None:
    #         latents = self.encoders[m](x_m)
    #         latents_style = latents[:2]
    #         latents_class = latents[2:]
    #     else:
    #         latents_style = [None, None]
    #         latents_class = [None, None]
    #     return latents_style, latents_class

    def get_random_styles(self, num_samples):
        styles = dict()
        for m in range(self.num_modalities):
            if self.flags.factorized_representation:
                z_style_m = torch.randn(num_samples, self.flags.style_dim)
                z_style_m = z_style_m.to(self.flags.device)
            else:
                z_style_m = None
            styles["m%d" % m] = z_style_m
        return styles

    def get_random_style_dists(self, num_samples):
        styles = dict()
        for m in range(self.num_modalities):
            s_mu_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            s_logvar_m = torch.zeros(num_samples, self.flags.style_dim).to(self.flags.device)
            dist_m = [s_mu_m, s_logvar_m]
            styles["m%d" % m] = dist_m
        return styles

    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size
        z_class = torch.randn(num_samples, self.flags.class_dim)
        z_class = z_class.to(self.flags.device)

        style_latents = self.get_random_styles(num_samples)
        random_latents = {'content': z_class, 'style': style_latents}
        random_samples = self.generate_from_latents(random_latents)
        return random_samples

    def generate_from_latents(self, latents):
        cond_gen = {}
        for m in range(self.num_modalities):
            suff_stats = self.generate_sufficient_statistics_from_latents(latents)
            cond_gen_m = suff_stats["m%d" % m].mean
            cond_gen["m%d" % m] = cond_gen_m
        return cond_gen

    def generate_sufficient_statistics_from_latents(self, latents):
        cond_gen = {}
        for m in range(self.num_modalities):
            style_m = latents['style']['m%d' % m]
            content = latents['content']
            cond_gen_m = self.likelihoods[m](*self.decoders[m](style_m, content))
            cond_gen["m%d" % m] = cond_gen_m
        return cond_gen

    def save_networks(self, save_path: str):
        for m in range(self.num_modalities):
            torch.save(self.encoders[m].state_dict(), os.path.join(save_path, "encoderM%d" % m))
            torch.save(self.decoders[m].state_dict(), os.path.join(save_path, "decoderM%d" % m))
