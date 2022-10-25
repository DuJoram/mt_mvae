import sys
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import chain, combinations

subsampling_values = [False]
beta_values = [1.0, 2.5, 5.0]
gamma_values = [0.1, 0.25, 0.5, 1.0, 2.5]
# gamma_n_values = [10.0]
# gamma_y_values = [0.1, 1.0, 10.0, 100.0]
modalities = ['m0', 'm1', 'm2']

l_subsets = chain.from_iterable(combinations(modalities, n) for n in
                                range(len(modalities)+1))
subset_names = []
for _, mods_s in enumerate(l_subsets):
    s_name = '_'.join(sorted(list(mods_s)))
    if s_name == '':
        continue
    subset_names.append(s_name)

values1 = beta_values
value1_name = 'beta'


def analyze_lr_modalities(dir_experiments, keyword, loss_type):
    if loss_type == 'weighted':
        prefix_res = 'w_'
    elif loss_type == 'unweighted':
        prefix_res = 'unw_'
    if keyword is not None:
        fn_results = os.path.join(dir_experiments,
                                  prefix_res + 'lr_modalities_' + keyword + '.csv')
        fn_fig = os.path.join(dir_experiments,
                              prefix_res + 'lr_modalities_' + keyword + '.png')
        str_suptitle = 'Latent Representation ' + keyword
    else:
        fn_results = os.path.join(dir_experiments,
                                  prefix_res + 'lr_modalities.csv')
        fn_fig = os.path.join(dir_experiments,
                              prefix_res + 'lr_modalities.png')
        str_suptitle = 'Latent Representation'

    df = pd.read_csv(fn_results)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    for j, s_key in enumerate(subset_names):
        df_s_in = df.loc[df['subset_in'] == s_key]
        df_s_in = df_s_in.sort_values(by=['beta'])
        subset_values = df_s_in['value'].values
        ax.plot(subset_values, label=s_key)
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(np.arange(0, len(subset_values)))
    ax.set_xticklabels(df_s_in['beta'].values)
    ax.set_xlabel('beta')
    ax.set_ylabel('classification accuracy')
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.suptitle(str_suptitle)
    plt.savefig(fn_fig, format='png')


def analyze_conditional_generation(dir_experiments, keywords, loss_type):
    if loss_type == 'weighted':
        prefix_res = 'w_'
    elif loss_type == 'unweighted':
        prefix_res = 'unw_'
    fn_results = os.path.join(dir_experiments,
                              prefix_res + 'gen_coherence_conditional.csv')
    df = pd.read_csv(fn_results)
    for keyword in keywords:
        if keyword is not None:
            str_suptitle = keyword
            str_fn = '_' + keyword
        else:
            str_suptitle = ''
            str_fn = ''
        fn_fig = os.path.join(dir_experiments,
                              prefix_res + 'cond_gen' + str_fn + '.png')
        fig = plt.figure(figsize=(8, 10))
        for k, mod in enumerate(modalities):
            if keyword is not None:
                mod_key = mod + '_' + keyword
            else:
                mod_key = mod
            ax = fig.add_subplot(len(modalities) + 1, 1, k+1)
            df_mod = df.loc[df['modality_out'] == mod_key]
            for j, s_key in enumerate(subset_names):
                df_s = df_mod.loc[df_mod['subset_in'] == s_key]
                df_s = df_s.sort_values(by=['beta'])
                s_values = df_s['value'].values
                ax.plot(s_values, label=s_key)
            ax.set_ylim([0.0, 1.0])
            ax.set_title(mod)
            ax.set_xticks(np.arange(0, len(df_s['beta'].values)))
            ax.set_xticklabels(df_s['beta'].values)
            ax.set_xlabel('beta')
            ax.legend(bbox_to_anchor=(1.0, 1.0))

        fn_results = os.path.join(dir_experiments, 'gen_coherence_random.csv')
        df_r = pd.read_csv(fn_results)
        df_r = df_r.sort_values(by=['beta'])
        ax = fig.add_subplot(len(modalities) + 1, 1, len(modalities) + 1)
        r_values = df_r['value'].values
        ax.plot(r_values, label='random')
        ax.set_ylim([0.0, 1.0])
        ax.set_title('random')
        ax.set_xticks(np.arange(0, len(df_r['beta'].values)))
        ax.set_xticklabels(df_r['beta'].values)
        ax.set_xlabel('beta')
        ax.legend(bbox_to_anchor=(1.0, 1.0))

        plt.suptitle(str_suptitle)
        fig.tight_layout()
        plt.savefig(fn_fig, format='png')


def analyze_likelihoods(dir_experiments, loss_type):
    if loss_type == 'weighted':
        prefix_res = 'w_'
    elif loss_type == 'unweighted':
        prefix_res = 'unw_'
    fn_results = os.path.join(dir_experiments, prefix_res + 'nll.csv')
    df = pd.read_csv(fn_results)
    out_sets = modalities.copy()
    out_sets.append('joint')
    for keyword in keywords:
        if keyword is not None:
            str_suptitle = keyword
            str_fn = '_' + keyword
        else:
            str_suptitle = ''
            str_fn = ''
        fn_fig = os.path.join(dir_experiments,
                              'lhood' + str_fn + '.png')
        fig = plt.figure(figsize=(8, 10))
        for k, out_key in enumerate(out_sets):
            if keyword is not None:
                mod_key = out_key + '_' + keyword
            else:
                mod_key = out_key
            ax = fig.add_subplot(len(out_sets), 1, k+1)
            df_mod = df.loc[df['out'] == mod_key]
            for j, s_key in enumerate(subset_names):
                df_s = df_mod.loc[df_mod['subset_in'] == s_key]
                df_s = df_s.sort_values(by=['beta'])
                s_values = df_s['value'].values
                ax.plot(s_values, label=s_key)
            ax.set_title(out_key)
            ax.set_xticks(np.arange(0, len(df_s['beta'].values)))
            ax.set_xticklabels(df_s['beta'].values)
            ax.set_xlabel('beta')
            ax.legend(bbox_to_anchor=(1.0, 1.0))
        plt.suptitle(str_suptitle)
        fig.tight_layout()
        plt.savefig(fn_fig, format='png')


def analyze_fids(dir_experiments, loss_type):
    if loss_type == 'weighted':
        prefix_res = 'w_'
    elif loss_type == 'unweighted':
        prefix_res = 'unw_'

    fn_results = os.path.join(dir_experiments, prefix_res + 'gen_fid.csv')
    df = pd.read_csv(fn_results)
    in_sets = subset_names.copy()
    in_sets.append('random')
    for keyword in keywords:
        if keyword is not None:
            str_suptitle = keyword
            str_fn = '_' + keyword
        else:
            str_suptitle = ''
            str_fn = ''
        fn_fig = os.path.join(dir_experiments,
                              prefix_res + 'fid' + str_fn + '.png')
        fig = plt.figure(figsize=(8, 10))
        for k, m_key in enumerate(modalities):
            if keyword is not None:
                mod_key = m_key + '_' + keyword
            else:
                mod_key = m_key
            ax = fig.add_subplot(len(modalities), 1, k+1)
            df_mod = df.loc[df['modality_out'] == mod_key]
            for j, s_key in enumerate(in_sets):
                df_s = df_mod.loc[df_mod['subset_in'] == s_key]
                df_s = df_s.sort_values(by=['beta'])
                s_values = df_s['value'].values
                ax.plot(s_values, label=s_key)
#             ax.set_ylim([-5000, 0.0])
            ax.set_title(m_key)
            ax.set_xticks(np.arange(0, len(df_s['beta'].values)))
            ax.set_xticklabels(df_s['beta'].values)
            ax.set_xlabel('beta')
            ax.legend(bbox_to_anchor=(1.0, 1.0))
        # handles, labels = plt.gca().get_legend_handles_labels()
        # fig.legend(handles, labels, bbox_to_anchor=(1.0,1.0))
        plt.suptitle(str_suptitle)
        fig.tight_layout()
        plt.savefig(fn_fig, format='png')


def analyze_losses(dir_experiments, loss_type):
    if loss_type == 'weighted':
        prefix_res = 'w_'
    elif loss_type == 'unweighted':
        prefix_res = 'unw_'

    fn_results = os.path.join(dir_experiments, prefix_res + 'test_loss.csv')
    df = pd.read_csv(fn_results)
    beta_values = sorted(list(set(df['beta'])))
    df_s = df.sort_values(by=['beta'])

    fn_fig = os.path.join(dir_experiments,
                          prefix_res + 'test_loss.png')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(df_s['value'].values)
    ax.set_xticks(np.arange(0, len(beta_values)))
    ax.set_xticklabels(beta_values)
    ax.set_xlabel('beta')
    ax.set_ylabel('test loss')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(df_s['step'].values)
    ax2.set_xticks(np.arange(0, len(beta_values)))
    ax2.set_xticklabels(beta_values)
    ax2.set_xlabel('beta')
    ax2.set_ylabel('step')
    fig.tight_layout()
    plt.savefig(fn_fig, format='png')


def analyze_losses_all(dir_experiments):
    fn_loss_w = os.path.join(dir_experiments, 'test_loss_all_w.csv')
    fn_loss_unw = os.path.join(dir_experiments, 'test_loss_all_unw.csv')
    df_loss_w = pd.read_csv(fn_loss_w)
    df_loss_unw = pd.read_csv(fn_loss_unw)
    df_loss_w = df_loss_w.sort_values(by=['beta'])
    df_loss_unw = df_loss_unw.sort_values(by=['beta'])
    fn_fig = os.path.join(dir_experiments, 'test_losses_all_steps.png')

    beta_values = list(set(df_loss_w['beta'].values))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 1)
    for beta in beta_values:
        df_loss_w_beta = df_loss_w.loc[df_loss_w['beta'] == beta]
        df_loss_w_beta = df_loss_w_beta.sort_values(by=['step'])
        ax.plot(df_loss_w_beta['value'].values, label='beta=' + str(beta),
                alpha=0.5)
    ax.set_xlabel('step')
    ax.set_ylabel('loss weighted')
    ax.legend(bbox_to_anchor=(1.0, 1.0))

    ax2 = fig.add_subplot(2, 1, 2)
    for beta in beta_values:
        df_loss_unw_beta = df_loss_unw.loc[df_loss_unw['beta'] == beta]
        df_loss_unw_beta = df_loss_unw_beta.sort_values(by=['step'])
        ax2.plot(df_loss_unw_beta['value'].values, label='beta=' + str(beta),
                 alpha=0.5)
    ax2.set_xlabel('step')
    ax2.set_ylabel('loss un-weighted')
    ax2.legend(bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    plt.savefig(fn_fig, format='png')


if __name__ == '__main__':
    # dir_base = '/usr/scratch/fusessh/ethsec_experiments'
    # str_exp = 'mm_sets/PolyMNIST/beta_gamma_hierarchical_double_sm'
    # dir_runs = os.path.join(dir_base, str_exp)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-experiments", type=str, required=True)
    # parser.add_argument("--search-string-regex", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="vanilla",
                        help="dataset name/type: vanilla or ext")
    parser.add_argument("--loss-type", type=str, default="weighted",
                        help="""model selection based on loss type
                        weighted/unweighted regularizer term""")
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)
    dir_exp = args.dir_experiments
    dataset_name = args.dataset

    if dataset_name == 'ext':
        analyze_lr_modalities(dir_exp, 'main', args.loss_type)
        analyze_lr_modalities(dir_exp,  'c1', args.loss_type)
        analyze_lr_modalities(dir_exp, 'c2', args.loss_type)
    else:
        analyze_lr_modalities(dir_exp, None, args.loss_type)

    # analyze_lr_subspaces(dir_runs)
    if dataset_name == 'ext':
        keywords = ['main', 'c1', 'c2']
    else:
        keywords = [None]
    analyze_conditional_generation(dir_exp, keywords, args.loss_type)
    analyze_likelihoods(dir_exp, args.loss_type)
    analyze_fids(dir_exp, args.loss_type)
    analyze_losses(dir_exp, args.loss_type)
    analyze_losses_all(dir_exp)

