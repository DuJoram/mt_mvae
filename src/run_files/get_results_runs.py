
import sys
import os
import glob
import argparse
from filter_tensorboard_logs import filter_tensorboard_logs
from filter_tensorboard_logs import filter_logs_lr
from filter_tensorboard_logs import filter_logs_fid
from filter_tensorboard_logs import filter_logs_likelihood
from filter_tensorboard_logs import filter_logs_coherence
from filter_tensorboard_logs import get_flags

import torch
import pandas as pd


def read_logs():
    pass


def get_eval_step(d_str, unw_loss=False):
    # we analyze all trained models at the point in training where
    # they reach the lowest test score.
    # because we mostly evaluate only every x epochs (latent representation, etc)
    # we take the next evaluation cycle for every model
    if unw_loss:
        str_filt_loss = 'test/Loss_unweighted'
    else:
        str_filt_loss = 'test/Loss'

    df_test_loss = filter_tensorboard_logs(d_str,
                                           str_filt_loss,
                                           None, True)
    min_idx_value = df_test_loss['value'].idxmin()
    min_value = df_test_loss['value'].min()
    df_min = df_test_loss.iloc[min_idx_value]
    min_value_step = df_min['step']
    df_idx_sel = filter_tensorboard_logs(d_str,
                                         'Latent Representation',
                                         None, True)
    try:
        test_step_idx = df_idx_sel[df_idx_sel['step'] >
                                   min_value_step]['step'].idxmin()
    except ValueError:
        test_step_idx = df_idx_sel[df_idx_sel['step'] <
                                   min_value_step]['step'].idxmax()
    df_min_lr = df_idx_sel.iloc[test_step_idx]
    eval_step = int(df_min_lr['step'])
    return eval_step, min_value


def get_eval_step_all_models(dir_runs, loss_unweighted=False):
    eval_steps = {}
    beta_values = []
    test_loss_values = []
    for d in dir_runs:
        logdir = os.path.join(d, 'logs')
        flags = torch.load(os.path.join(d, 'flags.rar'))
        beta_value = flags.beta
        eval_step_run, eval_test_loss = get_eval_step(logdir,
                                                      unw_loss=loss_unweighted)
        eval_steps[d] = eval_step_run
        beta_values.append(beta_value)
        test_loss_values.append(eval_test_loss)
    df_test_loss = pd.DataFrame()
    df_test_loss['beta'] = beta_values
    df_test_loss['value'] = test_loss_values
    df_test_loss['step'] = list(eval_steps.values())
    return eval_steps, df_test_loss


def get_results_lr_modalities(dirs, keyword, eval_steps=None):
    l_dfs = []
    for d in dirs:
        logdir = os.path.join(d, 'logs')
        flags = torch.load(os.path.join(d, 'flags.rar'))
        beta_value = flags.beta

        if eval_steps is None:
            # if the point in training time is not defined
            # we take the last possible point
            eval_step_run = -1
        else:
            eval_step_run = eval_steps[d]
        # Latent representation modalities
        df_lr_mods = filter_tensorboard_logs(logdir,
                                             'Latent Representation',
                                             eval_step_run,
                                             True)
        if keyword is not None:
            df_hp_m = filter_logs_lr(df_lr_mods, 'modalities,' + keyword)
        else:
            # df_hp_m = filter_logs_lr(df_lr_mods, 'digit')
            df_hp_m = filter_logs_lr(df_lr_mods)

        num_values = df_hp_m.shape[0]
        l_beta = [beta_value]*num_values
        df_hp_m['beta'] = l_beta
        l_dfs.append(df_hp_m)
    df_all = pd.concat(l_dfs)
    df_all = df_all.reset_index()
    df_all = df_all.drop(columns=['index'])
    return df_all


def get_results_coherence(dirs, eval_steps=None):
    l_dfs_random = []
    l_dfs_cond = []
    for d in dirs:
        logdir = os.path.join(d, 'logs')
        flags = torch.load(os.path.join(d, 'flags.rar'))
        beta_value = flags.beta

        if eval_steps is None:
            # if the point in training time is not defined
            # we take the last possible point
            eval_step_run = -1
        else:
            eval_step_run = eval_steps[d]
        # generation coherence
        df_gen = filter_tensorboard_logs(logdir,
                                         'Generation',
                                         eval_step_run,
                                         True)
        # df_cond, df_random = filter_logs_coherence(df_gen, 3)
        df_cond, df_random = filter_logs_coherence(df_gen)
        num_values = df_cond.shape[0]
        l_beta = [beta_value]*num_values
        df_cond['beta'] = l_beta
        num_values = df_random.shape[0]
        l_beta = [beta_value]*num_values
        df_random['beta'] = l_beta
        l_dfs_cond.append(df_cond)
        l_dfs_random.append(df_random)
    df_all_random = pd.concat(l_dfs_random)
    df_all_random = df_all_random.reset_index()
    df_all_random = df_all_random.drop(columns=['index'])

    df_all_cond = pd.concat(l_dfs_cond)
    df_all_cond = df_all_cond.reset_index()
    df_all_cond = df_all_cond.drop(columns=['index'])
    print(df_all_cond)
    return [df_all_cond, df_all_random]


def get_results_fid(dirs, eval_steps=None):
    l_dfs_fid = []
    for d in dirs:
        logdir = os.path.join(d, 'logs')
        flags = torch.load(os.path.join(d, 'flags.rar'))
        beta_value = flags.beta

        if eval_steps is None:
            # if the point in training time is not defined
            # we take the last possible point
            eval_step_run = -1
        else:
            eval_step_run = eval_steps[d]
        # generation quality of samples
        df_fid = filter_tensorboard_logs(logdir,
                                         'PRD',
                                         eval_step_run,
                                         True)
        df_fid = filter_logs_fid(df_fid)
        num_values = df_fid.shape[0]
        l_beta = [beta_value]*num_values
        df_fid['beta'] = l_beta
        l_dfs_fid.append(df_fid)
    df_all_fid = pd.concat(l_dfs_fid)
    df_all_fid = df_all_fid.reset_index()
    df_all_fid = df_all_fid.drop(columns=['index'])
    return df_all_fid


def get_results_likelihoods(dirs, eval_steps=None):
    l_dfs_nll = []
    for d in dirs:
        logdir = os.path.join(d, 'logs')
        flags = torch.load(os.path.join(d, 'flags.rar'))
        beta_value = flags.beta

        if eval_steps is None:
            # if the point in training time is not defined
            # we take the last possible point
            eval_step_run = -1
        else:
            eval_step_run = eval_steps[d]
        # generation coherence
        df_lhood = filter_tensorboard_logs(logdir,
                                           'Likelihood',
                                           eval_step_run,
                                           True)
        df_nll = filter_logs_likelihood(df_lhood)
        num_values = df_nll.shape[0]
        l_beta = [beta_value]*num_values
        df_nll['beta'] = l_beta
        l_dfs_nll.append(df_nll)
    df_all_nll = pd.concat(l_dfs_nll)
    df_all_nll = df_all_nll.reset_index()
    df_all_nll = df_all_nll.drop(columns=['index'])
    return df_all_nll


def get_results_losses_all(dirs):
    l_dfs_loss = []
    l_dfs_loss_unw = []
    for d in dirs:
        logdir = os.path.join(d, 'logs')
        flags = torch.load(os.path.join(d, 'flags.rar'))
        beta_value = flags.beta

        df_loss_scaled = filter_tensorboard_logs(logdir,
                                                 'test/Loss',
                                                 None, True)
        num_values = df_loss_scaled.shape[0]
        l_beta = [beta_value]*num_values
        df_loss_scaled['beta'] = l_beta
        l_dfs_loss.append(df_loss_scaled)

        df_loss_unscaled = filter_tensorboard_logs(logdir,
                                                   'test/Loss_unweighted',
                                                   None, True)
        num_values = df_loss_unscaled.shape[0]
        l_beta = [beta_value]*num_values
        df_loss_unscaled['beta'] = l_beta
        l_dfs_loss_unw.append(df_loss_unscaled)
    df_all_loss = pd.concat(l_dfs_loss)
    df_all_loss = df_all_loss.reset_index()
    df_all_loss = df_all_loss.drop(columns=['index'])
    df_all_loss_unw = pd.concat(l_dfs_loss_unw)
    df_all_loss_unw = df_all_loss_unw.reset_index()
    df_all_loss_unw = df_all_loss_unw.drop(columns=['index'])
    return [df_all_loss, df_all_loss_unw]


def get_results(dir_experiments, prefix='', dataset='vanilla',
                loss_type='weighted'):
    dirs = glob.glob(os.path.join(dir_experiments, prefix + '*/'))

    if loss_type == 'weighted':
        pre_results = 'w_'
        unw_loss = False
    elif loss_type == 'unweighted':
        pre_results = 'unw_'
        unw_loss = True
    else:
        print('unknown loss type...exit!')
        sys.exit()
    eval_steps, df_test_loss = get_eval_step_all_models(dirs,
                                                        loss_unweighted=unw_loss)
    fn_test_loss = os.path.join(dir_experiments,
                                pre_results + 'test_loss.csv')
    df_test_loss.to_csv(fn_test_loss, index=False)

    df_loss_w, df_loss_unw = get_results_losses_all(dirs)
    fn_loss_w = os.path.join(dir_experiments, 'test_loss_all_w.csv')
    fn_loss_unw = os.path.join(dir_experiments, 'test_loss_all_unw.csv')
    df_loss_w.to_csv(fn_loss_w, index=False)
    df_loss_unw.to_csv(fn_loss_unw, index=False)
    
    if dataset == 'ext':
        df_lr_mods = get_results_lr_modalities(dirs, 'main', eval_steps)
        fn_lr_mods_out = os.path.join(dir_experiments,
                                      pre_results + 'lr_modalities_main.csv')
        df_lr_mods.to_csv(fn_lr_mods_out, index=False)
        df_lr_mods = get_results_lr_modalities(dirs, 'c1', eval_steps)
        fn_lr_mods_out = os.path.join(dir_experiments, pre_results + 'lr_modalities_c1.csv')
        df_lr_mods.to_csv(fn_lr_mods_out, index=False)
        df_lr_mods = get_results_lr_modalities(dirs, 'c2', eval_steps)
        fn_lr_mods_out = os.path.join(dir_experiments, pre_results + 'lr_modalities_c2.csv')
        df_lr_mods.to_csv(fn_lr_mods_out, index=False)
    else:
        df_lr_mods = get_results_lr_modalities(dirs, None, eval_steps)
        fn_lr_mods_out = os.path.join(dir_experiments,
                                      pre_results + 'lr_modalities.csv')
        df_lr_mods.to_csv(fn_lr_mods_out, index=False)

    dfs_coherence = get_results_coherence(dirs, eval_steps)
    df_coherence_cond, df_coherence_random = dfs_coherence
    fn_coherence_cond_out = os.path.join(dir_experiments,
                                         pre_results + 'gen_coherence_conditional.csv')
    df_coherence_cond.to_csv(fn_coherence_cond_out, index=False)
    fn_coherence_random_out = os.path.join(dir_experiments,
                                           pre_results + 'gen_coherence_random.csv')
    df_coherence_random.to_csv(fn_coherence_random_out, index=False)

    df_fid = get_results_fid(dirs, eval_steps)
    fn_fid_out = os.path.join(dir_experiments, pre_results + 'gen_fid.csv')
    df_fid.to_csv(fn_fid_out, index=False)

    df_nll = get_results_likelihoods(dirs, eval_steps)
    fn_nll_out = os.path.join(dir_experiments, pre_results + 'nll.csv')
    df_nll.to_csv(fn_nll_out, index=False)


if __name__ == '__main__':
    # dir_base = '/usr/scratch/fusessh/ethsec_experiments'
    # str_exp = 'mm_sets/PolyMNIST/beta_gamma_hierarchical_double_sm'
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
    prefix_exp = 'MMNIST'
    # dir_runs = os.path.join(dir_base, str_exp)
    get_results(dir_exp, prefix=prefix_exp, dataset=dataset_name,
                loss_type=args.loss_type)
