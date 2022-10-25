#!/usr/bin/env python3

import re
import argparse
import pandas as pd
import tensorflow as tf

from pathlib import Path

from itertools import chain, combinations


# settings
pd.set_option('display.max_rows', None)

l_vals = ['Latent Representation', 'Likelihood', 'PRD', 'Generation']
# l_mods = ['m0', 'm1', 'm2', 'm3', 'm4']
# full_set = 'm0_m1_m2_m3_m4'
l_mods = ['m0', 'm1', 'm2']
full_set = 'm0_m1_m2'
l1out = {'m0': 'm1_m2_m3_m4', 'm1': 'm0_m2_m3_m4', 'm2': 'm0_m1_m3_m4', 'm3':
         'm0_m1_m2_m4', 'm4': 'm0_m1_m2_m3'}


def filter_logs_likelihood(df):
    print('')
    print('Likelihood:')
    df_s_in = df['tag'].str.split('/').apply(lambda x: x[-3])
    df_s_out = df['tag'].str.split('/').apply(lambda x: x[-2])
    df_value = df['value']
    df_out = pd.DataFrame()
    df_out['subset_in'] = df_s_in
    df_out['out'] = df_s_out
    df_out['value'] = df_value
    print(df_out)
    return df_out


def filter_logs_coherence(df):
    print('')
    print('Coherence:')
    df_random = df[df['tag'].str.split('/').apply(lambda x:
                                                  x[-3].startswith('Random'))]
    df_value = df_random['value']
    df_in = df_random['tag'].str.split('/').apply(lambda x: x[-3])
    df_r_out = pd.DataFrame()
    df_r_out['value'] = df_value
    df_r_out['subset_in'] = df_in

    df_mod_out = df[df['tag'].str.split('/').apply(lambda x: not
                                                   x[-3].startswith('Random'))]
    df_s_in = df_mod_out['tag'].str.split('/').apply(lambda x: x[-3])
    df_m_out = df_mod_out['tag'].str.split('/').apply(lambda x: x[-2])
    df_value = df_mod_out['value']
    df_out = pd.DataFrame()
    df_out['subset_in'] = df_s_in
    df_out['modality_out'] = df_m_out
    df_out['value'] = df_value
    print(df_out)
    return df_out, df_r_out


def filter_logs_fid(df):
    print('')
    print('FID:')
    # df_mod = df[df['tag'].str.split('/').apply(lambda x:
    #                                            x[-2].endswith(m_key))]
    df_m_out = df['tag'].str.split('/').apply(lambda x:
                                              x[-2].split('_')[-1])
    df_s_in = df['tag'].str.split('/').apply(lambda x:
                                             '_'.join(x[-2].split('_')[:-1]))
    df_value = df['value']
    df_mods_out = pd.DataFrame()
    df_mods_out['subset_in'] = df_s_in
    df_mods_out['modality_out'] = df_m_out
    df_mods_out['value'] = df_value
    print(df_mods_out)
    return df_mods_out


def filter_logs_lr(df):
    print('')
    print('Latent Representation:')
    print(df)
    df_tag = df['tag']
    df_tag_subset = df_tag.str.split('/').apply(lambda x: x[-2])
    df_out = pd.DataFrame()
    df_out['subset_in'] = df_tag_subset
    df_out['value'] = df['value']
    # df_out = df[df['tag'].str.split('/').apply(lambda x: x[-2])]
    return df_out


def filter_tensorboard_logs(logdir, search_string_regex,
                            step=None, silent=True):

    # find all tfevent files recursively in logdir
    tfevent_paths = []
    for path in Path(logdir).rglob("events.out.tfevents*"):
        tfevent_paths.append(path)
     
    # find search_string in filenames and retreive all values
    results = []
    unique_paths_with_matches = set()
    for path in tfevent_paths:
        tag = str(path).replace(logdir, "").replace(path.name, "")
        if re.search(search_string_regex, tag):
            unique_paths_with_matches.add(path)
            for element in tf.compat.v1.train.summary_iterator(str(path)):
                for v in element.summary.value:
                    results.append((tag, v.simple_value, element.step))
    df = pd.DataFrame(results, columns=["tag", "value", "step"]).sort_values(["tag", "step"])
    if not silent:
        print("\nraw: %d values found across %d files" % (len(df), len(unique_paths_with_matches)))
                    
    # filter values by 'step'
    if step is None:
        df_filtered = df
    elif step == -1:
        df_filtered = df.groupby("tag").nth(-1).reset_index()
    else:
        df_filtered = df[df["step"] == step].reset_index()
    if not silent:
        print("filtering by 'step': %d values found across %d files" % (len(df_filtered), len(df_filtered.tag.unique())))
    return df_filtered


def get_flags(logdir):
    # find all tfevent files recursively in logdir
    tfevent_paths = []
    for path in Path(logdir).rglob("events.out.tfevents*"):
        tfevent_paths.append(path)
     
    for path in tfevent_paths:
        for e in tf.compat.v1.train.summary_iterator(str(path)):
            for v in e.summary.value:
                if 'FLAGS' in v.tag:
                    print(v.tag)
                    print(v.tensor)
                    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    # parser.add_argument("--search-string-regex", type=str, required=True)
    parser.add_argument("--step", type=int, default=None, help="by default, take all (None), can also take the last (-1)")
    parser.add_argument("--silent", default=False, action="store_true")
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    get_flags(args.logdir)
    print('start filtering....')
    # execute
    for _, search_str in enumerate(l_vals):
        df_filtered = filter_tensorboard_logs(args.logdir, search_str, args.step, args.silent)
        if search_str == 'Generation':
            filter_logs_coherence(df_filtered)
        elif search_str == 'Likelihood':
            filter_logs_likelihood(df_filtered)
        elif search_str == 'PRD':
            filter_logs_fid(df_filtered)
        elif search_str == 'Latent Representation':
            filter_logs_lr(df_filtered)
    # print(df_filtered)
