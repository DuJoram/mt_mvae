import sys,os
import numpy as np
import pandas as pd

import torch
from torchvision.utils import save_image


def append_list_to_list_linear(l1, l2):
    for k in range(0, len(l2)):
        if isinstance(l2[k], str):
            l1.append(l2[k]);
        else:
            l1.append(l2[k].item());
    return l1;

def write_samples_text_to_file(samples, filename):
    file_samples = open(filename, 'w');
    for k in range(0, len(samples)):
        file_samples.write(''.join(samples[k]) + '\n');
    file_samples.close();

def getText(samples):
    lines = []
    for k in range(0, len(samples)):
        lines.append(''.join(samples[k])[::-1])
    text = '\n\n'.join(lines)
    print(text)
    return text

def write_samples_img_to_file(samples, filename, img_per_row=1):
    save_image(samples.data.cpu(), filename, nrow=img_per_row);


def save_generated_samples_singlegroup(exp, batch_id, group_name, samples):
    paths_fid = exp.paths_fid
    modalities = exp.modalities
    batch_size = exp.flags.batch_size

    cnt_samples = batch_id * batch_size
    dir_save = paths_fid[group_name];

    modality_sample_pairs = list()

    for modality in modalities:
        if modality.name not in samples:
            continue

        modality_sample_pairs.append((modality, samples[modality.name]))
        dir_f = os.path.join(dir_save, modality.name)
        os.makedirs(dir_f, exist_ok=True)

    cnt_samples = batch_id * exp.flags.batch_size;
    for k in range(0, exp.flags.batch_size):
        for modality, sample in modality_sample_pairs:
            fn_out = os.path.join(dir_save, modality.name, str(cnt_samples).zfill(6) +
                                  modality.file_suffix);
            modality.save_data(samples[modality.name][k], fn_out, {'img_per_row': 1});
        cnt_samples += 1;
