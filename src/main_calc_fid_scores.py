import os
import json

import torch

from tensorboardX import SummaryWriter
from experiment_logger import TBLogger

from utils.filehandling import create_dir_structure
from experiments.mmnist.experiment import MMNISTExperiment
from run_epochs import test

import argparse

if __name__ == '__main__':

    fid_parser = argparse.ArgumentParser(description='arguments for fid calculation.')
    fid_parser.add_argument('--unimodal-datapaths-train', nargs="+",
            type=str, help="directories where training data is stored")
    fid_parser.add_argument('--unimodal-datapaths-test', nargs="+",
            type=str, help="directories where testing data is stored")
    fid_parser.add_argument('--pretrained-classifier-paths', nargs="+",
            type=str, help="directories where testing data is stored")
    fid_parser.add_argument('--dir_exp_run', type=str, 
            default='data', help="directory where results are stored")


    args_fid = fid_parser.parse_args()
    flags_exp = torch.load(args_fid.dir_exp_run + '/flags.rar')
    use_cuda = torch.cuda.is_available()
    print(flags_exp)
    flags_exp.unimodal_datapaths_train = args_fid.unimodal_datapaths_train
    flags_exp.unimodal_datapaths_test = args_fid.unimodal_datapaths_test
    flags_exp.pretrained_classifier_paths = args_fid.pretrained_classifier_paths
    flags_exp.trained_model_path = os.path.join(args_fid.dir_exp_run, 'checkpoints', str(499).zfill(4), 'mm_vae')
    flags_exp.dir_experiment = args_fid.dir_exp_run
    print(flags_exp.unimodal_datapaths_train)

    # postprocess flags
    assert len(flags_exp.unimodal_datapaths_train) == len(flags_exp.unimodal_datapaths_test)
    # set number of modalities dynamically
    flags_exp.num_mods = len(flags_exp.unimodal_datapaths_train)  
    if flags_exp.div_weight_uniform_content is None:
        flags_exp.div_weight_uniform_content = 1 / (flags_exp.num_mods + 1)
    flags_exp.alpha_modalities = [flags_exp.div_weight_uniform_content]
    if flags_exp.div_weight is None:
        flags_exp.div_weight = 1 / (flags_exp.num_mods + 1)
    flags_exp.alpha_modalities.extend([flags_exp.div_weight for _ in
                                       range(flags_exp.num_mods)])
    print("alpha_modalities:", flags_exp.alpha_modalities)
    create_dir_structure(flags_exp, train=False)

    alphabet_path = os.path.join(os.getcwd(), 'resources/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mst = MMNISTExperiment(flags_exp, alphabet)

    writer = SummaryWriter(os.path.join(args_fid.dir_exp_run, 'logs'))
    tb_logger = TBLogger(mst.flags.str_experiment, writer)
    test(499, mst, tb_logger)
