import argparse
import os
from typing import Dict

import torch.cuda
from torch.utils.data import DataLoader

from eval_metrics.total_correlation import estimate_total_correlation
from experiment_logger import TBLogger, WAndBLogger
from experiments import MdSpritesExperiment, CelebaExperiment, MMNISTExperiment


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tc-experiment",
        type=str,
        choices=["MdSprites", "CelebA", "MMNIST"],
        help="Experiment to instantiate",
    )
    parser.add_argument(
        "--tc-num-workers",
        type=int,
        default=8,
        help="Number of workers for data loaders",
    )
    parser.add_argument(
        "--tc-training-experiment-dir",
        type=str,
        required=True,
        help="Path to training experiment base dir",
    )
    parser.add_argument(
        "--tc-checkpoints-epochs",
        type=str,
        default=["last"],
        nargs="+",
        help="If 'all' or 'last', will go through all or only the last checkpoint. If list of numbers, will evaluate all checkpoints of the corresponding epochs.",
    )
    parser.add_argument(
        "--tc-num-samples",
        type=int,
        default=10,
        help="Number of latent samples to use for estimating log probabilities",
    )

    parser.add_argument("--tc-run-name", type=str, help="Name of the run (for W&B)")
    parser.add_argument("--tc-run-group", type=str, help="Run group (for W&B)")

    args, remaining_args = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tc_experiment == "MdSprites":
        experiment = MdSpritesExperiment(args=remaining_args, device=device)
    elif args.tc_experiment == "CelebA":
        experiment = CelebaExperiment(args=remaining_args, device=device)
    elif args.tc_experiment == "MMNIST":
        experiment = MMNISTExperiment(args=remaining_args, device=device)
    else:
        raise ValueError(f"Unknown or unsupported experiment {args.tc_experiment}")

    experiment.init()
    experiment.mm_vae.eval()

    joint_args = argparse.Namespace(**vars(experiment.flags), **vars(args))

    # logger = TBLogger(name=joint_args.tc_run_name, dir_logs=joint_args.dir_logs)
    logger = WAndBLogger(
        logdir=joint_args.dir_logs,
        flags=joint_args,
        name=joint_args.tc_run_name,
        group=joint_args.tc_run_group,
        job_type="total-correlation",
    )

    logger.write_flags(joint_args)

    training_experiment_dir = args.tc_training_experiment_dir

    checkpoints_dir = None

    for subdir in os.listdir(training_experiment_dir):
        if checkpoints_dir is not None:
            break

        current_dir = os.path.join(training_experiment_dir, subdir)
        if not os.path.isdir(current_dir):
            continue
        if subdir == "checkpoints":
            checkpoints_dir = current_dir

        for subsubdir in os.listdir(current_dir):
            current_subdir = os.path.join(current_dir, subsubdir)
            if not os.path.isdir(current_subdir):
                continue

            if subsubdir == "checkpoints":
                checkpoints_dir = current_subdir
                break

    if checkpoints_dir is None:
        raise ValueError(
            f"Could not find checkpoints path in training experiment dir {training_experiment_dir}"
        )

    checkpoints_epochs = args.tc_checkpoints_epochs
    epochs: Dict[int, str] = dict()

    for epoch_dir in os.listdir(checkpoints_dir):
        checkpoint_dir = os.path.join(checkpoints_dir, epoch_dir)
        if not os.path.isdir(checkpoint_dir):
            continue

        try:
            epoch = int(epoch_dir)
            epochs[epoch] = epoch_dir
        except ValueError:
            continue

    epoch_keys = sorted(epochs)
    assert len(epoch_keys) > 0

    if checkpoints_epochs[0] == "last":
        epoch_keys = [epoch_keys[-1]]
    elif checkpoints_epochs[0] != "all":
        epoch_list = list()
        for epoch_str in checkpoints_epochs:
            epoch = int(epoch_str)
            assert epoch in epoch_keys
            epoch_list.append(epoch)

        epoch_keys = sorted(epoch_list)

    data_loader_train = DataLoader(
        dataset=experiment.dataset_train,
        num_workers=args.tc_num_workers,
        shuffle=True,
        batch_size=experiment.flags.batch_size,
        drop_last=True,
    )

    steps_per_train_epoch = len(experiment.dataset_train) // experiment.flags.batch_size
    steps_per_eval_epoch = len(experiment.dataset_test) // experiment.flags.batch_size

    for epoch in epoch_keys:
        model_checkpoint_file = os.path.join(checkpoints_dir, epochs[epoch], "mm_vae")
        experiment.load_checkpoint(model_checkpoint_file)

        step = (epoch + 1) * steps_per_train_epoch + epoch // experiment.flags.eval_freq * steps_per_eval_epoch
        experiment.mm_vae.pre_epoch_callback(epoch, num_iterations=steps_per_train_epoch)

        total_correlations = estimate_total_correlation(
            data_loader=data_loader_train,
            subset_names=experiment.modalities_subsets_names,
            mm_vae=experiment.mm_vae,
            batch_size=joint_args.batch_size,
            latent_size=joint_args.class_dim,
            num_samples=joint_args.tc_num_samples,
            device=device,
            verbose=True,
        )

        logger.write_total_correlations(
            total_correlations,
            step=step,
        )

    pass


if __name__ == "__main__":
    main()
