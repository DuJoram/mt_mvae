import argparse
import os
from typing import Dict

import torch.cuda
import tqdm
from torch.utils.data import DataLoader

from eval_metrics.coherence import test_generation
from experiment_logger import WAndBLogger
from experiments import (
    MMNISTExperiment,
    MdSpritesExperiment,
    CelebaExperiment,
    MNISTSVHNText,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coher-experiment",
        type=str,
        required=True,
        choices=["CelebA", "MdSprites", "MNISTSVHNText", "MMNIST"],
        help="Experiment to instantiate",
    )

    parser.add_argument("--coher-log-dir", type=str, help="Target log directory")

    parser.add_argument(
        "--coher-training-experiment-dir",
        type=str,
        required=True,
        help="Path to training experiment base dir",
    )
    parser.add_argument(
        "--coher-checkpoints-epochs",
        type=str,
        default=["last"],
        nargs="+",
        help="If 'all' or 'last', will go through all or only the last checkpoint. If list of numbers, will evaluate all checkpoints of the corresponding epochs.",
    )

    parser.add_argument("--coher-run-name", type=str, help="Name of the run (for W&B)")
    parser.add_argument("--coher-run-group", type=str, help="Run group (for W&B)")

    args, remaining_args = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = None
    if args.coher_experiment == "CelebA":
        experiment = CelebaExperiment(device=device, args=remaining_args)
    elif args.coher_experiment == "MdSprites":
        experiment = MdSpritesExperiment(device=device, args=remaining_args)
    elif args.coher_experiment == "MMNIST":
        experiment = MMNISTExperiment(device=device, args=remaining_args)
    elif args.coher_experiment == "MNISTSVHNText":
        experiment = MNISTSVHNText(device=device, args=remaining_args)

    else:
        raise ValueError(f"Unknown experiment {args.coher_experiment}!")

    experiment.init()
    experiment.mm_vae.eval()

    joint_args = argparse.Namespace(**vars(experiment.flags), **vars(args))

    # logger = TBLogger(experiment.flags.str_experiment, experiment.flags.dir_logs)
    logger = WAndBLogger(
        logdir=joint_args.dir_logs,
        flags=joint_args,
        name=joint_args.coher_run_name,
        group=joint_args.coher_run_group,
        job_type="coherence",
    )

    logger.write_flags(joint_args)

    checkpoints_dir = None
    training_experiment_dir = args.coher_training_experiment_dir

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

    checkpoints_epochs = args.coher_checkpoints_epochs
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

    if checkpoints_epochs[0] == "last":
        epoch_keys = [epoch_keys[-1]]
    elif checkpoints_epochs[0] != "all":
        epoch_list = list()
        for epoch_str in checkpoints_epochs:
            epoch = int(epoch_str)
            assert epoch in epoch_keys
            epoch_list.append(epoch)

        epoch_keys = sorted(epoch_list)

    test_data_loader = DataLoader(
        experiment.dataset_test,
        batch_size=experiment.flags.batch_size,
        shuffle=True,
        num_workers=experiment.flags.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    train_data_loader = DataLoader(
        experiment.dataset_train,
        batch_size=experiment.flags.batch_size,
        shuffle=True,
        num_workers=experiment.flags.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    num_train_steps = len(train_data_loader)
    num_test_steps = len(test_data_loader)

    for epoch in tqdm.tqdm(epoch_keys, desc="Evaluating epoch"):
        model_checkpoint_file = os.path.join(checkpoints_dir, epochs[epoch], "mm_vae")
        experiment.load_checkpoint(model_checkpoint_file)
        step = (
                   epoch + 1
               ) * num_train_steps + epoch // experiment.flags.eval_freq * num_test_steps
        experiment.mm_vae.pre_epoch_callback(epoch, num_iterations=num_train_steps)
        with torch.no_grad():
            coherence_logs = test_generation(
                epoch=epoch, exp=experiment, data_loader=test_data_loader
            )
        logger.write_coherence_logs(coherence_logs, step=step)


if __name__ == "__main__":
    main()
