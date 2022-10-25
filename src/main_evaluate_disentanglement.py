import argparse
import os
from typing import Dict, List

import torch

from eval_metrics.disentanglement import disentanglement_metric_higgins
from experiment_logger import WAndBLogger
from experiments import CelebaExperiment, MdSpritesExperiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disent-experiment",
        type=str,
        choices=["MdSprites", "CelebA"],
        help="Experiment to instantiate",
    )
    parser.add_argument(
        "--disent-num-workers",
        type=int,
        default=8,
        help="Number of workers for data loaders",
    )
    parser.add_argument(
        "--disent-training-experiment-dir",
        type=str,
        required=True,
        help="Path to training experiment base dir",
    )
    parser.add_argument(
        "--disent-checkpoints-epochs",
        type=str,
        default=["last"],
        nargs="+",
        help="If 'all' or 'last', will go through all or only the last checkpoint. If list of numbers, will evaluate all checkpoints of the corresponding epochs.",
    )
    parser.add_argument(
        "--disent-num-training-samples",
        type=int,
        default=100_000,
        help="Number of training samples to generate and use for training",
    )
    parser.add_argument(
        "--disent-num-test-samples",
        type=int,
        default=10_000,
        help="Number of test samples to generate and use for evaluation",
    )
    parser.add_argument(
        "--disent-num-diff-samples",
        type=int,
        default=64,
        help="Number of samples to average over to build difference latent",
    )
    parser.add_argument(
        "--disent-num-redundant-classifiers",
        type=int,
        default=50,
        help="Number of classifiers to train/test per subset",
    )

    parser.add_argument("--disent-run-name", type=str, help="Name of the run (for W&B)")
    parser.add_argument("--disent-run-group", type=str, help="Run group (for W&B)")

    args, remaining_args = parser.parse_known_args()

    device = torch.device("cpu")

    if args.disent_experiment == "MdSprites":
        experiment = MdSpritesExperiment(args=remaining_args, device=device)
    elif args.disent_experiment == "CelebA":
        experiments = CelebaExperiment(args=remaining_args, device=device)
    else:
        raise ValueError(f"Unknown or unsupported experiment {args.disent_experiment}")

    experiment.init()
    experiment.mm_vae.eval()

    joint_args = argparse.Namespace(**vars(experiment.flags), **vars(args))

    logger = WAndBLogger(
        logdir=joint_args.dir_logs,
        flags=joint_args,
        name=joint_args.disent_run_name,
        group=joint_args.disent_run_group,
        job_type="disentanglement",
    )

    logger.write_flags(joint_args)

    steps_per_train_epoch = len(experiment.dataset_train) // experiment.flags.batch_size
    steps_per_eval_epoch = len(experiment.dataset_test) // experiment.flags.batch_size

    training_experiment_dir = args.disent_training_experiment_dir

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

    checkpoints_epochs = args.disent_checkpoints_epochs
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

    num_attributes = len(experiment.subsets_shared_attributes_modalities_indices[-1])
    attribute_names = [
        experiment.attributes[experiment.modalities_subsets_indices[-1][0]][attr_idx[0]]
        for attr_idx in experiment.subsets_shared_attributes_modalities_indices[-1]
    ]

    eval_subsets_indices: List[List[int]] = list()
    eval_subsets_names: List[str] = list()
    num_modalities = len(experiment.modalities)

    individual_subsets = list()

    for subset_idx, (subset_name, subset) in enumerate(
        zip(experiment.modalities_subsets_names, experiment.modalities_subsets_indices)
    ):
        if len(subset) == 1 or len(subset) == num_modalities:
            eval_subsets_indices.append([subset_idx])
            eval_subsets_names.append(subset_name)
            if len(subset) == 1:
                individual_subsets.append(subset_idx)

    eval_subsets_names.append("individuals")
    eval_subsets_indices.append(individual_subsets)

    eval_subsets_names.append("all")
    eval_subsets_indices.append(list(range(1, len(experiment.modalities_subsets_indices))))


    for epoch in epoch_keys:
        model_checkpoint_file = os.path.join(checkpoints_dir, epochs[epoch], "mm_vae")
        experiment.load_checkpoint(model_checkpoint_file)
        step = (epoch + 1) * steps_per_train_epoch + epoch // experiment.flags.eval_freq * steps_per_eval_epoch
        experiment.mm_vae.pre_epoch_callback(epoch, num_iterations=steps_per_train_epoch)

        results = disentanglement_metric_higgins(
            experiment=experiment,
            subsets_sets=eval_subsets_indices,
            attribute_names=attribute_names,
            num_attributes=num_attributes,
            num_workers=joint_args.disent_num_workers,
            num_diff_samples=joint_args.disent_num_diff_samples,
            num_training_samples=joint_args.disent_num_training_samples,
            num_test_samples=joint_args.disent_num_test_samples,
            num_redundant_classifiers=joint_args.disent_num_redundant_classifiers,
            mix_subsets_in_diffs=True,
            class_dim=joint_args.class_dim,
        )

        for subset_name, subset_indices, result in zip(
            eval_subsets_names, eval_subsets_indices, results
        ):
            result.confusion_matrix_display.ax_.set_title(
                f"Confusion Matrix Epoch {epoch} on Joint Means for Subsets {subset_name}"
            )

            logger.write_disentanglement_metric(
                result=result,
                attribute_names=attribute_names,
                subset_name=subset_name,
                step=step,
                prefix="test",
            )

    logger.close()

if __name__ == "__main__":
    main()
