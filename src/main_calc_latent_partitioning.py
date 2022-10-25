import argparse
import glob
import json
import os
import typing

import torch

from eval_metrics.latent_partitioning import (
    train_experts_classifier,
    test_expert_classifier,
    get_latents_tables,
    plot_identify_disentanglement
)
from experiments.mmnist.experiment import MMNISTExperiment
from utils.filehandling import create_dir_structure


def plot_figure(exp):
    from matplotlib import pyplot as plt

    image = 1 - exp.mm_vae.generate(num_samples=1)["m0"][0]
    image = image.cpu().detach().numpy().transpose(1, 2, 0)

    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(image)


def floatify_dict(config: typing.Dict) -> typing.Dict:
    for key in config:
        if isinstance(config[key], dict):
            config[key].update(floatify_dict(config[key]))
        elif isinstance(config[key], str):
            try:
                config[key] = float(config[key])
            except:
                pass

    return config


def main():
    args, flags, exp = load_exp()

    # wandb.init(
    #     project='mvae',
    #     name=args.run_name,
    #     group=args.run_group,
    #     dir=args.log_dir,
    #     config=floatify_dict(flags.__dict__),
    #     job_type="latent_partitioning"
    # )

    os.makedirs(args.log_dir, exist_ok=True)

    train_data_loader = torch.utils.data.DataLoader(
        exp.dataset_train,
        batch_size=exp.flags.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        exp.dataset_test,
        batch_size=exp.flags.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    if args.experts_classifier:
        expert_classifiers = train_experts_classifier(exp, train_data_loader, verbose=args.verbose)
        results_test = test_expert_classifier(
            exp, expert_classifiers, test_data_loader, verbose=args.verbose
        )
        results_train = test_expert_classifier(
            exp, expert_classifiers, train_data_loader, verbose=args.verbose
        )

        # for modality in results_test:
        #     wandb.run.summary[f"Expert Classifier/test_{modality}"] = results_test[
        #         modality
        #     ].__dict__
        #     wandb.run.summary[f"Expert Classifier/train_{modality}"] = results_train[
        #         modality
        #     ].__dict__

    if args.latents:
        latents = get_latents_tables(exp, test_data_loader, verbose=args.verbose)
        if args.verbose:
            print("Logging latent embeddings... ", end="", flush=True)
        for key in latents:
            if key == "columns":
                continue
            if args.verbose:
                print(f"'{key}', ", end="", flush=True)
            # wandb.log(
            #     {
            #         f"Latent Embeddings/table_{key}": wandb.Table(
            #             columns=latents["columns"], data=latents[key]
            #         )
            #     }
            # )

        if args.verbose:
            print("done.")

    if args.plot_latents_projections:
        latents = get_latents_tables(exp, test_data_loader, verbose=args.verbose)

        # figures, latents_transformed = plot_latents_projections(
        # figures, latents_transformed = plot_latent_correlations(
        #     exp, latents, verbose=args.verbose
        # )
        # figures = plot_latent_histograms(
        #     exp, latents, verbose=args.verbose
        # )

        figures = plot_identify_disentanglement(
            exp, latents, verbose=args.verbose
        )


        for fig_key, fig in figures.items():
            fig.savefig(os.path.join(args.log_dir, f"{fig_key}.png"))

        if args.verbose:
            print("done.")

        # for figure_id in figures:
        #     wandb.log({f"Latent Embeddings/figure_{figure_id}": figures[figure_id]})

        #for latents_transformed_id in latents_transformed:
        #    wandb.log(
        #        {
        #            f"Latent Embeddings/table_{latents_transformed_id}": wandb.Table(
        #                dataframe=latents_transformed[latents_transformed_id]
        #            )
        #        }
        #    )

    # wandb.finish()


def load_exp(args=None) -> typing.Tuple[
    argparse.Namespace, argparse.Namespace, MMNISTExperiment
]:
    try:
        parser.add_argument(
            "--run-path",
            type=str,
            default=None,
            required=True,
            help="Path to experiment training run",
        )
        parser.add_argument(
            "--epoch",
            type=int,
            default=-1,
            help="Which epoch to load VAE checkpoint from. If not found, the latest epoch is chosen.",
        )

        parser.add_argument(
            "--cpu", default=False, action="store_true", help="Force using cpu"
        )

        parser.add_argument(
            "--experts-classifier",
            default=False,
            action="store_true",
            help="Compute expert classifier summary",
        )

        parser.add_argument(
            "--latents", default=False, action="store_true", help="Compute latents summary"
        )

        parser.add_argument(
            "--plot-latents-projections",
            default=False,
            action="store_true",
            help="Plot projections from latents",
        )

        parser.add_argument(
            '--log-dir',
            type=str,
            default=None,
            required=True,
            help="Path to directory where offline logs will be written."
        )

        parser.add_argument(
            '--run-group',
            type=str,
            default=None,
            required=True,
            help="W&B run group"
        )

        parser.add_argument(
            '--run-name',
            type=str,
            default=None,
            required=True,
            help="W&B run name"
        )

        parser.add_argument(
            '--quiet',
            dest='verbose',
            default=True,
            action='store_false',
            help="Don't print status messages"
        )
    except:
        pass

    args = parser.parse_args(args=args)

    device = None
    cuda_available = torch.cuda.is_available()
    if args.cpu or not cuda_available:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if args.verbose:
        print(f"{args.run_path=}")

    run_path = glob.glob("*".join(list(map(glob.escape, args.run_path.split("*")))), recursive=True)[0]
    if args.verbose:
        print(f"{run_path=}")
    flags_rar = os.path.join(run_path, "flags.rar")
    flags = torch.load(flags_rar, map_location=device)
    assert len(flags.unimodal_datapaths_train) == len(flags.unimodal_datapaths_test)

    flags.unimodal_datapaths_train = args.unimodal_datapaths_train
    flags.unimodal_datapaths_test = args.unimodal_datapaths_test
    flags.pretrained_classifier_paths = args.pretrained_classifier_paths
    flags.dir_classifier = args.dir_classifier
    flags.dir_data = args.dir_data
    flags.dir_fid = args.dir_fid
    flags.inception_state_dict = args.inception_state_dict
    flags.dir_experiment = run_path

    checkpoints_path = os.path.join(run_path, "checkpoints")
    available_epochs_dirs = os.listdir(checkpoints_path)
    available_epochs = list()
    broken_checkpoints = False
    for epoch in available_epochs_dirs:
        epoch_path = os.path.join(checkpoints_path, epoch)
        if epoch.startswith("encoderM"):
            broken_checkpoints = True
            available_epochs = list()
            break
        elif not (os.path.isdir(epoch_path) and os.path.isfile(os.path.join(epoch_path, "mm_vae"))):
            continue

        available_epochs.append(int(epoch))

    available_epochs = sorted(available_epochs)
    if args.epoch in available_epochs:
        epoch = args.epoch
    elif len(available_epochs) > 0:
        epoch = available_epochs[-1]
    else:
        epoch = 1000

    flags.load_saved = True
    if not broken_checkpoints:
        flags.trained_model_path = os.path.join(
            run_path, "checkpoints", str(epoch).zfill(4), "mm_vae"
        )
    else:
        flags.trained_model_path = os.path.join(run_path, "checkpoints", "encoderM")

    if args.verbose:
        print(f"{flags.trained_model_path=}")
    flags.dir_experiment = run_path
    flags.num_mods = len(flags.unimodal_datapaths_train)
    flags.device = device

    if flags.div_weight_uniform_content is None:
        flags.div_weight_uniform_content = 1 / (flags.num_mods + 1)
    flags.alpha_modalities = [flags.div_weight_uniform_content]
    if flags.div_weight is None:
        flags.div_weight = 1 / (flags.num_mods + 1)
    flags.alpha_modalities.extend([flags.div_weight for _ in range(flags.num_mods)])

    flags = create_dir_structure(flags, train=False)
    alphabet_path = os.path.join(os.getcwd(), "resources/alphabet.json")
    with open(alphabet_path) as alphabet_file:
        alphabet = str(json.load(alphabet_file))

    exp = MMNISTExperiment(flags, alphabet)
    flags.exp = exp

    if broken_checkpoints:
        print(
            "WARNING: Using experiment where checkpoints were broken. Loading latest encoders instead..."
        )
        for idx in range(len(exp.mm_vae.modalities)):
            exp.mm_vae.encoders[idx].load_state_dict(
                torch.load(
                    os.path.join(flags.dir_checkpoints, f"encoderM{idx}"),
                    map_location=flags.device,
                )
            )
            exp.mm_vae.to(device=device)
    else:
        exp.mm_vae.load_state_dict(torch.load(flags.trained_model_path, map_location=flags.device))

    exp.mm_vae.pre_epoch_callback(epoch=epoch, num_iterations=0)

    return args, flags, exp


if __name__ == "__main__":
    main()
