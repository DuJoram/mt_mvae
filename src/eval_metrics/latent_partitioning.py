import itertools
import typing

import numpy as np
import pandas as pd
import torch


from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader

from eval_metrics.expert_classifier import ExpertClassifier
from experiments.experiment import Experiment
from utils.utils import printProgressBar

import plotly.express as px
import matplotlib as mpl
import matplotlib.figure
from matplotlib import pyplot as plt

import dataclasses


@dataclasses.dataclass
class Latents:
    modalities: typing.List[str]
    means: pd.DataFrame
    log_variances: pd.DataFrame
    subset_means: pd.DataFrame
    subset_log_variances: pd.DataFrame


class ExpertClassifierStatistics(object):
    def __init__(
        self,
        expert_name: str,
        true_positives: int,
        false_positives: int,
        true_negatives: int,
        false_negatives: int,
    ):
        self.expert_name = expert_name
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives

        self.positives = true_positives + false_positives
        self.negatives = true_negatives + false_negatives

        self.accuracy = (true_positives + true_negatives) / (
            self.positives + self.negatives
        )
        self.precision = true_positives / (true_positives + false_positives)

        if self.negatives > 0:
            self.recall = true_positives / self.positives
        else:
            self.recall = -1

        self.f1 = (
            2
            * true_positives
            / (2 * true_positives + false_positives + false_negatives)
        )


@torch.enable_grad()
def train_experts_classifier(exp: Experiment, data_loader: torch.utils.data.DataLoader, verbose: bool = False) -> ExpertClassifier:
    mm_vae = exp.mm_vae
    batch_size = exp.flags.batch_size
    class_dim = exp.flags.class_dim
    modalities = exp.modalities
    subsets = exp.subsets
    subsets_indices = exp.modalities_subsets_indices
    subsets_names = exp.modalities_subsets_names

    expert_classifier_layers = exp.flags.expert_classifier_layers
    expert_classifier_learning_rate = exp.flags.expert_classifier_learning_rate
    expert_classifier_epochs = exp.flags.expert_classifier_epochs

    device = exp.flags.device

    mm_vae.eval()
    mm_vae.train(False)


    num_experts = len(mm_vae.modalities)
    expert_classifier = ExpertClassifier(
        latent_size=class_dim,
        n_experts=num_experts,
        n_layers=expert_classifier_layers,
    )

    expert_classifier.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        expert_classifier.parameters(recurse=True),
        lr=expert_classifier_learning_rate,
    )

    subset2label = dict()

    for subset_name, subset_indices in zip(subsets_names, subsets_indices):
        label = torch.zeros(num_experts, device=device)
        for modality_idx in subset_indices:
            label[modality_idx] = 1

        label = label.unsqueeze(0).repeat(batch_size, 1)
        subset2label[subset_name] = label

    iterations_per_epoch = len(data_loader)
    total_iterations = exp.flags.expert_classifier_epochs * iterations_per_epoch
    print_mod = max(1, total_iterations // 100)
    for epoch in range(expert_classifier_epochs):
        steps = 0
        mean_loss = torch.zeros(1, device=device)
        for iteration, batch in enumerate(data_loader):
            batch_data = batch[0]

            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)

            with torch.no_grad():
                inferred = mm_vae.inference(batch_data)
            latent_subsets = inferred["subsets"]

            for subset_key in latent_subsets:
                mean, log_var = latent_subsets[subset_key]
                if len(subset_key) == 0 or mean is None or log_var is None:
                    continue

                label = subset2label[subset_key]

                optimizer.zero_grad()

                prediction = expert_classifier(mean, log_var)
                loss = loss_fn(prediction, label)
                loss.backward()
                optimizer.step()

                mean_loss += loss
                steps += 1
            with torch.no_grad():
                if (
                    verbose
                    and (iterations_per_epoch * epoch + iteration) % print_mod == 0
                ):
                    printProgressBar(
                        epoch * iterations_per_epoch + iteration + 1,
                        total_iterations,
                        suffix=f"epoch={epoch+1}/{expert_classifier_epochs}, loss={mean_loss/steps}",
                    )
                    mean_loss = 0
                    steps = 0
        with torch.no_grad():
            if verbose and steps > 0:
                printProgressBar(
                    epoch * iterations_per_epoch + iteration + 1,
                    total_iterations,
                    suffix=f"epoch={epoch + 1}/{expert_classifier_epochs}, loss={mean_loss / steps}",
                )

    if verbose:
        print()

    return expert_classifier


def test_expert_classifier(
    exp, expert_classifier, data_loader: torch.utils.data.DataLoader, verbose: bool = False
) -> typing.Dict[str, ExpertClassifierStatistics]:
    mm_vae = exp.mm_vae
    batch_size = exp.flags.batch_size
    num_experts = len(mm_vae.modalities)
    subsets = exp.subsets
    device = exp.flags.device

    mm_vae.eval()
    expert_classifier.eval()

    modality2index = dict()
    index2modality = list()
    subset2label = dict()

    split_true_positives = torch.zeros(num_experts, device=device)
    split_false_positives = torch.zeros(num_experts, device=device)
    split_true_negatives = torch.zeros(num_experts, device=device)
    split_false_negatives = torch.zeros(num_experts, device=device)

    for idx, modality in enumerate(mm_vae.modalities):
        modality2index[modality.name] = idx

    for subset in subsets:
        label = torch.zeros(len(mm_vae.modalities), device=device)
        for modality in subsets[subset]:
            label[modality2index[modality.name]] = 1

        label = label.unsqueeze(0).repeat(batch_size, 1)
        subset2label[subset] = label

    one = torch.ones(1, device=device)
    zero = torch.zeros(1, device=device)

    total_iterations = len(data_loader)
    print_mod = max(1, total_iterations // 100)
    for iteration, batch in enumerate(data_loader):
        batch_data = batch[0]

        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)

        inferred = mm_vae.inference(batch_data)
        latent_subsets = inferred["subsets"]

        subset_key = ""
        while len(subset_key) == 0:
            subset_key = np.random.choice(list(subsets))

        mean, log_var = latent_subsets[subset_key]
        if len(subset_key) == 0 or mean is None or log_var is None:
            continue

        label = subset2label[subset_key]
        label_true = torch.isclose(label, one)
        label_false = torch.isclose(label, zero)

        prediction = expert_classifier(mean, log_var)

        prediction_rounded = torch.round(prediction)
        correct = torch.isclose(prediction_rounded, label)
        incorrect = correct == False  # Has to be == for element wise 'inversion'

        true_positives = correct & label_true
        false_positives = incorrect & label_false
        true_negatives = correct & label_false
        false_negatives = incorrect & label_true

        split_true_positives += true_positives.sum(0)
        split_false_positives += false_positives.sum(0)
        split_true_negatives += true_negatives.sum(0)
        split_false_negatives += false_negatives.sum(0)

        if iteration % print_mod == 0:
            printProgressBar(iteration, total_iterations)

    if verbose:
        printProgressBar(1, 1)
    total_true_positives = split_true_positives.sum()
    total_false_positives = split_false_positives.sum()
    total_true_negatives = split_true_negatives.sum()
    total_false_negatives = split_false_negatives.sum()

    result = dict()
    result["total"] = ExpertClassifierStatistics(
        expert_name="total",
        true_positives=total_true_positives.cpu().item(),
        false_positives=total_false_positives.cpu().item(),
        true_negatives=total_true_negatives.cpu().item(),
        false_negatives=total_false_negatives.cpu().item(),
    )

    for modality in modality2index:
        modality_index = modality2index[modality]
        result[modality] = ExpertClassifierStatistics(
            expert_name=modality,
            true_positives=split_true_positives[modality_index].cpu().item(),
            false_positives=split_false_positives[modality_index].cpu().item(),
            true_negatives=split_true_negatives[modality_index].cpu().item(),
            false_negatives=split_false_negatives[modality_index].cpu().item(),
        )

    if verbose:
        print()

    return result


@torch.no_grad()
def get_latents_tables(
    exp, data_loader: torch.utils.data.DataLoader, verbose: bool = False
) -> Latents:
    mm_vae = exp.mm_vae
    modalities = exp.modalities
    class_dim = exp.flags.class_dim
    batch_size = exp.flags.batch_size
    data_loader_test = exp.data_loader_test
    device = exp.flags.device

    mm_vae.eval()
    mm_vae.train(False)


    all_modalities_subset_key = "_".join([modality.name for modality in modalities])

    columns_shared = [f"latent_{idx}" for idx in range(class_dim)]
    columns_modalities = ["label", "modality"] + columns_shared
    columns_joint = (
        ["label", "subset"] + [modality for modality in modalities] + columns_shared
    )

    total_iterations = len(data_loader)
    print_modulus = max(1, total_iterations // 100)

    means: typing.Optional[pd.DataFrame] = None
    log_variances: typing.Optional[pd.DataFrame] = None
    subset_means: typing.Optional[pd.DataFrame] = None
    subset_log_variances: typing.Optional[pd.DataFrame] = None

    for iteration, batch in enumerate(data_loader):
        batch: typing.Tuple[typing.Dict[str, torch.Tensor], torch.Tensor]
        images, labels = batch
        for modality in images:
            images[modality] = images[modality].to(device)
        inferred = mm_vae.inference(images)

        inferred_modalities = inferred["modalities"]
        batch_joint_means, batch_joint_log_variances = inferred["joint"]
        # subset_means = np.append(np.concatenate(batch_joint_means.cpu().detach().numpy().flatten())
        # subset_log_variances(batch_joint_log_variances.cpu().detach().numpy().flatten().tolist())

        labels_df = pd.DataFrame(
            labels.detach().cpu().numpy().astype(str), columns=["label"], dtype="string"
        )

        inferred_subsets = inferred["subsets"]

        for subset_key, (subset_mean, subset_log_variance) in inferred["subsets"].items():
            if subset_key not in modalities and subset_key != all_modalities_subset_key:
                continue
            subset_df = pd.DataFrame(
                np.array([subset_key], dtype=str).repeat(batch_size),
                columns=["subset"],
                dtype="string",
            )
            modalities_df = pd.DataFrame(
                np.array(
                    [
                        [
                            True if modality.name in subset_key else False
                            for modality in modalities
                        ]
                    ],
                    dtype=bool,
                ).repeat(batch_size, 0),
                columns=[modality.name for modality in modalities],
                dtype="bool",
            )

            batch_subset_means = pd.concat(
                (
                    labels_df,
                    subset_df,
                    modalities_df,
                    pd.DataFrame(
                        subset_mean.detach().cpu().numpy(), columns=columns_shared
                    ),
                ),
                axis=1,
            )

            batch_subset_log_variances = pd.concat(
                (
                    labels_df,
                    subset_df,
                    modalities_df,
                    pd.DataFrame(
                        subset_log_variance.detach().cpu().numpy(),
                        columns=columns_shared,
                    ),
                ),
                axis=1,
            )

            if subset_means is None:
                subset_means = batch_subset_means
            else:
                subset_means = pd.concat(
                    (subset_means, batch_subset_means), axis=0, ignore_index=True
                )

            if subset_log_variances is None:
                subset_log_variances = batch_subset_log_variances
            else:
                subset_log_variances = pd.concat(
                    (subset_log_variances, batch_subset_log_variances),
                    axis=0,
                    ignore_index=True,
                )

        for modality, (means_batch, log_variances_batch) in inferred[
            "modalities"
        ].items():

            if means_batch is None or log_variances_batch is None:
                continue

            modality_df = pd.DataFrame(
                np.array([modality], dtype=str).repeat(batch_size),
                columns=["modality"],
                dtype="string",
            )

            means_batch_data_df = pd.DataFrame(
                means_batch.detach().cpu().numpy(), columns=columns_shared
            )
            means_batch_df = pd.concat(
                (labels_df, modality_df, means_batch_data_df),
                axis=1,
            )

            log_variances_batch_data_df = pd.DataFrame(
                log_variances_batch.detach().cpu().numpy(), columns=columns_shared
            )
            log_variances_batch_df = pd.concat(
                (labels_df, modality_df, log_variances_batch_data_df),
                axis=1,
            )

            if means is None:
                means = means_batch_df

            else:
                means = pd.concat((means, means_batch_df), axis=0, ignore_index=True)

            if log_variances is None:
                log_variances = log_variances_batch_df

            else:
                log_variances = pd.concat(
                    (log_variances, log_variances_batch_df), axis=0, ignore_index=True
                )

        if verbose and ((iteration + 1) % print_modulus) == 0:
            printProgressBar(iteration + 1, total_iterations, prefix="Latents")

    if verbose:
        print("")

    latents = Latents(
        modalities=modalities,
        means=means,
        log_variances=log_variances,
        subset_means=subset_means,
        subset_log_variances=subset_log_variances,
    )
    return latents


@torch.no_grad()
def plot_latents_projections(
    exp,
    latents: Latents,
    verbose: bool = False,
) -> typing.Tuple[typing.Dict[str, plt.Figure], typing.Dict[str, pd.DataFrame]]:
    modalities = exp.modalities

    modality_names = [modality.name for modality in modalities]
    subsets = list(
        map(
            list,
            itertools.chain.from_iterable(
                [
                    itertools.combinations(modality_names, r=r)
                    for r in range(1, len(modalities) + 1)
                ]
            ),
        )
    )

    figures = dict()
    latents_transformed = dict()

    for key, latent_data in dataclasses.asdict(latents).items():
        latent_data: pd.DataFrame
        for subset in subsets:
            subset_string = "_".join(subset)

            if "modality" not in latent_data.columns:
                continue

            if verbose:
                print(f"{key}: {subset_string}")
                print("\tComputing PCA... ", end="", flush=True)

            latents_subset = latent_data[latent_data["modality"].isin(subset)]
            pca = PCA(
                n_components=2,
            )
            latents_subset_values = latents_subset.drop(["label", "modality"], axis=1)

            if verbose:
                print("")

            pca.fit(latents_subset_values)
            latents_subset_transformed = pca.transform(latents_subset_values)

            latents_subset_transformed = pd.DataFrame(
                latents_subset_transformed, columns=["projected_x", "projected_y"]
            )

            latents_subset_transformed["label"] = latents_subset["label"].values
            latents_subset_transformed["modality"] = latents_subset["modality"].values

            num_labels = len(latents_subset_transformed["label"])

            figure_id = f"{key}_{subset_string}"
            if len(subset) == 1:
                figure = px.scatter(
                    data_frame=latents_subset_transformed,
                    x="projected_x",
                    y="projected_y",
                    color="label",
                    title=f"PCA {key} on {subset_string}",
                )
            else:
                figure = px.scatter(
                    data_frame=latents_subset_transformed,
                    x="projected_x",
                    y="projected_y",
                    color="modality",
                    symbol="label",
                    title=f"PCA {key} on {subset_string}",
                )
            figures[figure_id] = figure

            # for label in sorted(latents_subset_transformed["label"].unique()):
            #     figure_id = f"{key}_{subset_string}_{label}"

            # figure = plt.figure()
            # ax = figure.subplots()

            # for modality in subset:
            #     data = latents_subset_transformed[
            #         (latents_subset_transformed["label"] == label)
            #         & (latents_subset_transformed["modality"] == modality)
            #     ]
            #     ax.scatter(data["projected_x"], data["projected_y"], label=modality)

            # ax.set_title(f"PCA {key} on {subset_string}, label={label}")
            # ax.legend(loc="best", shadow=False, scatterpoints=1)

            # figures[figure_id] = figure

            latents_transformed[f"{key}_{subset_string}"] = latents_subset_transformed

    return figures, latents_transformed


def softmax(x: np.ndarray) -> np.ndarray:
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


def mean_diff_samples(means: np.array, log_variances: np.array) -> np.array:
    samples_standard = np.random.randn(*means.shape)

    samples = np.exp(0.5 * log_variances) + means

    permutation = np.random.permutation(means.shape[0])

    half_length = len(permutation) // 2

    samples_1 = samples[:half_length]
    samples_2 = samples[half_length : 2 * half_length]

    samples_diffs = np.abs(samples_1 - samples_2)
    samples_mean_diffs = np.mean(samples_diffs, axis=tuple(range(len(means.shape) - 1)))
    return samples_mean_diffs


@torch.no_grad()
def plot_identify_disentanglement(
    exp: Experiment,
    latents: Latents,
    softmin_temp: float = 1,
    softmax_temp: float = 1,
    verbose: bool = False,
) -> typing.Tuple[
    typing.Dict[str, mpl.figure.Figure],
    typing.Dict[str, typing.List[np.array]],
    typing.Dict[str, typing.List[np.array]],
    typing.Dict[str, typing.List[np.array]],
]:

    labels = latents.means["label"].unique()
    labels = sorted(labels)

    subset_means = latents.subset_means
    subset_log_variances = latents.subset_log_variances

    modalities = latents.modalities

    plots: typing.Dict[str, mpl.figure.Figure] = dict()

    joint_subset_key = "_".join(modalities)

    y_ticks = list(range(len(modalities) + 1))
    y_tick_labels = modalities + ["joint"]

    figures: typing.Dict[str, mpl.figure.Figure] = dict()

    mean_diffs: typing.Dict[str, typing.List[np.array]] = dict()
    mean_diffs_softmax: typing.Dict[str, typing.List[np.array]] = dict()
    mean_diffs_softmin: typing.Dict[typing.List[np.array]] = dict()

    for label_idx, label in enumerate(labels):
        subset_means_filtered_label = subset_means[subset_means["label"] == label]
        subset_log_variances_filtered_label = subset_log_variances[
            subset_log_variances["label"] == label
        ]

        figure_key = f"mean_diffs_{label}"
        figure_softmax_key = f"mean_diffs_softmax_{label}"
        figure_softmin_key = f"mean_diffs_softmin_{label}"

        mean_diffs[label] = list()
        mean_diffs_softmax[label] = list()
        mean_diffs_softmin[label] = list()

        for modality_idx, modality in enumerate(modalities):
            subset_means_filtered_label_modality = (
                subset_means_filtered_label[
                    subset_means_filtered_label["subset"] == modality
                ]
                .drop(modalities + ["subset"] + ["label"], axis=1)
                .to_numpy()
            )
            subset_log_variances_filtered_label_modality = (
                subset_log_variances_filtered_label[
                    subset_log_variances_filtered_label["subset"] == modality
                ]
                .drop(modalities + ["subset"] + ["label"], axis=1)
                .to_numpy()
            )

            modality_means_diff_samples = mean_diff_samples(
                means=subset_means_filtered_label_modality,
                log_variances=subset_log_variances_filtered_label_modality,
            )
            mean_diffs[label].append(modality_means_diff_samples)
            mean_diffs_softmax[label].append(
                softmax(1.0 / softmax_temp * modality_means_diff_samples)
            )
            mean_diffs_softmin[label].append(
                softmax(1.0 / softmin_temp * (-modality_means_diff_samples))
            )

        joint_means_filtered_label = (
            subset_means[subset_means["subset"] == joint_subset_key]
            .drop(["subset", "label"] + modalities, axis=1)
            .to_numpy()
        )
        joint_log_variances_filtered_label = (
            subset_log_variances[subset_log_variances["subset"] == joint_subset_key]
            .drop(["subset", "label"] + modalities, axis=1)
            .to_numpy()
        )

        joint_mean_diff_samples = mean_diff_samples(
            means=joint_means_filtered_label,
            log_variances=joint_log_variances_filtered_label,
        )
        mean_diffs[label].append(joint_mean_diff_samples)
        mean_diffs_softmax[label].append(
            softmax(1.0 / softmax_temp * joint_mean_diff_samples)
        )
        mean_diffs_softmin[label].append(
            softmax(1.0 / softmin_temp * (-joint_mean_diff_samples))
        )

        fig = plt.figure(figsize=(10, 5))
        fig.set_dpi(196)
        ax = fig.subplots()
        ax.matshow(mean_diffs[label], cmap=mpl.cm.Blues)
        ax.set_title(f"Mean Sample Diffs, label = {label}")
        ax.set_aspect("auto")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Latent coordinate")
        ax.set_ylabel("Modality")

        fig_softmax = plt.figure(figsize=(10, 5))
        fig_softmax.set_dpi(196)
        ax = fig_softmax.subplots()
        ax.matshow(mean_diffs_softmax[label], cmap=mpl.cm.Blues)
        ax.set_title(f"Softmax Mean Sample Diffs, label = {label}")
        ax.set_aspect("auto")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Latent coordinate")
        ax.set_ylabel("Modality")

        fig_softmin = plt.figure(figsize=(10, 5))
        fig_softmin.set_dpi(196)
        ax = fig_softmin.subplots()
        ax.matshow(mean_diffs_softmin[label], cmap=mpl.cm.Blues)
        ax.set_title(f"Softmin Mean Sample Diffs, label = {label}")
        ax.set_aspect("auto")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Latent coordinate")
        ax.set_ylabel("Modality")

        figures[figure_key] = fig
        figures[figure_softmax_key] = fig_softmax
        figures[figure_softmin_key] = fig_softmin

    return figures, mean_diffs, mean_diffs_softmin, mean_diffs_softmax


@torch.no_grad()
def plot_latent_histograms(
    exp: Experiment, latents: Latents, verbose: bool = False
) -> typing.Dict[str, mpl.figure.Figure]:

    mm_vae = exp.mm_vae
    modalities = exp.modalities
    class_dim = exp.flags.class_dim
    device = exp.flags.device


    labels = latents.means["label"].unique()
    labels = sorted(labels)

    latent_size = class_dim
    modality_names = [modality.name for modality in modalities]

    y_labels = modality_names + ["joint"]
    softmax_precisions: typing.List[typing.List[np.array]] = list()

    for label in labels:
        log_var_filtered_label = latents.log_variances[
            latents.log_variances["label"] == label
        ]

        label_softmax_precisions = list()

        for modality in modalities:
            log_var_filtered_label_modality = log_var_filtered_label[
                log_var_filtered_label["modality"] == modality.name
            ].drop(["label", "modality"], axis=1)
            mean_log_var = log_var_filtered_label_modality.to_numpy().mean(0)
            softmax_mean_log_precisions = softmax(-mean_log_var)
            label_softmax_precisions.append(softmax_mean_log_precisions)

        log_variance_joint_filtered_label = latents.subset_log_variances[
            latents.subset_log_variances["label"] == label
        ].drop(["label"], axis=1)
        mean_log_variance_joint = log_variance_joint_filtered_label.to_numpy().mean(0)
        softmax_mean_log_precision_joint = softmax(-mean_log_variance_joint)
        label_softmax_precisions.append(softmax_mean_log_precision_joint)

        softmax_precisions.append(label_softmax_precisions)

    def wowplots(coord: int, seed: torch.Tensor, mod: int):
        seed = seed.to(device)
        seed[:, coord] = torch.linspace(-2, 2, 25).to(device)

        imgs, _ = mm_vae.decoders[mod](None, seed)

        fig = plt.figure()
        ax = fig.subplots(5, 5)
        for y in range(5):
            for x in range(5):
                idx = y * 5 + x
                ax[y, x].imshow(X=imgs[idx].permute(2, 1, 0).cpu())

        plt.imshow()

    figures: typing.Dict[str, mpl.figure.Figure] = dict()

    for label_idx, label in enumerate(labels):
        y_labels = [""] + modality_names + ["joint"]

        fig = mpl.figure.Figure()
        fig.set_size_inches(10, 5)
        fig.set_dpi(196)
        ax = fig.subplots()

        ax.matshow(softmax_precisions[label_idx], cmap=mpl.cm.Blues, vmin=0, vmax=1)
        ax.set_aspect("auto")
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("latent dimension")
        ax.set_ylabel("modality")
        ax.set_title(
            f"Softmax of mean log precisions with label {label}, grouped by modalities"
        )
        figure_key = f"softmax_avg_log_precision_label_{label}"

        figures[figure_key] = fig

    return figures


@torch.no_grad()
def plot_latent_correlations(
    exp: Experiment,
    latents: Latents,
    verbose: bool = False,
) -> typing.Tuple[typing.Dict[str, plt.Figure], typing.Dict[str, pd.DataFrame]]:

    modalities = exp.modalities

    correlations = dict()

    modality_names = [modality.name for modality in modalities]
    pairs = list(map(list, itertools.combinations(modality_names, 2)))

    mean = latents.means
    log_var = latents.log_variances
    var = log_var.exp()
    normalized = mean / var

    labels = mean["label"].unique()
    labels.sort()

    def corr(mat1, mat2):
        return (mat1 - mat1.mean(axis=1, keepdims=True)) @ (
            mat2 - mat2.mean(axis=1, keepdims=True)
        ).T

    for label in labels:
        var_filtered = var[var["label"] == label]
        log_var_filtered = log_var[log_var["label"] == label]

        var_latents = list()
        prec_latents = list()

        fig = mpl.figure.Figure(figsize=(15, 10), dpi=96)
        axes = fig.subplots(len(modalities))

        for modality_idx, modality in enumerate(modalities):
            var_latent_label_mod = (
                var_filtered[var_filtered["modality"] == modality.name]
                .drop(["modality", "label"], axis=1)
                .abs()
                .mean(axis=0)
            )
            log_var_latent_label_mod = log_var_filtered[
                log_var_filtered["modality"] == modality.name
            ].drop(["modality", "label"], axis=1)
            var_latent_label_mod = var_latent_label_mod / var_latent_label_mod.max()
            var_latents.append(var_latent_label_mod)
            prec_latent_label_mod = var_latent_label_mod.min() / var_latent_label_mod
            prec_latents.append(prec_latent_label_mod)

            log_exp_latent_label_mod = -log_var_latent_label_mod
            xedges = np.arange(0, log_exp_latent_label_mod.shape[1] + 1)
            xcoords = (
                np.arange(0, log_exp_latent_label_mod.shape[1]).astype(float)[None, :]
                + 0.5
            ).repeat(log_exp_latent_label_mod.shape[0], 0)
            hist, xedges, yedges = np.histogram2d(
                xcoords.flatten(),
                log_exp_latent_label_mod.to_numpy().flatten(),
                bins=[xedges, 40],
            )
            X, Y = np.meshgrid(xedges, yedges)
            axes[modality_idx].pcolormesh(X, Y, hist.T)
            axes[modality_idx].set_aspect("auto")
            axes[modality_idx].set_title(
                f"Precisions histogram modality = {modality.name}, label = {label}"
            )

        fig.savefig(f"tmpfigs/precision_histogram_l{label}.png", dpi="figure")

        fig = mpl.figure.Figure(figsize=(15, 3), dpi=96)
        ax = fig.subplots()
        ax.matshow(var_latents, cmap=mpl.cm.Blues)
        ax.set_aspect("auto")
        ax.set_title(f"Absolute variances: label = {label}")
        fig.savefig(f"tmpfigs/absolute_variances_{label}.png", dpi="figure")

        fig = mpl.figure.Figure(figsize=(15, 3), dpi=96)
        ax = fig.subplots()
        ax.matshow(prec_latents, cmap=mpl.cm.Blues)
        ax.set_aspect("auto")
        ax.set_title(f"Absolute precisions: label = {label}")
        fig.savefig(f"tmpfigs/absolute_precisions_{label}.png", dpi="figure")

    for modality1, modality2 in pairs:
        mean_modality1 = mean[mean["modality"] == modality1]
        mean_modality2 = mean[mean["modality"] == modality2]
        var_modality1 = var[var["modality"] == modality1]
        var_modality2 = var[var["modality"] == modality2]
        normalized_modality1 = normalized[normalized["modality"] == modality1]
        normalized_modality2 = normalized[normalized["modality"] == modality2]

        mean_latent_modality1 = (
            mean_modality1.drop(["modality", "label"], axis=1).to_numpy().T
        )
        mean_latent_modality2 = (
            mean_modality2.drop(["modality", "label"], axis=1).to_numpy().T
        )
        var_latent_modality1 = (
            var_modality1.drop(["modality", "label"], axis=1).to_numpy().T
        )
        var_latent_modality2 = (
            var_modality2.drop(["modality", "label"], axis=1).to_numpy().T
        )
        normalized_latent_modality1 = (
            normalized_modality1.drop(["modality", "label"], axis=1).to_numpy().T
        )
        normalized_latent_modality2 = (
            normalized_modality2.drop(["modality", "label"], axis=1).to_numpy().T
        )

        mean_correlation = corr(mean_latent_modality1, mean_latent_modality2)
        var_correlation = corr(var_latent_modality1, var_latent_modality2)
        normalized_correlation = corr(
            normalized_latent_modality1, normalized_latent_modality2
        )

        for label in labels:
            mean_latent_modality1 = (
                mean_modality1[mean_modality1["label"] == label]
                .drop(["modality", "label"], axis=1)
                .to_numpy()
                .T
            )
            mean_latent_modality2 = (
                mean_modality2[mean_modality2["label"] == label]
                .drop(["modality", "label"], axis=1)
                .to_numpy()
                .T
            )
            var_latent_modality1 = (
                var_modality1[var_modality1["label"] == label]
                .drop(["modality", "label"], axis=1)
                .to_numpy()
                .T
            )
            var_latent_modality2 = (
                var_modality2[var_modality2["label"] == label]
                .drop(["modality", "label"], axis=1)
                .to_numpy()
                .T
            )
            normalized_latent_modality1 = (
                normalized_modality1[normalized_modality1["label"] == label]
                .drop(["modality", "label"], axis=1)
                .to_numpy()
                .T
            )
            normalized_latent_modality2 = (
                normalized_modality2[normalized_modality2["label"] == label]
                .drop(["modality", "label"], axis=1)
                .to_numpy()
                .T
            )

            n_mean_latent_modality1 = (
                mean_latent_modality1 / mean_latent_modality1.max()
            )
            n_mean_latent_modality2 = (
                mean_latent_modality2 / mean_latent_modality2.max()
            )
            n_var_latent_modality1 = var_latent_modality1 / var_latent_modality1.max()
            n_var_latent_modality2 = var_latent_modality2 / var_latent_modality2.max()
            n_normalized_latent_modality1 = (
                normalized_latent_modality1 / normalized_latent_modality1.max()
            )
            n_normalized_latent_modality2 = (
                normalized_latent_modality2 / normalized_latent_modality2.max()
            )

            mean_correlation_label = corr(mean_latent_modality1, mean_latent_modality2)
            var_correlation_label = corr(var_latent_modality1, var_latent_modality2)
            normalized_correlation_label = corr(
                normalized_latent_modality1, normalized_latent_modality2
            )

            n_mean_correlation_label = corr(
                n_mean_latent_modality1, n_mean_latent_modality2
            )
            n_var_correlation_label = corr(
                n_var_latent_modality1, n_var_latent_modality2
            )
            n_normalized_correlation_label = corr(
                n_normalized_latent_modality1, n_normalized_latent_modality2
            )

            fig = mpl.figure.Figure(figsize=(15, 15), dpi=96)
            ax = fig.subplots()
            ax.matshow(mean_correlation_label, cmap=mpl.cm.Blues)
            ax.set_title(f"Mean correlations {modality1} - {modality2}, label={label}")
            fig.savefig(f"tmpfigs/mean_corr_{modality1}_{modality1}_{label}.png")

            fig = mpl.figure.Figure(figsize=(15, 15), dpi=96)
            ax = fig.subplots()
            ax.matshow(var_correlation_label, cmap=mpl.cm.Blues)
            ax.set_title(f"Var correlations {modality1} - {modality2}, label={label}")
            fig.savefig(f"tmpfigs/var_corr_{modality1}_{modality1}_{label}.png")

            fig = mpl.figure.Figure(figsize=(15, 15), dpi=96)
            ax = fig.subplots()
            ax.matshow(normalized_correlation_label, cmap=mpl.cm.Blues)
            ax.set_title(
                f"Normalized mean correlations {modality1} - {modality2}, label={label}"
            )
            fig.savefig(f"tmpfigs/normalized_corr_{modality1}_{modality1}_{label}.png")

            fig = mpl.figure.Figure(figsize=(15, 15), dpi=96)
            ax = fig.subplots()
            ax.matshow(n_mean_correlation_label, cmap=mpl.cm.Blues)
            ax.set_title(
                f"N Mean correlations {modality1} - {modality2}, label={label}"
            )
            fig.savefig(f"tmpfigs/n_mean_corr_{modality1}_{modality1}_{label}.png")

            fig = mpl.figure.Figure(figsize=(15, 15), dpi=96)
            ax = fig.subplots()
            ax.matshow(n_var_correlation_label, cmap=mpl.cm.Blues)
            ax.set_title(f"N Var correlations {modality1} - {modality2}, label={label}")
            fig.savefig(f"tmpfigs/n_var_corr_{modality1}_{modality1}_{label}.png")

            fig = mpl.figure.Figure(figsize=(15, 15), dpi=96)
            ax = fig.subplots()
            ax.matshow(n_normalized_correlation_label, cmap=mpl.cm.Blues)
            ax.set_title(
                f"N Normalized mean correlations {modality1} - {modality2}, label={label}"
            )
            fig.savefig(
                f"tmpfigs/n_normalized_corr_{modality1}_{modality1}_{label}.png"
            )

    return None, None
