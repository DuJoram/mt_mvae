from dataclasses import dataclass
from typing import List, Tuple

import matplotlib as mpl
import numpy as np
import torch
import tqdm

from experiments import Experiment

mpl.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader

from data.modalities import Modality
from models import MultimodalVAE


@dataclass
class DisentanglementMetricHigginsResult:
    confusion_matrix: np.ndarray
    confusion_matrix_display: ConfusionMatrixDisplay
    disentanglement: float
    per_attribute_disentanglement: np.ndarray
    attribute_names: List[str]


@torch.no_grad()
def convert_to_disentanglement_dataset(
    dataset: Dataset,
    num_samples: int,
    num_workers: int,
    num_diff_samples: int,
    allowed_subsets: List[int],
    mix_subsets_in_diffs: bool,
    modalities: List[Modality],
    subsets_shared_attributes_modalities_indices: List[List[List[int]]],
    modalities_subsets_indices: List[List[int]],
    attributes_sizes: List[List[int]],
    attributes: List[List[str]],
    modalities_subsets_names: List[str],
    class_dim: int,
    mm_vae: MultimodalVAE,
    seed: int = 42,
) -> Tuple[np.array, np.array]:
    all_shared_attributes = subsets_shared_attributes_modalities_indices[-1]
    attributes = attributes

    rng = np.random.default_rng(seed=seed)

    loader = DataLoader(
        dataset=dataset, num_workers=num_workers, batch_size=len(dataset)
    )
    images, labels = next(iter(loader))

    modality0_shared_attributes_indices = list(
        map(lambda x: x[0], all_shared_attributes)
    )
    modality0_idx = modalities_subsets_indices[-1][0]
    modality0_name = modalities[modalities_subsets_indices[-1][0]].name

    shared_attributes_sizes = attributes_sizes[modality0_idx][
        modality0_shared_attributes_indices
    ]

    labels_shared_attributes = (
        labels[modality0_name][:, modality0_shared_attributes_indices]
        .cpu()
        .numpy()
        .astype(int)
    )

    num_attributes = len(shared_attributes_sizes)

    num_samples -= num_samples % num_attributes

    slice_size = num_samples // num_attributes

    inferred = mm_vae.inference(images)
    latents = np.concatenate(
        [
            inferred["subsets"][subset_name][0][None, :, :]
            for subset_name in modalities_subsets_names[1:]
        ],
        axis=0,
    )

    data = np.empty((num_samples, class_dim))

    targets = np.empty((num_samples,))

    for attribute_idx, attribute_size in tqdm.tqdm(
        enumerate(shared_attributes_sizes),
        total=len(shared_attributes_sizes),
        desc="Attribute loop",
    ):
        targets[
            attribute_idx * slice_size : ((attribute_idx + 1) * slice_size)
        ] = attribute_idx

        num_attribute_samples = slice_size * num_diff_samples

        latents_diff_attribute = np.empty((num_attribute_samples, class_dim))

        # The first subset is the empty set, which we filtered. Hence, -1.
        subsets_lhs = rng.choice(allowed_subsets, size=num_attribute_samples) - 1
        subsets_rhs = subsets_lhs

        if mix_subsets_in_diffs:
            subsets_rhs = rng.choice(allowed_subsets, size=num_attribute_samples) - 1

        value_slice_size = num_attribute_samples // attribute_size
        for attribute_value in tqdm.trange(attribute_size, desc="Attribute value loop"):
            slice_start = attribute_value * value_slice_size
            slice_end = slice_start + value_slice_size
            if attribute_value == (attribute_size - 1):
                slice_end = num_attribute_samples

            valid_indices = np.argwhere(
                labels_shared_attributes[:, attribute_idx] == attribute_value
            )
            num_valid_indices = len(valid_indices)

            # Randomly choose pairs of distinct indices:
            # 1. Create matrix of all possible index pairs (N * N matrix)
            # 2. Remove diagonal and shift up lower triangular matrix up by one (=> (N-1) * N matrix)
            # 4. Chose a random element from that matrix and compute its column/row indices.
            # 5. If the row index is greater than or equal to the column index, then that entry was originally in the
            #    lower triangular matrix. The true index is therefore += 1
            chosen_indices_indices = rng.integers(
                0,
                num_valid_indices * (num_valid_indices - 1),
                size=(slice_end - slice_start),
            )
            chosen_indices_lhs, chosen_indices_rhs = np.divmod(
                chosen_indices_indices, num_valid_indices
            )
            chosen_indices_lhs[chosen_indices_lhs >= chosen_indices_rhs] += 1

            latents_lhs = np.take_along_axis(
                latents[:, valid_indices[chosen_indices_lhs].flatten()],
                subsets_lhs[None, slice_start:slice_end, None],
                axis=0,
            )[0]
            latents_rhs = np.take_along_axis(
                latents[:, valid_indices[chosen_indices_rhs].flatten()],
                subsets_rhs[None, slice_start:slice_end, None],
                axis=0,
            )[0]
            latents_diff_attribute[slice_start:slice_end] = np.abs(
                latents_lhs - latents_rhs
            )

        data[attribute_idx * slice_size : (attribute_idx + 1) * slice_size] = np.mean(
            rng.permutation(latents_diff_attribute).reshape(
                num_diff_samples, slice_size, -1
            ),
            axis=0,
        )

    permutation = rng.permutation(np.arange(0, num_samples))
    data_shuffled = data[permutation]
    targets_shuffled = targets[permutation]

    return data_shuffled, targets_shuffled


def train_disentanglement_classifier(
    data: np.ndarray,
    targets: np.ndarray,
    num_redundant_classifiers: int = 1,
    verbose: bool = True,
) -> List[LogisticRegression]:
    classifiers: List[LogisticRegression] = list()

    for idx in tqdm.trange(num_redundant_classifiers, desc="Training classifiers"):
        classifier = LogisticRegression(
            penalty="l2", verbose=int(verbose), max_iter=1_000
        )
        classifier.fit(data, targets)
        classifiers.append(classifier)

    return classifiers


def test_disentanglement_classifier(
    data: np.ndarray,
    targets: np.ndarray,
    classifiers: List[LogisticRegression],
    num_attributes: int,
    attribute_names: List[str],
) -> DisentanglementMetricHigginsResult:
    labels = np.arange(0, num_attributes)
    results: List[DisentanglementMetricHigginsResult] = list()

    accuracies_predictions: List[Tuple[float, np.array]] = list()

    num_classifiers = len(classifiers)
    num_predictions = num_classifiers//2

    for classifier_idx, classifier in enumerate(classifiers):
        prediction = classifier.predict(data)
        accuracy = accuracy_score(y_true=targets, y_pred=prediction)
        accuracies_predictions.append((accuracy, prediction))

    accuracies_predictions = list(sorted(accuracies_predictions, key=lambda x: -x[0]))
    top_half_accuracies_predictions = accuracies_predictions[:num_predictions]

    targets = np.tile(targets, num_predictions)

    predictions = np.concatenate(list(map(lambda x: x[1], top_half_accuracies_predictions)), axis=0)
    disentanglement = np.mean(list(map(lambda x: x[0], top_half_accuracies_predictions)))

    matrix = confusion_matrix(
        y_true=targets, y_pred=predictions, labels=labels, normalize="true"
    )

    confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(
        y_true=targets,
        y_pred=predictions,
        labels=labels,
        normalize="true",
        display_labels=attribute_names,
        cmap="Blues",
        colorbar=True,
    )

    per_attribute_disentanglement = np.diag(matrix)

    result = DisentanglementMetricHigginsResult(
        confusion_matrix=matrix,
        confusion_matrix_display=confusion_matrix_display,
        disentanglement=disentanglement.item(),
        per_attribute_disentanglement=per_attribute_disentanglement,
        attribute_names=attribute_names,
    )

    return result


def disentanglement_metric_higgins(
    experiment: Experiment,
    subsets_sets: List[List[int]],
    attribute_names: List[str],
    num_attributes: int,
    num_workers: int,
    num_diff_samples: int,
    num_training_samples: int,
    num_test_samples: int,
    num_redundant_classifiers: int,
    mix_subsets_in_diffs: bool,
    class_dim: int,
    seed: int = 42,
    verbose: bool = True,
) -> List[DisentanglementMetricHigginsResult]:

    results: List[DisentanglementMetricHigginsResult] = list()

    for allowed_subsets in subsets_sets:
        data_train, labels_train = convert_to_disentanglement_dataset(
            dataset=experiment.dataset_train,
            num_samples=num_training_samples,
            num_workers=num_workers,
            num_diff_samples=num_diff_samples,
            allowed_subsets=allowed_subsets,
            mix_subsets_in_diffs=mix_subsets_in_diffs,
            modalities=experiment.modalities,
            subsets_shared_attributes_modalities_indices=experiment.subsets_shared_attributes_modalities_indices,
            modalities_subsets_indices=experiment.modalities_subsets_indices,
            attributes_sizes=experiment.attributes_sizes,
            attributes=experiment.attributes,
            modalities_subsets_names=experiment.modalities_subsets_names,
            class_dim=class_dim,
            mm_vae=experiment.mm_vae.eval(),
            seed=seed,
        )

        data_test, labels_test = convert_to_disentanglement_dataset(
            dataset=experiment.dataset_test,
            num_samples=num_test_samples,
            num_workers=num_workers,
            num_diff_samples=num_diff_samples,
            allowed_subsets=allowed_subsets,
            mix_subsets_in_diffs=mix_subsets_in_diffs,
            modalities=experiment.modalities,
            subsets_shared_attributes_modalities_indices=experiment.subsets_shared_attributes_modalities_indices,
            modalities_subsets_indices=experiment.modalities_subsets_indices,
            attributes_sizes=experiment.attributes_sizes,
            attributes=experiment.attributes,
            modalities_subsets_names=experiment.modalities_subsets_names,
            class_dim=class_dim,
            mm_vae=experiment.mm_vae.eval(),
            seed=seed,
        )

        classifiers = train_disentanglement_classifier(
            data=data_train,
            targets=labels_train,
            num_redundant_classifiers=num_redundant_classifiers,
            verbose=verbose,
        )
        result = test_disentanglement_classifier(
            data=data_test,
            targets=labels_test,
            classifiers=classifiers,
            num_attributes=num_attributes,
            attribute_names=attribute_names,
        )
        results.append(result)

    return results


def mutual_information_gap() -> torch.Tensor:
    pass
