from typing import Dict, List, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.modalities import Modality
from utils.save_samples import save_generated_samples_singlegroup


def classify_cond_gen_samples(
    shared_attributes_modalities_indices: List[List[int]],
    classifiers_dict: Dict[str, nn.Module],
    modalities_indices: List[int],
    modalities: List[Modality],
    attributes: List[List[str]],
    generated_samples_dict: Dict[str, torch.Tensor],
    labels_dict: Dict[str, np.array],
    eval_fn: Callable,
):
    classifiers = list()
    generated_samples = list()
    attributes_labels = list()
    attributes_names = list()

    for modality_idx in modalities_indices:
        modality_name = modalities[modality_idx].name
        classifiers.append(classifiers_dict[modality_name])
        generated_samples.append(generated_samples_dict[modality_name])
        attributes_labels.append(labels_dict[modality_name])

    eval_attributes = dict()
    for attribute_modalities_indices in shared_attributes_modalities_indices:
        attribute_name = attributes[modalities_indices[0]][
            attribute_modalities_indices[0]
        ]
        if attribute_name not in eval_attributes:
            eval_attributes[attribute_name] = dict()

        for (
            modality_idx,
            modality_attribute_idx,
            classifier,
            generated_sample,
            attribute_labels,
        ) in zip(
            modalities_indices,
            attribute_modalities_indices,
            classifiers,
            generated_samples,
            attributes_labels,
        ):
            attribute_predictions_features = classifier(generated_sample)
            attribute_predictions = list()
            for prediction in attribute_predictions_features:
                if prediction.shape[-1] == 1:
                    attribute_predictions.append((prediction >= 0.5).cpu().data.numpy()[:, None])
                else:
                    attribute_predictions.append(np.argmax(prediction.cpu().data.numpy(), axis=-1)[:, None])

            attribute_prediction = np.concatenate(attribute_predictions, axis=-1)
            eval_attributes[attribute_name][modalities[modality_idx].name] = eval_fn(
                attribute_prediction, attribute_labels, index=modality_attribute_idx
            )

    return eval_attributes


def calculate_coherence(exp, samples):
    classifiers = exp.classifiers
    modalities = exp.modalities

    # The subset of all modalities is always the last one.
    shared_attributes = exp.subsets_shared_attributes_modalities_indices[-1]
    attributes = exp.attributes
    batch_size = exp.flags.batch_size

    # TODO: make work for num samples NOT EQUAL to batch_size
    modalities_all_attributes_predictions = list()
    for modality in modalities:
        modalities_all_attributes_predictions.append(
            classifiers[modality.name](samples[modality.name])
        )

    attributes_coherences: Dict[str, float] = dict()

    for shared_attribute_index, attributes_modality_indices in enumerate(
        shared_attributes
    ):
        modalities_predictions = np.zeros((len(modalities), batch_size))
        attribute_name = attributes[0][attributes_modality_indices[0]]
        for modality_idx, modality_attribute_index in enumerate(
            attributes_modality_indices
        ):
            if modalities_all_attributes_predictions[modality_idx][modality_attribute_index].shape[-1] > 1:
                modality_attribute_prediction = np.argmax(
                    modalities_all_attributes_predictions[modality_idx][
                        modality_attribute_index
                    ]
                    .cpu()
                    .data.numpy(),
                    axis=-1,
                ).astype(int)
            else:
                modality_attribute_prediction = (modalities_all_attributes_predictions[modality_idx][modality_attribute_index] >= 0.5).cpu().data.numpy().astype(int)

            modalities_predictions[modality_idx, :] = modality_attribute_prediction

        matching_predictions = np.all(
            modalities_predictions == modalities_predictions[0, :], axis=0
        )
        coherence_attribute = np.sum(matching_predictions.astype(int)) / float(
            batch_size
        )
        attributes_coherences[attribute_name] = coherence_attribute

    return attributes_coherences


def test_generation(epoch, exp, data_loader: torch.utils.data.DataLoader):
    modalities = exp.modalities
    mm_vae = exp.mm_vae
    subsets = exp.subsets
    labels = exp.labels
    dataset_test = exp.dataset_test
    batch_size = exp.flags.batch_size
    num_samples_fid = exp.flags.num_samples_fid
    device = exp.flags.device
    modalities_subsets_indices = exp.modalities_subsets_indices
    modalities_subsets_names = exp.modalities_subsets_names
    subsets_attributes_indices = exp.subsets_shared_attributes_modalities_indices
    classifiers = exp.classifiers
    attributes = exp.attributes
    eval_fn = exp.eval_label

    gen_perf = {"cond": dict(), "random": dict()}

    full_set = modalities_subsets_indices[-1]
    shared_attributes_indices = subsets_attributes_indices[-1]
    shared_attributes_names = [attributes[-1][attribute_indices[0]] for attribute_indices in shared_attributes_indices]

    for subset_name in modalities_subsets_names:
        if subset_name == '':
            continue

        for attribute_name in shared_attributes_names:
            if attribute_name not in gen_perf["cond"]:
                gen_perf["cond"][attribute_name] = dict()
            if attribute_name not in gen_perf["random"]:
                gen_perf["random"][attribute_name] = list()

            if subset_name not in gen_perf["cond"][attribute_name]:
                gen_perf["cond"][attribute_name][subset_name] = dict()

            for modality in modalities:
                if modality.name not in gen_perf["cond"][attribute_name][subset_name]:
                    gen_perf["cond"][attribute_name][subset_name][modality.name] = list()

    num_batches_epoch = int(len(dataset_test) / float(batch_size))
    cnt_s = 0
    for iteration, batch in enumerate(data_loader):
        batch_data = batch[0]
        batch_label = batch[1]
        rand_gen = mm_vae.generate()
        coherence_random = calculate_coherence(exp, rand_gen)
        for attribute_name, coherence in coherence_random.items():
            gen_perf["random"][attribute_name].append(coherence)

        if (batch_size * iteration) < num_samples_fid:
            save_generated_samples_singlegroup(exp, iteration, "random", rand_gen)
            save_generated_samples_singlegroup(exp, iteration, "real", batch_data)
        for modality_name, modality_batch in batch_data.items():
            batch_data[modality_name] = modality_batch.to(device)
        inferred = mm_vae.inference(batch_data)
        lr_subsets = inferred["subsets"]
        cg = mm_vae.cond_generation(lr_subsets)

        for (subset_name, subset, subset_shared_attributes_modalities_indices) in zip(
            modalities_subsets_names,
            modalities_subsets_indices,
            subsets_attributes_indices,
        ):
            if len(subset) == 0:
                continue
            subset_samples = cg[subset_name]
            classified_subset = classify_cond_gen_samples(
                shared_attributes_modalities_indices=shared_attributes_indices,
                classifiers_dict=classifiers,
                modalities_indices=full_set,
                modalities=modalities,
                attributes=attributes,
                generated_samples_dict=cg[subset_name],
                labels_dict=batch_label,
                eval_fn=eval_fn,
            )

            for attribute_name, classified_attribute in classified_subset.items():
                for modality_name, classified_modality in classified_attribute.items():
                    gen_perf["cond"][attribute_name][subset_name][modality_name].append(classified_modality)


            if (batch_size * iteration) < num_samples_fid:
                save_generated_samples_singlegroup(
                    exp, iteration, subset_name, subset_samples
                )

    for (subset_name, subset, subset_shared_attributes_modalities_indices) in zip(
        modalities_subsets_names,
        modalities_subsets_indices,
        subsets_attributes_indices,
    ):
        if len(subset) == 0:
            continue

        for attribute_name, gen_perf_attribute in gen_perf["cond"].items():
            for modality_name, gen_perf_attribute_modality in gen_perf_attribute[subset_name].items():
                perf = exp.mean_eval_metric(gen_perf_attribute_modality)
                gen_perf["cond"][attribute_name][subset_name][modality_name] = perf

        for attribute_modalities_indices in subset_shared_attributes_modalities_indices:
            for modality_idx, attribute_idx in zip(
                subset, attribute_modalities_indices
            ):
                modality_name = modalities[modality_idx].name
                attribute_name = attributes[modality_idx][attribute_idx]

                perf = exp.mean_eval_metric(
                    gen_perf["cond"][attribute_name][subset_name][modality_name]
                )
                gen_perf["cond"][attribute_name][subset_name][modality_name] = perf

    for attribute_name, attribute_perf in gen_perf["random"].items():
        gen_perf["random"][attribute_name] = exp.mean_eval_metric(attribute_perf)

    # Compute leave-one-out coherence
    gen_perf["loo"] = dict()
    num_modalities = len(modalities)
    aggregated_perfs = list()
    for attribute_name, gen_perf_attribute in gen_perf["cond"].items():
        aggregated_perfs_attribute = list()
        for subset_name, subset in zip(modalities_subsets_names, modalities_subsets_indices):
            if len(subset) != (num_modalities - 1):
                continue

            missing_modality_idx = -1
            for modality_idx in range(num_modalities):
                if modality_idx not in subset:
                    missing_modality_idx = modality_idx
                    break

            missing_modality_name = modalities[missing_modality_idx].name

            if subset_name not in gen_perf_attribute or missing_modality_name not in gen_perf_attribute[subset_name]:
                continue

            aggregated_perfs_attribute.append(gen_perf_attribute[subset_name][missing_modality_name])
            aggregated_perfs.append(gen_perf_attribute[subset_name][missing_modality_name])

        gen_perf["loo"][attribute_name] = np.mean(aggregated_perfs_attribute)

    gen_perf["loo"]["all_attributes"] = np.mean(aggregated_perfs)
    return gen_perf
