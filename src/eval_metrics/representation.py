from typing import List, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_classifier_latent_representation_all_subsets(exp, data_loader):
    mm_vae = exp.mm_vae;
    mm_vae.eval();
    dataset_train = exp.dataset_train
    batch_size = exp.flags.batch_size;
    class_dim = exp.flags.class_dim;
    device = exp.flags.device
    attributes = exp.attributes
    subsets = exp.subsets;
    subsets_names = exp.modalities_subsets_names
    subset_name_to_index = exp.modalities_subsets_name_to_index
    subsets_attributes_indices = exp.subsets_shared_attributes_modalities_indices
    modalities_subsets_indices = exp.modalities_subsets_indices
    modalities = exp.modalities
    n_train_samples = exp.flags.num_training_samples_lr

    n_samples = len(dataset_train)
    num_batches_epoch = int(n_samples/float(batch_size))

    data_train = list()
    attributes_labels: List[np.array] = list()
    subsets_first_modality_subsets_indices: List[np.ndarray] = list()
    subsets_first_modality_name: List[str] = list()
    subsets_shared_attributes_names: List[List[str]] = list()
    for subset_idx, (subset_name, subset_modalities, subset_attributes) in enumerate(zip(subsets_names, modalities_subsets_indices, subsets_attributes_indices)):
        if subset_name != '':
            data_train.append(np.zeros((n_samples, class_dim)))
            subset_label_indices = np.array(list(map(lambda x: x[0], subset_attributes)))
            attributes_labels.append(np.zeros((n_samples, len(subset_attributes))))
            subsets_first_modality_subsets_indices.append(subset_label_indices)
            subsets_first_modality_name.append(modalities[subset_modalities[0]].name)
            subsets_shared_attributes_names.append([attributes[subset_modalities[0]][attribute_indices[0]] for attribute_indices in subset_attributes])
        else:
            data_train.append(np.array([]))
            attributes_labels.append(np.array([]))
            subsets_first_modality_subsets_indices.append(np.array([]))
            subsets_first_modality_name.append("")
            subsets_shared_attributes_names.append(list())

    for iteration, (batch_data, batch_labels) in enumerate(data_loader):
        for modality_name, modality_data in batch_data.items():
            batch_data[modality_name] = modality_data.to(device)

        inferred = mm_vae.inference(batch_data)
        latent_representations_subset = inferred["subsets"]

        for subset_idx, (subset_name, first_modality_name, first_modality_attribute_indices) in enumerate(zip(subsets_names, subsets_first_modality_name, subsets_first_modality_subsets_indices)):
            if subset_name == '':
                continue
            mean, log_var = latent_representations_subset[subset_name]
            data_train[subset_idx][(iteration*batch_size):((iteration+1)*batch_size), :] = mean.cpu().data.numpy()

            attributes_labels[subset_idx][(iteration*batch_size):((iteration+1)*batch_size), :] = np.reshape(
                batch_labels[first_modality_name][:, first_modality_attribute_indices], (batch_size, len(first_modality_attribute_indices))
            )

    training_sample_indices = np.random.choice(np.arange(0, n_samples), size=n_train_samples, replace=False)

    for subset_idx in range(len(data_train)):
        if len(data_train[subset_idx]) > 0:
            data_train[subset_idx] = data_train[subset_idx][training_sample_indices]
            attributes_labels[subset_idx] = attributes_labels[subset_idx][training_sample_indices]

    classifier_latent_representation = train_classifier_latent_representation(exp, data_train, attributes_labels, subsets_attribute_names=subsets_shared_attributes_names, subset_names=subsets_names)
    return classifier_latent_representation


def test_classifier_latent_representation_all_subsets(epoch, classifiers_latent_representations_subsets, exp, data_loader):
    mm_vae = exp.mm_vae;
    mm_vae.eval();
    subsets = exp.subsets;
    modalities = exp.modalities
    modalities_subsets_indices = exp.modalities_subsets_indices
    modalities_subsets_names = exp.modalities_subsets_names
    subsets_attributes_indices = exp.subsets_shared_attributes_modalities_indices
    attributes = exp.attributes
    batch_size = exp.flags.batch_size
    device = exp.flags.device

    latent_representation_eval = dict()

    subsets_first_modality_subsets_indices: List[np.ndarray] = list()
    subsets_first_modality_name: List[str] = list()
    subsets_shared_attributes_names: List[List[str]] = list()
    for subset_name, subset_modalities, subset_attributes in zip(modalities_subsets_names, modalities_subsets_indices, subsets_attributes_indices):
        if subset_name != '':
            subset_label_indices = np.array(list(map(lambda x: x[0], subset_attributes)))
            subsets_first_modality_subsets_indices.append(subset_label_indices)
            subsets_first_modality_name.append(modalities[subset_modalities[0]].name)
            subsets_shared_attributes_names.append([attributes[subset_modalities[0]][attribute_indices[0]] for attribute_indices in subset_attributes])
        else:
            subsets_first_modality_subsets_indices.append(np.array([]))
            subsets_first_modality_name.append("")
            subsets_shared_attributes_names.append(list())

    for subset_name, subset_shared_attributes, modalities_indices in zip(modalities_subsets_names, subsets_attributes_indices, modalities_subsets_indices):
        if subset_name == '':
            continue

        first_modality = modalities_indices[0]
        for modality_attribute_index in subset_shared_attributes:
            attribute_name = attributes[first_modality][modality_attribute_index[0]]
            if attribute_name not in latent_representation_eval:
                latent_representation_eval[attribute_name] = dict()

            latent_representation_eval[attribute_name][subset_name] = list()

    for iteration, (batch_data, batch_labels) in enumerate(data_loader):
        for modality_name, modality_batch in batch_data.items():
            batch_data[modality_name] = modality_batch.to(device)

        inferred = mm_vae.inference(batch_data)
        latent_representations_subsets = inferred["subsets"]

        data_test = list()
        labels_test = list()

        for subset_name, first_modality_name, first_modality_attribute_indices in zip(modalities_subsets_names, subsets_first_modality_name, subsets_first_modality_subsets_indices):
            if subset_name == '':
                data_test.append(np.array([]))
                labels_test.append(np.array([]))
                continue
            data_test.append(latent_representations_subsets[subset_name][0].cpu().data.numpy())
            labels_test.append(
                np.reshape(batch_labels[first_modality_name][:, first_modality_attribute_indices], (batch_size, len(first_modality_attribute_indices)))
            )

        batch_eval = classify_latent_representations(
            exp,
            epoch,
            classifiers_latent_representations_subsets,
            data_test,
            labels_test,
            subset_attribute_names=subsets_shared_attributes_names,
            subset_names=modalities_subsets_names,
        )

        for attribute_name, attribute_eval in batch_eval.items():
            for subset_name, subset_eval in attribute_eval.items():
                if subset_name == '':
                    continue
                latent_representation_eval[attribute_name][subset_name].append(subset_eval)

    for attribute_name, attribute_eval in latent_representation_eval.items():
        for subset_name, subset_eval in attribute_eval.items():
            if subset_name == '':
                continue
            latent_representation_eval[attribute_name][subset_name] = exp.mean_eval_metric(latent_representation_eval[attribute_name][subset_name])

    return latent_representation_eval


def classify_latent_representations(exp, epoch, classifiers_latent_representation, data, labels, subset_attribute_names: List[List[str]], subset_names: List[str]):
    eval_all_attributes: Dict[str, Dict[str, float]] = dict()

    for data_subset, labels_subset, subset_name, attribute_names in zip(data, labels, subset_names, subset_attribute_names):
        if subset_name == '':
            continue
        for attribute_idx, attribute_name in enumerate(attribute_names):
            ground_truth = labels_subset[:, attribute_idx]
            classifier = classifiers_latent_representation[attribute_name][subset_name]
            predicted_labels = classifier.predict(data_subset)
            eval_subset = exp.eval_metric(ground_truth, predicted_labels)

            if attribute_name not in eval_all_attributes:
                eval_all_attributes[attribute_name] = dict()

            eval_all_attributes[attribute_name][subset_name] = eval_subset
    return eval_all_attributes


def train_classifier_latent_representation(exp, data, labels, subsets_attribute_names: List[List[str]], subset_names: List[str]):
    latent_representation_classifiers: Dict[str, Dict[str, LogisticRegression]] = dict();
    for data_subset, labels_subset, subset_name, attribute_names in zip(data, labels, subset_names, subsets_attribute_names):
        if subset_name == '':
            continue
        for attribute_idx, attribute_name in enumerate(attribute_names):
            ground_truth = labels_subset[:, attribute_idx]
            classifier_latent_representation_subset = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
            classifier_latent_representation_subset.fit(data_subset, ground_truth.ravel())

            if attribute_name not in latent_representation_classifiers:
                latent_representation_classifiers[attribute_name] = dict()

            latent_representation_classifiers[attribute_name][subset_name] = classifier_latent_representation_subset
    return latent_representation_classifiers

