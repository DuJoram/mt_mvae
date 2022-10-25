import argparse
import os.path
from typing import List, Dict, Optional

import numpy as np
import torch.utils.data

from data.modalities import Modality, DSprite
from . import MultimodalDataset

import tqdm


class DSpritesDataset(torch.utils.data.Dataset):
    """
    Unimodal dSprites dataset for pre-training classifiers
    """

    def __init__(
        self,
        dsprites_archive_path: str,
        train: bool = True,
    ):

        test_size = 50_000
        if not os.path.exists(dsprites_archive_path):
            raise FileNotFoundError(
                f"dSprites archive '{dsprites_archive_path}' not found!"
            )

        dsprites_data = np.load(
            dsprites_archive_path, allow_pickle=True, encoding="latin1"
        )
        metadata = dsprites_data["metadata"][()]

        self._images = dsprites_data["imgs"]
        self._latents_values = dsprites_data["latents_values"]
        self._latents_classes = dsprites_data["latents_classes"]
        self._latents_sizes = metadata["latents_sizes"]
        self._latents_names = np.array(metadata["latents_names"])
        self._latents_possible_values = metadata["latents_possible_values"]

        indices = np.arange(len(self._images))

        valid_indices_square = np.repeat(
            indices[
                (self._latents_classes[:, 1] == 0) & (self._latents_classes[:, 3] < 10)
            ],
            4,
        )
        valid_indices_ellipse = np.repeat(
            indices[
                (self._latents_classes[:, 1] == 1) & (self._latents_classes[:, 3] < 20)
            ],
            2,
        )
        valid_indices_heart = indices[self._latents_classes[:, 1] == 2]

        valid_indices = np.concatenate(
            (valid_indices_square, valid_indices_ellipse, valid_indices_heart)
        )

        rng = np.random.default_rng(seed=42)

        permutation = rng.permutation(valid_indices)

        if train:
            data_indices = np.sort(permutation[:-test_size])
            self._len = len(self._images) - test_size
        else:
            data_indices = np.sort(permutation[-test_size:])
            self._len = test_size

        self._indices = data_indices
        self._images = self._images[data_indices]
        self._latents_classes = self._latents_classes[data_indices]
        self._latents_values = self._latents_values[data_indices]

    def __getitem__(self, idx):
        image = self._images[idx, None].astype(np.float32)
        class_labels = self._latents_classes[idx]
        class_labels = tuple(class_labels[1:])

        return image, class_labels

    def __len__(self):
        return self._len

    def get_latents_names(self) -> np.array:
        return self._latents_names[1:]

    def get_latents_sizes(self) -> np.array:
        return self._latents_sizes[1:]

    def get_latents_values(self) -> np.array:
        return self._latents_values[1:]

    def get_latents_possible_values(self) -> Dict[str, np.array]:
        return self._latents_possible_values


class MdSpritesDataset(MultimodalDataset):
    """
    Multimodal dSprites Dataset
    """

    def __init__(
        self,
        dsprites_archive_path: str,
        num_modalities: int = 5,
        decoder_distribution: str = "bernoulli",
        shared_attributes: Optional[List[str]] = None,
        num_training_samples: int = 100_000,
        num_test_samples: int = 10_000,
        filter_symmetries: bool = True,
        train: bool = True,
        seed: int = 42
    ):
        if not os.path.exists(dsprites_archive_path):
            raise FileNotFoundError

        if shared_attributes is None:
            shared_attributes = ["shape", "scale", "posX"]

        num_samples = num_training_samples + num_test_samples

        dsprites_data = np.load(dsprites_archive_path, allow_pickle=True, encoding='latin1')
        images = dsprites_data["imgs"]
        dsprites_metadata = dsprites_data["metadata"][()]
        attributes_classes = dsprites_data["latents_classes"]
        attributes_names = np.array(dsprites_metadata["latents_names"])
        attributes_possible_values = dsprites_metadata["latents_possible_values"]
        attributes_sizes = dsprites_metadata["latents_sizes"]
        attributes_values = dsprites_data["latents_values"]

        # Dropping color, as it is always the same
        attributes_classes = attributes_classes[:, 1:]
        attributes_names = attributes_names[1:]
        del attributes_possible_values["color"]
        attributes_sizes = attributes_sizes[1:]
        attributes_values = attributes_values[:, 1:]

        rng = np.random.default_rng(seed=seed)

        shape_idx = np.argwhere(attributes_names == "shape").squeeze().item()
        orientation_idx = np.argwhere(attributes_names == "orientation").squeeze().item()

        for shared_attribute in shared_attributes:
            if shared_attribute not in attributes_names:
                raise ValueError(f"Invalid shared attribute: {shared_attribute} is not one of the recognized attribtues. Must be one of {attributes_names}")

        # Squares look the same when rotating by 90 degrees. Hence only the first 10 rotations
        # (2pi is done in 40 steps) look unique. So we replace all non-uniques with the first
        # rotation angle. Same for the ellipse.
        if filter_symmetries:
            valid_square_indices = (attributes_classes[:, shape_idx] == 0) & (attributes_classes[:, orientation_idx] < 10)
            for lower, upper in [(10, 20), (20, 30), (30, 40)]:
                invalid_indices = (attributes_classes[:, shape_idx] == 0) & (lower <= attributes_classes[:, orientation_idx]) & (attributes_classes[:, orientation_idx] < upper)

                images[invalid_indices] = images[valid_square_indices]
                attributes_classes[invalid_indices] = attributes_classes[valid_square_indices]
                attributes_values[invalid_indices] = attributes_values[valid_square_indices]

            valid_ellipse_indices = (attributes_classes[:, shape_idx] == 1) & (attributes_classes[:, orientation_idx] < 20)
            invalid_ellipse_indices = (attributes_classes[:, shape_idx] == 1) & (attributes_classes[:, orientation_idx] >= 20)

            images[invalid_ellipse_indices] = images[valid_ellipse_indices]
            attributes_classes[invalid_ellipse_indices] = attributes_classes[valid_ellipse_indices]
            attributes_values[invalid_ellipse_indices] = attributes_classes[valid_ellipse_indices]

        shared_sizes = list()
        shared_indices = list()
        specific_sizes = list()
        specific_indices = list()

        self._shared_attributes_indices = shared_indices

        for attribute_idx, (attribute_name, attribute_size) in enumerate(zip(attributes_names, attributes_sizes)):
            if attribute_name in shared_attributes:
                shared_sizes.append(attribute_size)
                shared_indices.append(attribute_idx)
            else:
                specific_sizes.append(attribute_size)
                specific_indices.append(attribute_idx)

        def sizes_to_base(sizes: List[int]) -> np.array:
            return np.append(np.array(sizes)[::-1].cumprod()[::-1][1:], 1).astype(int)

        def value_indices_to_global_index(indices: np.array, base: np.array):
            if indices.shape[-1] != base.shape[0]:
                indices = indices.T
            return np.dot(indices, base).flatten().astype(int)

        def global_index_to_value_indices(index: int, base: np.array):
            result = []
            remainder = index
            for unit in base:
                item_result, remainder = np.divmod(remainder, unit)
                result.append(item_result)

            return np.array(result).astype(int).T

        shared_base = sizes_to_base(shared_sizes)
        specific_base = sizes_to_base(specific_sizes)
        base = sizes_to_base(attributes_sizes)

        # To ensure uniqueness of samples:
        # Make sure that the first modality only contains unique samples. The others follow.

        if num_samples > len(images):
            raise ValueError(f"ERROR: Only support a combined number of samples at most the size of the dSprites dataset!: {num_training_samples} + {num_test_samples} > {len(images)}")

        data_indices = np.arange(0, len(images))
        shared_chosen_indices = rng.choice(data_indices, size=num_samples, replace=False)
        shared_chosen_classes = attributes_classes[shared_chosen_indices]

        modalities_chosen_indices = np.zeros((num_samples, num_modalities))


        shared_attributes_classes = attributes_classes[:, shared_indices]
        shared_chosen_classes = shared_chosen_classes[:, shared_indices]

        self._size = num_training_samples if train else num_test_samples

        self._images = list()
        self._labels = list()
        for modality_idx in range(num_modalities):
            chosen_based_index = rng.choice(np.arange(0, specific_base[0]), size=num_samples)
            chosen_specific_classes = global_index_to_value_indices(chosen_based_index, specific_base)

            chosen_classes = np.zeros((num_samples, len(attributes_names)))
            chosen_classes[:, shared_indices] = shared_chosen_classes
            for attribute_idx in specific_indices:
                chosen_classes[:, attribute_idx] = rng.choice(np.arange(0, attributes_sizes[attribute_idx]), size=num_samples)

            chosen_indices = value_indices_to_global_index(chosen_classes, base)
            if train:
                chosen_indices = chosen_indices[:num_training_samples]
            else:
                chosen_indices = chosen_indices[-num_test_samples:]

            self._images.append(images[chosen_indices])
            self._labels.append(attributes_classes[chosen_indices])


        self._modalities = [DSprite(f"m{idx}", likelihood=decoder_distribution) for idx in range(num_modalities)]

        self._attributes: List[List[str]] = [attributes_names.tolist()] * num_modalities
        self._attributes_sizes = num_modalities*[attributes_sizes]

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        sample_images = dict()
        sample_latents_classes = dict()

        for modality, images, labels in zip(self._modalities, self._images, self._labels):
            sample_images[modality.name] = torch.tensor(images[index], dtype=torch.float).unsqueeze(0)
            sample_latents_classes[modality.name] = torch.tensor(labels[index], dtype=torch.float)

        return sample_images, sample_latents_classes

    def get_modalities(self) -> List[Modality]:
        return self._modalities

    def get_attributes(self) -> List[str]:
        return self._attributes

    def get_attributes_sizes(self) -> List[List[int]]:
        return self._attributes_sizes

    def get_modalities_subset_shared_attributes_indices(self, subset: List[int]) -> List[List[int]]:
        return [len(subset)*[attribute_idx] for attribute_idx in self._shared_attributes_indices]

