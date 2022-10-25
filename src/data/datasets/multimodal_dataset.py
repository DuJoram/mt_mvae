import abc
from typing import List

from torch.utils.data import Dataset

from ..modalities import Modality


class MultimodalDataset(abc.ABC, Dataset):
    @abc.abstractmethod
    def get_modalities(self) -> List[Modality]:
        pass

    @abc.abstractmethod
    def get_attributes(self) -> List[List[str]]:
        pass

    @abc.abstractmethod
    def get_attributes_sizes(self) -> List[List[int]]:
        pass

    @abc.abstractmethod
    def get_modalities_subset_shared_attributes_indices(self, subset: List[int]) -> List[List[int]]:
        pass

