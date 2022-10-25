import os
from typing import List

import pandas as pd
from PIL import ImageFont
from torch.utils.data import Dataset
import PIL.Image as Image

import torch

from utils import text as text

from .multimodal_dataset import MultimodalDataset
from ..modalities import Modality, CelebaImage
from ..modalities.CelebaText import CelebaText


class CelebaDataset(MultimodalDataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, args, alphabet, partition=0, transform=None):

        self.dir_dataset_base = os.path.join(args.dir_data, args.dataset)

        filename_text = os.path.join(args.dir_text, 'list_attr_text_' + str(args.len_sequence).zfill(3) + '_' + str(args.random_text_ordering) + '_' + str(args.random_text_startindex) + '_celeba.csv');
        filename_partition = os.path.join(self.dir_dataset_base, 'list_eval_partition.csv');
        filename_attributes = os.path.join(self.dir_dataset_base, 'list_attr_celeba.csv')

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition);
        df_attributes = pd.read_csv(filename_attributes);

        self.args = args;
        self.img_dir = os.path.join(self.dir_dataset_base, 'img_align_celeba');
        self.txt_path = filename_text
        self.attributes_path = filename_attributes;
        self.partition_path = filename_partition;

        self.alphabet = alphabet;
        self.img_names = df_text.loc[df_partition['partition'] == partition]['image_id'].values
        self.attributes = df_attributes.loc[df_partition['partition'] == partition];
        self.labels = df_attributes.loc[df_partition['partition'] == partition].values; #atm, i am just using blond_hair as labels
        self.y = df_text.loc[df_partition['partition'] == partition]['text'].values
        self.transform = transform

        plot_image_size = torch.Size((3, 64, 64))
        self._modalities = [CelebaImage(plot_image_size), CelebaText(self.args.len_sequence, self.args.alphabet, plot_image_size, ImageFont.truetype('resources/FreeSerif.ttf', 38))]
        shared_attributes = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
            'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
            'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
            'Mouth_Slightly_Open', 'Mustache',
            'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young'
        ]

        self._attributes: List[List[str]] = len(self._modalities)*[ shared_attributes ]
        self._attributes_sizes: List[List[int]] = len(self._modalities)*[len(shared_attributes)*[1]]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
        text_str = text.one_hot_encode(self.args.len_sequence, self.alphabet, self.y[index])
        label = torch.from_numpy((self.labels[index,1:] > 0).astype(int)).float();
        sample = {"image": img, "text": text_str}
        return sample, {"image": label, "text": text_str}

    def __len__(self):
        return self.y.shape[0]

    def get_text_str(self, index):
        return self.y[index];

    def get_modalities(self) -> List[Modality]:
        return self._modalities

    def get_attributes_sizes(self) -> List[List[int]]:
        return self._attributes_sizes

    def get_attributes(self) -> List[List[str]]:
        return self._attributes

    def get_modalities_subset_shared_attributes_indices(self, subset: List[int]) -> List[List[int]]:
        # All subsets share all attributes
        return [len(subset)*[attribute_idx] for attribute_idx in range(len(self._attributes[0]))]
