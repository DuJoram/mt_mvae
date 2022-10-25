import argparse
import glob
import io
import os
import zipfile
from typing import List, Optional

import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

from data.datasets import MultimodalDataset
from data.modalities import Modality
from data.modalities.CMNIST import CMNIST


class MMNISTDataset(MultimodalDataset):
    """Multimodal MNIST Dataset."""

    def __init__(
        self,
        unimodal_datapaths: Optional[List[str]] = None,
        dataset_archive: Optional[str] = None,
        num_modalities: Optional[int] = None,
        load_test: bool = False,
        transform=None,
        target_transform=None,
    ):
        """
        Args:
            unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                correspond by index. Therefore the numbers of samples of all datapaths should match.
            transform: tranforms on colored MNIST digits.
            target_transform: transforms on labels.
        """
        super().__init__()

        self.transform = transform
        self.target_transform = target_transform

        self.images: np.array = None
        self.labels: np.array = None

        if unimodal_datapaths is not None:
            self.num_modalities = len(unimodal_datapaths)
            self.unimodal_datapaths = unimodal_datapaths

            # save all paths to individual files
            self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
            for dp in unimodal_datapaths:
                files = glob.glob(os.path.join(dp, "*.png"))
                self.file_paths[dp] = files
            # assert that each modality has the same number of images
            num_files = len(self.file_paths[dp])
            for files in self.file_paths.values():
                assert len(files) == num_files
            self.num_files = num_files
        else:
            if dataset_archive is None:
                raise ValueError(
                    "Must provide either unimodal data paths or dataset archive (and num modalities)"
                )
            if num_modalities is None:
                raise ValueError(
                    "Must provide either unimodal data paths or num modalities (and dataset archive)"
                )

            self.num_modalities = num_modalities

            self.folder = "test" if load_test else "train"
            self.file_paths = [list() for _ in range(self.num_modalities)]

            with zipfile.ZipFile(dataset_archive, "r") as archive:

                for file in archive.filelist:
                    if (
                        not file.filename.endswith(".png")
                        or file.filename.split("/")[1] != self.folder
                    ):
                        continue
                    modality_idx = int(file.filename.split("/")[2][-1])
                    if modality_idx >= self.num_modalities:
                        continue

                    self.file_paths[modality_idx].append(file.filename)

                for modality_idx, images in enumerate(self.file_paths):
                    self.file_paths[modality_idx] = sorted(
                        images, key=lambda x: int(x.split("/")[-1].split(".")[0])
                    )
                self.num_files = len(self.file_paths[0])

                self.images = np.zeros(
                    (self.num_files, self.num_modalities, 28, 28, 3),
                    dtype=np.uint8,
                )
                self.labels = np.zeros(
                    (self.num_files, self.num_modalities), dtype=np.float32
                )

                for idx in tqdm.trange(self.num_files, desc="Loading images"):
                    for modality_idx in range(self.num_modalities):
                        file_name = self.file_paths[modality_idx][idx]
                        label = int(file_name.split(".")[-2])

                        self.images[idx, modality_idx] = Image.open(
                            io.BytesIO(archive.read(file_name))
                        )
                        self.labels[idx, modality_idx] = label

        self._modalities = [
            CMNIST(name=f"m{idx}") for idx in range(self.num_modalities)
        ]

        shared_attributes = ["digit"]
        self._attributes: List[List[str]] = [
            shared_attributes for _ in self._modalities
        ]
        self._attributes_sizes: List[List[int]] = len(self._modalities) * [[1]]

    @staticmethod
    def _create_mmnist_dataset(savepath, backgroundimagepath, num_modalities, train):
        """Created the Multimodal MNIST Dataset under 'savepath' given a directory of background images.

        Args:
            savepath (str): path to directory that the dataset will be written to. Will be created if it does not
                exist.
            backgroundimagepath (str): path to a directory filled with background images. One background images is
                used per modality.
            num_modalities (int): number of modalities to create.
            train (bool): create the dataset based on MNIST training (True) or test data (False).

        """

        # load MNIST data
        mnist = datasets.MNIST("/tmp", train=train, download=True, transform=None)

        # load background images
        background_filepaths = sorted(
            glob.glob(os.path.join(backgroundimagepath, "*.jpg"))
        )  # TODO: handle more filetypes
        print("\nbackground_filepaths:\n", background_filepaths, "\n")
        if num_modalities > len(background_filepaths):
            raise ValueError(
                "Number of background images must be larger or equal to number of modalities"
            )
        background_images = [Image.open(fp) for fp in background_filepaths]

        # create the folder structure: savepath/m{1..num_modalities}
        for m in range(num_modalities):
            unimodal_path = os.path.join(savepath, "m%d" % m)
            if not os.path.exists(unimodal_path):
                os.makedirs(unimodal_path)
                print("Created directory", unimodal_path)

        # create random pairing of images with the same digit label, add background image, and save to disk
        cnt = 0
        for digit in range(10):
            ixs = (mnist.targets == digit).nonzero()
            for m in range(num_modalities):
                ixs_perm = ixs[
                    torch.randperm(len(ixs))
                ]  # one permutation per modality and digit label
                for i, ix in enumerate(ixs_perm):
                    # add background image
                    new_img = MMNISTDataset._add_background_image(
                        background_images[m], mnist.data[ix]
                    )
                    # save as png
                    filepath = os.path.join(savepath, "m%d/%d.%d.png" % (m, i, digit))
                    save_image(new_img, filepath)
                    # log the progress
                    cnt += 1
                    if cnt % 10000 == 0:
                        print(
                            "Saved %d/%d images to %s"
                            % (cnt, len(mnist) * num_modalities, savepath)
                        )
        assert cnt == len(mnist) * num_modalities

    @staticmethod
    def _add_background_image(
        background_image_pil, mnist_image_tensor, change_colors=False
    ):

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 128).type(
            torch.bool
        )  # NOTE: mnist is _not_ normalized to [0, 1]

        # squeeze away color channel
        if img_binarized.ndimension() == 2:
            pass
        elif img_binarized.ndimension() == 3:
            img_binarized = img_binarized.squeeze(0)
        else:
            raise ValueError(
                "Unexpected dimensionality of MNIST image:", img_binarized.shape
            )

        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 28)
        y_c = np.random.randint(0, background_image_pil.size[1] - 28)
        new_img = background_image_pil.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img)
        if change_colors:  # Change color distribution
            for j in range(3):
                new_img[:, :, j] = (new_img[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        new_img[:, img_binarized] = 1 - new_img[:, img_binarized]

        return new_img

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        # transforms
        images = dict()
        labels = dict()
        if self.images is None or self.labels is None:
            for modality_idx, data_path in enumerate(self.unimodal_datapaths):
                file = self.file_paths[data_path][index]

                images[f"m{modality_idx}"] = self.transform(Image.open(file))
                labels[f"m{modality_idx}"] = torch.tensor(
                    [int(file.split(".")[-2])], dtype=torch.float
                )
        else:
            for modality_idx in range(self.num_modalities):
                images[f"m{modality_idx}"] = self.transform(
                    self.images[index, modality_idx]
                )
                labels[f"m{modality_idx}"] = torch.tensor(
                    [self.labels[index, modality_idx]]
                )

        return (
            images,
            labels,
        )  # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files

    def get_modalities(self) -> List[Modality]:
        return self._modalities

    def get_attributes(self) -> List[List[str]]:
        return self._attributes

    def get_attributes_sizes(self) -> List[List[int]]:
        return self._attributes_sizes

    def get_modalities_subset_shared_attributes_indices(
        self, subset: List[int]
    ) -> List[List[int]]:
        return [len(subset) * [0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-modalities", type=int, default=5)
    parser.add_argument("--savepath-train", type=str, required=True)
    parser.add_argument("--savepath-test", type=str, required=True)
    parser.add_argument("--backgroundimagepath", type=str, required=True)
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # create dataset
    MMNISTDataset._create_mmnist_dataset(
        args.savepath_train, args.backgroundimagepath, args.num_modalities, train=True
    )
    MMNISTDataset._create_mmnist_dataset(
        args.savepath_test, args.backgroundimagepath, args.num_modalities, train=False
    )
    print("Done.")
