import argparse
import random
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import transforms

from data.datasets import MultimodalDataset
from data.datasets.MMNISTDataset import MMNISTDataset
from data.modalities.modality import Modality
from experiments.experiment import Experiment
from experiments.mmnist.networks.ConvNetworkImageClassifierCMNIST import (
    ImageClassifier as CMNISTImageClassifier,
)
from experiments.mmnist.networks.ConvNetworksImageCMNIST import EncoderImg, DecoderImg
from models import MultimodalVAE, Encoder, Decoder


class MMNISTExperiment(Experiment):
    def __init__(self, device: torch.device = None, args: Optional[List[str]] = None):
        super(MMNISTExperiment, self).__init__(
            plot_image_size=torch.Size((3, 28, 28)), device=device, args=args
        )

        self._class_dim = self.flags.class_dim
        self._style_dim = self.flags.style_dim
        self._force_partition_latent_space_mode = (
            self.flags.force_partition_latent_space_mode
        )
        self._factorized_representation = self.flags.factorized_representation
        self._pretrained_classifier_paths = self.flags.pretrained_classifier_paths

        self._initial_learning_rate = self.flags.initial_learning_rate
        self._beta_1 = self.flags.beta_1
        self._beta_2 = self.flags.beta_2

        self._beta_style = self.flags.beta_style

    def init_datasets(self) -> Tuple[MultimodalDataset, MultimodalDataset]:
        transform = transforms.Compose([transforms.ToTensor()])
        if self.flags.dataset_archive is not None:
            train = MMNISTDataset(
                dataset_archive=self.flags.dataset_archive,
                num_modalities=self.flags.num_mods,
                transform=transform,
            )
            test = MMNISTDataset(
                dataset_archive=self.flags.dataset_archive,
                load_test=True,
                num_modalities=self.flags.num_mods,
                transform=transform,
            )
        else:
            train = MMNISTDataset(
                self.flags.unimodal_datapaths_train, transform=transform
            )
            test = MMNISTDataset(
                self.flags.unimodal_datapaths_test, transform=transform
            )
        image, label = train[0]
        return train, test

    def init_model(
        self,
        flags: argparse.Namespace,
        modalities: List[Modality],
        subsets: Dict[str, List[Modality]],
    ) -> MultimodalVAE:
        num_modalities = len(modalities)
        encoders: List[Encoder] = list()
        decoders: List[Decoder] = list()
        for _ in range(num_modalities):
            encoders.append(
                EncoderImg(
                    class_dim=self._class_dim,
                    style_dim=self._style_dim,
                    num_modalities=num_modalities,
                    force_partition_latent_space_mode=self._force_partition_latent_space_mode,
                    factorized_representation=self._factorized_representation,
                )
            )
            decoders.append(
                DecoderImg(
                    class_dim=self._class_dim,
                    style_dim=self._style_dim,
                    factorized_representation=self._factorized_representation,
                )
            )

        model = MultimodalVAE(
            flags=self.flags,
            encoders=encoders,
            decoders=decoders,
            modalities=modalities,
            subsets=subsets,
        )
        return model

    def init_classifiers(self, device: torch.device) -> Dict[str, torch.nn.Module]:
        classifiers: Dict[str, torch.nn.Module] = dict()
        for modality, pretrained_classifier_path in zip(
            self.modalities, self._pretrained_classifier_paths
        ):
            classifier = CMNISTImageClassifier()
            classifier.load_state_dict(
                torch.load(pretrained_classifier_path, map_location=device)
            )
            classifier = classifier.to(device)
            classifiers[modality.name] = classifier

        for m, classifier in classifiers.items():
            if classifier is None:
                raise ValueError("Classifier is 'None' for modality %s" % str(m))
        return classifiers

    def init_optimizer(self, mm_vae: MultimodalVAE) -> torch.optim.Optimizer:
        # optimizer definition
        optimizer = optim.Adam(
            mm_vae.parameters(),
            lr=self._initial_learning_rate,
            betas=(self._beta_1, self._beta_2),
        )
        return optimizer

    def init_rec_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        rec_weights = dict()
        for modality in modalities:
            numel_mod = modality.data_size.numel()
            rec_weights[modality.name] = 1.0
        return rec_weights

    def init_style_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        weights = {modality.name: self._beta_style for modality in modalities}
        return weights

    def get_transform_mmnist(self):
        # transform_mnist = transforms.Compose([transforms.ToTensor(),
        #                                       transforms.ToPILImage(),
        #                                       transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
        #                                       transforms.ToTensor()])
        transform_mnist = transforms.Compose([transforms.ToTensor()])
        return transform_mnist

    def init_test_samples(
        self,
        num_images=10,
        dataset_test: MultimodalDataset = None,
        device: torch.device = None,
    ) -> List[Dict[str, torch.Tensor]]:
        n_test = len(dataset_test)
        samples = []
        for i in range(num_images):
            while True:
                ix = random.randint(0, n_test - 1)
                sample, target = dataset_test[ix]
                if target["m0"][0] == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key].to(self.flags.device)
                    samples.append(sample)
                    break
        return samples

    def eval_metric(self, ground_truth, prediction):
        return accuracy_score(ground_truth.ravel(), prediction.ravel())

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def eval_label(self, predictions, labels, index):
        return self.eval_metric(labels, predictions)

    def init_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--dataset", type=str, default="MMNIST", help="name of the dataset"
        )
        parser.add_argument(
            "--style_dim", type=int, default=0, help="style dimensionality"
        )  # TODO: use modality-specific style dimensions?
        parser.add_argument(
            "--num_classes",
            type=int,
            default=10,
            help="number of classes on which the data set trained",
        )
        parser.add_argument(
            "--len_sequence", type=int, default=8, help="length of sequence"
        )
        parser.add_argument(
            "--img_size_m1", type=int, default=28, help="img dimension (width/height)"
        )
        parser.add_argument(
            "--num_channels_m1",
            type=int,
            default=1,
            help="number of channels in images",
        )
        parser.add_argument(
            "--img_size_m2", type=int, default=32, help="img dimension (width/height)"
        )
        parser.add_argument(
            "--num_channels_m2",
            type=int,
            default=3,
            help="number of channels in images",
        )
        parser.add_argument(
            "--dim",
            type=int,
            default=64,
            help="number of classes on which the data set trained",
        )
        parser.add_argument(
            "--data_multiplications",
            type=int,
            default=1,
            help="number of pairs per sample",
        )
        parser.add_argument(
            "--num_hidden_layers",
            type=int,
            default=1,
            help="number of channels in images",
        )
        parser.add_argument(
            "--likelihood", type=str, default="laplace", help="output distribution"
        )

        # data
        parser.add_argument(
            "--unimodal-datapaths-train",
            nargs="+",
            type=str,
            help="directories where training data is stored",
        )
        parser.add_argument(
            "--unimodal-datapaths-test",
            nargs="+",
            type=str,
            help="directories where test data is stored",
        )
        parser.add_argument(
            "--dataset-archive", type=str, default=None, help="Path to dataset archive"
        )
        parser.add_argument(
            "--num-mods",
            type=int,
            default=None,
            help="Number of modalities. Only required when loading from archive.",
        )
        parser.add_argument(
            "--pretrained-classifier-paths",
            nargs="+",
            type=str,
            help="paths to pretrained classifiers",
        )

        # multimodal
        parser.add_argument(
            "--subsampled_reconstruction",
            default=True,
            help="subsample reconstruction path",
        )

        # Expert temperatures
        for idx in range(5):
            parser.add_argument(
                f"--expert_temperature_m{idx}",
                type=float,
                default=None,
                help=f"Temperature for expert m{idx}. Overrides global temperature.",
            )

        # weighting of loss terms
        parser.add_argument(
            "--div_weight",
            type=float,
            default=None,
            help="default weight divergence per modality, if None use 1/(num_mods+1).",
        )
        parser.add_argument(
            "--div_weight_uniform_content",
            type=float,
            default=None,
            help="default weight divergence term prior, if None use (1/num_mods+1)",
        )

        # annealing
        parser.add_argument(
            "--kl_annealing",
            type=int,
            default=0,
            help="number of kl annealing steps; 0 if no annealing should be done",
        )

    def post_process_flags(self, flags: argparse.Namespace) -> argparse.Namespace:
        if flags.dataset_archive is None:
            assert len(flags.unimodal_datapaths_train) == len(
                flags.unimodal_datapaths_test
            )
            flags.num_mods = len(
                flags.unimodal_datapaths_train
            )  # set number of modalities dynamically

            assert self.flags.num_mods is not None

        if flags.div_weight_uniform_content is None:
            flags.div_weight_uniform_content = 1 / (flags.num_mods + 1)
        flags.alpha_modalities = [flags.div_weight_uniform_content]
        if flags.div_weight is None:
            flags.div_weight = 1 / (flags.num_mods + 1)
        flags.alpha_modalities.extend([flags.div_weight for _ in range(flags.num_mods)])

        expert_temperature = None
        if flags.expert_temperature is not None:
            expert_temperature = torch.tensor(
                [flags.expert_temperature], dtype=torch.float, device=flags.device
            )
        flags.expert_temperature = dict()
        for mod in range(5):
            expert_temp_key = f"expert_temperature_m{mod}"
            if (
                hasattr(flags, expert_temp_key)
                and flags.__dict__[expert_temp_key] is not None
            ):
                flags.expert_temperature[f"m{mod}"] = torch.tensor(
                    [flags.__dict__[expert_temp_key]],
                    dtype=torch.float,
                    device=flags.device,
                )
            elif expert_temperature is not None:
                flags.expert_temperature[f"m{mod}"] = expert_temperature

        return flags
