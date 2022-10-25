import argparse
import os
import random
from typing import Tuple, List, Dict, Optional

import PIL.Image as Image
import numpy as np
import torch
import torch.optim as optim
from PIL import ImageFont
from sklearn.metrics import accuracy_score
from torchvision import transforms

from data.datasets import MultimodalDataset
from data.datasets.SVHNMNISTDataset import SVHNMNIST
from data.modalities import Modality
from data.modalities.MNIST import MNIST
from data.modalities.SVHN import SVHN
from data.modalities.Text import Text
from experiments.experiment import Experiment
from experiments.mnistsvhntext.networks.ConvNetworkImageClassifierMNIST import (
    ClfImg as ClfImgMNIST,
)
from experiments.mnistsvhntext.networks.ConvNetworkImageClassifierSVHN import ClfImgSVHN
from experiments.mnistsvhntext.networks.ConvNetworkTextClassifier import (
    ClfText as ClfText,
)
from experiments.mnistsvhntext.networks.ConvNetworksImageMNIST import (
    EncoderImg,
    DecoderImg,
)
from experiments.mnistsvhntext.networks.ConvNetworksImageSVHN import (
    EncoderSVHN,
    DecoderSVHN,
)
from experiments.mnistsvhntext.networks.ConvNetworksTextMNIST import (
    EncoderText,
    DecoderText,
)
from experiments.mnistsvhntext.networks.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST


# from utils.Experiment import Experiment
from models import MultimodalVAE, Encoder, Decoder


class MNISTSVHNText(Experiment):
    def __init__(self, device: torch.device = None, args: Optional[List[str]] = None):
        super(MNISTSVHNText, self).__init__(plot_image_size = torch.Size((3, 28, 28)), device=device, args=args)

    def init_datasets(self) -> Tuple[MultimodalDataset, MultimodalDataset]:
        transform_mnist = self.get_transform_mnist()
        transform_svhn = self.get_transform_svhn()
        transforms = [transform_mnist, transform_svhn]
        train = SVHNMNIST(self.flags, self.alphabet, font=self.font, train=True, transform=transforms)
        test = SVHNMNIST(self.flags, self.alphabet, font=self.font, train=False, transform=transforms)
        return train, test

    def init_model(self, flags: argparse.Namespace, modalities: List[Modality], subsets: Dict[str, List[Modality]]) -> MultimodalVAE:
        encoders: List[Encoder] = list()
        decoders: List[Decoder] = list()

        for modality in modalities:
            if modality.name == "mnist":
                encoders.append(EncoderImg(
                    num_hidden_layers=flags.num_hidden_layers,
                    class_dim=flags.class_dim,
                    style_mnist_dim=flags.style_mnist_dim,
                    num_mods=2,
                    force_partition_latent_space_mode=flags.force_partition_latent_space_mode,
                    factorized_representation=flags.factorized_representation
                ))
                decoders.append(DecoderImg(
                    class_dim=flags.class_dim,
                    style_mnist_dim=flags.style_mnist_dim,
                    num_hidden_layers=flags.num_hidden_layers,
                    factorized_representation=flags.factorized_representation
                ))
            elif modality.name == "svhn":
                encoders.append(EncoderSVHN(flags))
                decoders.append(DecoderSVHN(flags))
            elif modality.name == "text":
                encoders.append(EncoderText(flags))
                decoders.append(DecoderText(flags))

        model = MultimodalVAE(
            flags=flags,
            encoders=encoders,
            decoders=decoders,
            modalities=modalities,
            subsets=subsets
        )
        return model


    def get_transform_mnist(self):
        transform_mnist = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        return transform_mnist

    def get_transform_svhn(self):
        transform_svhn = transforms.Compose([transforms.ToTensor()])
        return transform_svhn


    def init_classifiers(self, device) -> Dict[str, torch.nn.Module]:
        model_classifier_m1 = None
        model_classifier_m2 = None
        model_classifier_m3 = None
        if self.flags.use_classifier:
            model_classifier_m1 = ClfImgMNIST()
            model_classifier_m1.load_state_dict(
                torch.load(
                    os.path.join(
                        self.flags.dir_classifier, self.flags.classifier_save_m1
                    )
                )
            )
            model_classifier_m1 = model_classifier_m1.to(device)

            model_classifier_m2 = ClfImgSVHN()
            model_classifier_m2.load_state_dict(
                torch.load(
                    os.path.join(
                        self.flags.dir_classifier, self.flags.classifier_save_m2
                    )
                )
            )
            model_classifier_m2 = model_classifier_m2.to(device)

            model_classifier_m3 = ClfText(self.flags)
            model_classifier_m3.load_state_dict(
                torch.load(
                    os.path.join(
                        self.flags.dir_classifier, self.flags.classifier_save_m3
                    )
                )
            )
            model_classifier_m3 = model_classifier_m3.to(device)

        classifiers = {
            "mnist": model_classifier_m1,
            "svhn": model_classifier_m2,
            "text": model_classifier_m3,
        }
        return classifiers

    def init_optimizer(self, mm_vae: MultimodalVAE):
        # optimizer definition
        optimizer = optim.Adam(
            list(mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2),
        )
        return optimizer

    def init_rec_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        rec_weights = dict()
        ref_mod_d_size = 0
        for modality in modalities:
            if modality.name == "svhn":
                ref_mod_d_size = modality.data_size.numel()

        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        return rec_weights

    def init_style_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        weights = dict()
        weights["mnist"] = self.flags.beta_m1_style
        weights["svhn"] = self.flags.beta_m2_style
        weights["text"] = self.flags.beta_m3_style
        return weights

    def init_test_samples(self, num_images=10, dataset_test: MultimodalDataset = None, device: torch.device = None) -> List[Dict[str, torch.Tensor]]:
        n_test = len(dataset_test)
        samples = []
        for i in range(num_images):
            while True:
                sample, target = self.dataset_test[random.randint(0, n_test)]
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key].to(device)
                    samples.append(sample)
                    break
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def get_prediction_from_attr(self, attr, index=None):
        return np.argmax(attr.cpu().data.numpy(), axis=1).astype(int)

    def eval_metric(self, ground_truth, prediction):
        return accuracy_score(ground_truth.ravel(), prediction.ravel())

    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)

    def init_parser(self, parser: argparse.ArgumentParser):
        # DATASET NAME
        parser.add_argument(
            "--dataset", type=str, default="SVHN_MNIST_text", help="name of the dataset"
        )

        # DATA DEPENDENT
        # to be set by experiments themselves
        parser.add_argument(
            "--style_m1_dim",
            type=int,
            default=0,
            help="dimension of varying factor latent space",
        )
        parser.add_argument(
            "--style_m2_dim",
            type=int,
            default=0,
            help="dimension of varying factor latent space",
        )
        parser.add_argument(
            "--style_m3_dim",
            type=int,
            default=0,
            help="dimension of varying factor latent space",
        )
        parser.add_argument(
            "--len_sequence", type=int, default=8, help="length of sequence"
        )
        parser.add_argument(
            "--num_classes",
            type=int,
            default=10,
            help="number of classes on which the data set trained",
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
            default=20,
            help="number of pairs per sample",
        )
        parser.add_argument(
            "--num_hidden_layers",
            type=int,
            default=1,
            help="number of channels in images",
        )
        parser.add_argument(
            "--likelihood_m1", type=str, default="laplace", help="output distribution"
        )
        parser.add_argument(
            "--likelihood_m2", type=str, default="laplace", help="output distribution"
        )
        parser.add_argument(
            "--likelihood_m3",
            type=str,
            default="categorical",
            help="output distribution",
        )

        # SAVE and LOAD
        # to bet set by experiments themselves
        parser.add_argument(
            "--encoder_save_m1",
            type=str,
            default="encoderM1",
            help="model save for encoder",
        )
        parser.add_argument(
            "--encoder_save_m2",
            type=str,
            default="encoderM2",
            help="model save for encoder",
        )
        parser.add_argument(
            "--encoder_save_m3",
            type=str,
            default="encoderM3",
            help="model save for decoder",
        )
        parser.add_argument(
            "--decoder_save_m1",
            type=str,
            default="decoderM1",
            help="model save for decoder",
        )
        parser.add_argument(
            "--decoder_save_m2",
            type=str,
            default="decoderM2",
            help="model save for decoder",
        )
        parser.add_argument(
            "--decoder_save_m3",
            type=str,
            default="decoderM3",
            help="model save for decoder",
        )
        parser.add_argument(
            "--classifier_save_m1",
            type=str,
            default="classifier_m1",
            help="model save for classifier",
        )
        parser.add_argument(
            "--classifier_save_m2",
            type=str,
            default="classifier_m2",
            help="model save for classifier",
        )
        parser.add_argument(
            "--classifier_save_m3",
            type=str,
            default="classifier_m3",
            help="model save for classifier",
        )

        # LOSS TERM WEIGHTS
        parser.add_argument(
            "--beta_m1_style",
            type=float,
            default=1.0,
            help="default weight divergence term style modality 1",
        )
        parser.add_argument(
            "--beta_m2_style",
            type=float,
            default=1.0,
            help="default weight divergence term style modality 2",
        )
        parser.add_argument(
            "--beta_m3_style",
            type=float,
            default=1.0,
            help="default weight divergence term style modality 2",
        )
        parser.add_argument(
            "--div_weight_m1_content",
            type=float,
            default=0.25,
            help="default weight divergence term content modality 1",
        )
        parser.add_argument(
            "--div_weight_m2_content",
            type=float,
            default=0.25,
            help="default weight divergence term content modality 2",
        )
        parser.add_argument(
            "--div_weight_m3_content",
            type=float,
            default=0.25,
            help="default weight divergence term content modality 2",
        )
        parser.add_argument(
            "--div_weight_uniform_content",
            type=float,
            default=0.25,
            help="default weight divergence term prior",
        )

    def post_process_flags(self, flags: argparse.Namespace) -> argparse.Namespace:
        flags.alpha_modalities = [
            flags.div_weight_uniform_content,
            flags.div_weight_m1_content,
            flags.div_weight_m2_content,
            flags.div_weight_m3_content,
        ]
        return flags
