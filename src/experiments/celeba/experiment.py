import argparse
import os 
import random
from typing import Tuple, List, Dict, Optional

import numpy as np

import PIL.Image as Image
from PIL import ImageFont 
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import average_precision_score

from data.datasets import MultimodalDataset
from data.modalities import Modality
from data.modalities.CelebaImage import CelebaImage
from data.modalities.CelebaText import CelebaText
from data.datasets.CelebADataset import CelebaDataset
from experiments.celeba.networks.VAEbimodalCelebA import VAEbimodalCelebA
from experiments.celeba.networks.ConvNetworkImageClassifierCelebA import ImageClassifier
from experiments.celeba.networks.ConvNetworkTextClassifierCelebA import TextClassifier

from experiments.celeba.networks.ConvNetworksImageCelebA import EncoderImg, DecoderImg
from experiments.celeba.networks.ConvNetworksTextCelebA import EncoderText, DecoderText

from experiments.experiment import Experiment
from models import Encoder, Decoder, MultimodalVAE


class CelebaExperiment(Experiment):
    def __init__(self, device: torch.device = None, args: Optional[List[str]] = None):
        super(CelebaExperiment, self).__init__(plot_image_size=torch.Size((3, 64, 64)), device=device, args=args)

        self._dir_classifier = args.dir_classifier
        self._classifier_save_m1 = args.classifier_save_m1
        self._classifier_save_m2 = args.classifier_save_m2

        self._image_size = args.img_size
        self._crop_size_image = args.crop_size_img

        self._beta_1 = args.beta_1
        self._beta_2 = args.beta_2
        self._initial_learning_rate = args.initial_learning_rate

    def post_process_flags(self, flags: argparse.Namespace) -> argparse.Namespace:
        flags.alpha_modalities = [flags.div_weight_uniform_content, flags.div_weight_m1_content, flags.div_weight_m2_content]
        return flags

    def init_datasets(self) -> Tuple[MultimodalDataset, MultimodalDataset]:
        transform = self.get_transform_celeba();
        dataset_train = CelebaDataset(self.flags, self.alphabet, partition=0, transform=transform)
        dataset_test = CelebaDataset(self.flags, self.alphabet, partition=1, transform=transform)
        return dataset_train, dataset_test

    def init_model(self, flags: argparse.Namespace, modalities: List[Modality], subsets: Dict[str, List[Modality]]) -> MultimodalVAE:
        encoders: List[Encoder] = list()
        decoders: List[Decoder] = list()

        for modality in modalities:
            if modality.name == "image":
                encoders.append(EncoderImg(self.flags))
                decoders.append(DecoderImg(self.flags))
            elif modality.name == "text":
                encoders.append(EncoderText(self.flags))
                decoders.append(DecoderText(self.flags))

        model = MultimodalVAE(self.flags, encoders, decoders, modalities, self.subsets)
        return model

    def get_transform_celeba(self):
        offset_height = (218 - self._crop_size_image) // 2
        offset_width = (178 - self._crop_size_image) // 2
        crop = lambda x: x[:, offset_height:offset_height + self._crop_size_image,
                         offset_width:offset_width + self._crop_size_image]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(crop),
                                        transforms.ToPILImage(),
                                        transforms.Resize(size=(self._image_size,
                                                                self._image_size),
                                                          interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])

        return transform;


    def init_classifiers(self, device: torch.device) -> Dict[str, torch.nn.Module]:
        model_classifier_m1 = ImageClassifier(self.flags);
        model_classifier_m1.load_state_dict(torch.load(os.path.join(self._dir_classifier, self._classifier_save_m1)))
        model_classifier_m1 = model_classifier_m1.to(device);

        model_classifier_m2 = TextClassifier(self.flags);
        model_classifier_m2.load_state_dict(torch.load(os.path.join(self._dir_classifier, self._classifier_save_m2)))
        model_classifier_m2 = model_classifier_m2.to(device);

        classifiers = {
            "image": model_classifier_m1,
            "text": model_classifier_m2
        }
        return classifiers


    def init_optimizer(self, mm_vae: MultimodalVAE) -> torch.optim.Optimizer:
        # optimizer definition
        optimizer = optim.Adam(
            mm_vae.parameters(),
            lr=self._initial_learning_rate,
            betas=(self._beta_1, self._beta_2)
        )
        return optimizer


    def init_rec_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        rec_weights = dict();
        ref_mod_d_size = None
        for modality in modalities:
            if modality.name == "image":
                ref_mod_d_size = modality.data_size.numel()

        for modality in modalities:
            numel_mod = modality.data_size.numel()
            rec_weights[modality.name] = float(ref_mod_d_size/numel_mod)
        return rec_weights;


    def init_style_weights(self, modalities: List[Modality], args: argparse.Namespace) -> Dict[str, float]:
        weights = dict();
        for modality in modalities:
            weights[modality.name] = args.beta_m1_style if modality.name == "image" else args.beta_m2_style
        return weights


    def init_test_samples(self, num_images: int = 10, dataset_test: MultimodalDataset = None, device: torch.device = None) -> List[Dict[str, torch.Tensor]]:
        dataset_size = len(dataset_test)
        samples = list()

        for idx in range(10):
            sample, target = self.dataset_test[random.randint(0, dataset_size)]

            for modality_name, sample_modality in sample.items():
                sample[modality_name] = sample_modality.to(device)
            samples.append(sample)

        return samples

    def get_prediction_from_attr(self, values):
        return values.ravel();

    def get_prediction_from_attr_random(self, values, index=None):
        return values[:,index] > 0.5;

    def eval_label(self, values, labels, index=None):
        pred = values[:,index];
        gt = labels[:,index];
        return self.eval_metric(gt, pred);

    def eval_metric(self, ground_truth, prediction):
        return average_precision_score(ground_truth.ravel(), prediction.ravel())

    def mean_eval_metric(self, values):
        return np.mean(np.array(values));

    def init_parser(self, parser: argparse.ArgumentParser):
        # DATASET NAME
        parser.add_argument('--dataset', type=str, default='CelebA', help="name of the dataset")

        # add arguments
        parser.add_argument('--style_img_dim', type=int, default=32, help="dimension of varying factor latent space")
        parser.add_argument('--style_text_dim', type=int, default=32, help="dimension of varying factor latent space")
        parser.add_argument('--len_sequence', type=int, default=256, help="length of sequence")
        parser.add_argument('--img_size', type=int, default=64, help="img dimension (width/height)")
        parser.add_argument('--image_channels', type=int, default=3, help="number of channels in images")
        parser.add_argument('--crop_size_img', type=int, default=148, help="number of channels in images")
        parser.add_argument('--dir_text', type=str, default='text', help="directory where text is stored")
        parser.add_argument('--random_text_ordering', type=bool, default=False, help="flag to indicate if attributes are shuffled randomly")
        parser.add_argument('--random_text_startindex', type=bool, default=True, help="flag to indicate if start index is random")

        parser.add_argument('--DIM_text', type=int, default=128, help="filter dimensions of residual layers")
        parser.add_argument('--DIM_img', type=int, default=128, help="filter dimensions of residual layers")
        parser.add_argument('--num_layers_text', type=int, default=7, help="number of residual layers")
        parser.add_argument('--num_layers_img', type=int, default=5, help="number of residual layers")
        parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
        parser.add_argument('--likelihood_m2', type=str, default='categorical', help="output distribution")

        # classifier
        parser.add_argument('--classifier_save_m1', type=str, default='classifier_m1', help="model save for classifier")
        parser.add_argument('--classifier_save_m2', type=str, default='classifier_m2', help="model save for classifier")

        # weighting of loss terms
        parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
        parser.add_argument('--beta_m2_style', type=float, default=2.0, help="default weight divergence term style modality 2")
        parser.add_argument('--div_weight_m1_content', type=float, default=0.35, help="default weight divergence term content modality 1")
        parser.add_argument('--div_weight_m2_content', type=float, default=0.35, help="default weight divergence term content modality 2")
        parser.add_argument('--div_weight_uniform_content', type=float, default=0.3, help="default weight divergence term prior")
