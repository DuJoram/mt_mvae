import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from data.datasets import MultimodalDataset, MdSpritesDataset
from data.modalities import Modality
from experiments.experiment import Experiment
from experiments.mdsprites.networks import DSpritesImageClassifier
from models import MultimodalVAE, Encoder, Decoder
from .networks.mdsprites_dense_image import (
    MdSpritesDenseImageEncoder,
    MdSpritesDenseImageDecoder,
)


class MdSpritesExperiment(Experiment):
    def __init__(self, device: torch.device = None, args: Optional[List[str]] = None):
        super(MdSpritesExperiment, self).__init__(
            plot_image_size=torch.Size((1, 64, 64)), device=device, args=args
        )

        self._dsprites_archive = self.flags.dsprites_archive
        self._mdsprites_archive = self.flags.multimodal_dsprites_archive
        self._class_dim = self.flags.class_dim
        self._style_dim = self.flags.style_dim
        self._force_partition_latent_space_mode = (
            self.flags.force_partition_latent_space_mode
        )
        self._pretrained_classifier_path = self.flags.pretrained_classifier
        self._factorized_representation = self.flags.factorized_representation
        self._num_modalities = self.flags.num_mods
        self._shared_attributes = self.flags.shared_attributes
        self._likelihood = self.flags.likelihood

        self._encoder_num_hidden_layers = self.flags.encoder_num_hidden_layers
        self._decoder_num_hidden_layers = self.flags.decoder_num_hidden_layers

        self._initial_learning_rate = self.flags.initial_learning_rate
        self._beta1 = self.flags.beta_1
        self._beta2 = self.flags.beta_2

        self._beta_style = self.flags.beta_style

        self._num_training_samples = 100_000
        self._num_test_samples = 10_000

    def init_datasets(self) -> Tuple[MultimodalDataset, MultimodalDataset]:
        train = MdSpritesDataset(
            dsprites_archive_path=self._dsprites_archive,
            num_modalities=self._num_modalities,
            shared_attributes=self._shared_attributes,
            decoder_distribution=self._likelihood,
            num_training_samples=self._num_training_samples,
            num_test_samples=self._num_test_samples,
            filter_symmetries=True,
            train=True,
        )
        test = MdSpritesDataset(
            dsprites_archive_path=self._dsprites_archive,
            num_modalities=self._num_modalities,
            shared_attributes=self._shared_attributes,
            decoder_distribution=self._likelihood,
            num_training_samples=self._num_training_samples,
            num_test_samples=self._num_test_samples,
            filter_symmetries=False,
            train=False,
        )
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

        predict_variance = "bernoulli" not in flags.likelihood
        predict_logits = "logits" in flags.likelihood

        for _ in range(num_modalities):
            encoders.append(
                MdSpritesDenseImageEncoder(
                    class_dim=self._class_dim,
                    style_dim=self._style_dim,
                    input_channels=1,
                    num_hidden_layers=self._encoder_num_hidden_layers,
                    num_modalities=num_modalities,
                    force_partition_latent_space_mode=self._force_partition_latent_space_mode,
                    factorized_representation=self._factorized_representation,
                )
            )
            decoders.append(
                MdSpritesDenseImageDecoder(
                    class_dim=self._class_dim,
                    style_dim=self._style_dim,
                    num_hidden_layers=self._decoder_num_hidden_layers,
                    output_channels=1,
                    factorized_representation=self._factorized_representation,
                    predict_variance=predict_variance,
                    predict_logits=predict_logits,
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
        # Since all modalities are the same, we only need one classifier.
        classifiers: Dict[str, torch.nn.Module] = dict()
        classifier = DSpritesImageClassifier(in_channels=1)
        classifier.load_state_dict(
            torch.load(self._pretrained_classifier_path, map_location=device)
        )
        classifier = classifier.to(device)

        for modality in self.modalities:
            classifiers[modality.name] = classifier

        return classifiers

    def init_optimizer(self, mm_vae: MultimodalVAE) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            mm_vae.parameters(),
            lr=self._initial_learning_rate,
            betas=(self._beta1, self._beta2)
        )
        return optimizer

    def init_rec_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        rec_weights = dict()
        for modality in modalities:
            rec_weights[modality.name] = 1.0

        return rec_weights

    def init_style_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        weights = {modality.name: self._beta_style for modality in modalities}
        return weights

    def init_test_samples(
        self,
        num_images: int = 10,
        dataset_test: MultimodalDataset = None,
        device: torch.device = None,
    ) -> List[Dict[str, torch.Tensor]]:
        samples = []

        indices = np.random.choice(
            np.arange(0, len(dataset_test)), num_images, replace=False
        )

        for index in indices:
            sample, target = self.dataset_test[index]
            for modality_name, sample_modality in sample.items():
                sample[modality_name] = sample_modality.to(device)

            samples.append(sample)

        return samples

    def eval_metric(self, ground_truth, prediction):
        return accuracy_score(ground_truth.ravel(), prediction.ravel())

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def eval_label(self, values, labels, index=None):
        return self.eval_metric(values[:, index], labels[:, index])

    def init_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--dataset", type=str, default="MdSprites", help="Name of the dataset"
        )
        parser.add_argument(
            "--dsprites-archive",
            type=str,
            default="resources/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
            help="Original dSprites archive (.npz)",
        )
        parser.add_argument(
            "--multimodal-dsprites-archive",
            type=str,
            default="resources/data/mdsprites_m4_col_sc_posx.npz",
            help="Multimodal dSprites dataset archive",
        )
        parser.add_argument(
            "--pretrained-classifier",
            type=str,
            default="resources/trained_classifiers/trained_classifiers_dsprites/dsprites_classifier",
            help="Pretrained dSprites multi-attribute classifier",
        )
        parser.add_argument(
            "--style_dim", type=int, default=0, help="style dimensionality"
        )

        parser.add_argument(
            "--likelihood", type=str, default="bernoulli", help="Output distribution"
        )

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

        parser.add_argument(
            "--num_mods", type=int, default=4, help="Number of modalities"
        )
        parser.add_argument(
            "--shared-attributes",
            type=str,
            nargs="*",
            default=None,
            choices=["shape", "scale", "orientation", "posX", "posY"],
            help="List of names of shared attributes",
        )

        parser.add_argument(
            "--encoder_num_hidden_layers",
            type=int,
            default=2,
            help="Number of hidden layers in the encoder network",
        )

        parser.add_argument(
            "--decoder_num_hidden_layers",
            type=int,
            default=2,
            help="Number of hidden layers in the decoder network",
        )

        parser.add_argument(
            "--learning_rate_decay",
            type=float,
            default=5e-3,
            help="Learning rate decay for AdaGrad"
        )

        parser.add_argument(
            "--no_filter_symmetries",
            dest="filter_symmetries",
            default=True,
            action="store_false",
            help="Filter rotational symmetries from square and ellipse shapes"
        )

    def post_process_flags(self, flags: argparse.Namespace) -> argparse.Namespace:
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

        for modality_idx in range(flags.num_mods):
            expert_temperature_key = f"expert_temperature_m{modality_idx}"
            if (
                hasattr(flags, expert_temperature_key)
                and flags.__dict__[expert_temperature_key] is not None
            ):
                flags.expert_temperature[f"m{modality_idx}"] = torch.tensor(
                    [flags.__dict__[expert_temperature_key]],
                    dtype=torch.float,
                    device=flags.device,
                )
            elif expert_temperature is not None:
                flags.expert_temperature[f"m{modality_idx}"] = expert_temperature

        return flags
