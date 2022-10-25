import argparse
import json
import os
from abc import ABC, abstractmethod, abstractclassmethod
from itertools import chain, combinations
from typing import List, Dict, Optional, Tuple, Callable, NoReturn

import torch
import torch.cuda
from PIL import ImageFont

from data.datasets import MultimodalDataset
from data.modalities import Modality
from models import MultimodalVAE
from utils.filehandling import create_dir_structure, create_dir_structure_testing


class Experiment(ABC):
    def __init__(
        self,
        plot_image_size=None,
        device: torch.device = None,
        args: Optional[List[str]] = None,
    ):

        if plot_image_size is None:
            self.plot_image_size = torch.Size((3, 64, 64))
        else:
            self.plot_image_size = plot_image_size

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        alphabet_path = os.path.join(os.getcwd(), "resources/alphabet.json")
        with open(alphabet_path) as alphabet_file:
            alphabet = str("".join(json.load(alphabet_file)))
        self.alphabet: str = alphabet

        self.font = ImageFont.truetype("resources/FreeSerif.ttf", 38)

        self._arg_parser = argparse.ArgumentParser()
        self._init_argparse()
        self.flags: argparse.Namespace = self._parse_args(args)

        self._load_saved: bool
        self._trained_model_path: str

        self.dataset: str

        self.dataset_train: MultimodalDataset
        self.dataset_test: MultimodalDataset

        # FIXME: `labels` is not the right term for this.
        self.labels: List[str]
        self.attributes: List[List[str]]

        self.modalities: List[Modality]
        self.num_modalities: int
        self.modalities_subsets: Dict[str, List[Modality]]
        self.modalities_subsets_indices: List[List[int]]
        self.modalities_subsets_names: List[str]
        self.subsets_shared_attributes_modalities_indices: List[List[List[int]]]

        # FIXME: Replace with `self.modalities_subsets`
        self.subsets: Dict[str, List[Modality]]

        self.mm_vae: MultimodalVAE

        self.classifiers: Optional[Dict[str, torch.nn.Module]]
        self.optimizer: torch.optim.Optimizer

        self.rec_weights: Dict[str, float]
        self.style_weights: Dict[str, float]

        self.test_samples: List[Dict[str, torch.Tensor]]
        self.paths_fid: Dict[str, str]

        self._initialized: bool = False

    def init(self):
        if self._initialized:
            return

        self._load_saved = self.flags.load_saved
        self._trained_model_path = self.flags.trained_model_path

        self.dataset: str = self.flags.dataset

        dataset_train, dataset_test = self.init_datasets()
        self.dataset_train: MultimodalDataset = dataset_train
        self.dataset_test: MultimodalDataset = dataset_test

        # FIXME: `labels` is not the right term for this.
        self.labels = self.dataset_train.get_attributes()[0]

        self.modalities: List[Modality] = self.dataset_train.get_modalities()
        self.num_modalities: int = len(self.modalities)
        (
            modalities_subsets,
            modalities_subsets_indices,
            modalities_subsets_names,
        ) = self._init_subsets()
        self.modalities_subsets: Dict[str, List[Modality]] = modalities_subsets
        self.modalities_subsets_indices: List[List[int]] = modalities_subsets_indices
        self.modalities_subsets_names: List[str] = modalities_subsets_names
        self.modalities_subsets_name_to_index: Dict[str, int] = {
            subset_name: subset_idx
            for subset_idx, subset_name in enumerate(self.modalities_subsets_names)
        }

        self.attributes: List[List[str]] = self.dataset_train.get_attributes()
        self.attributes_sizes: List[
            List[int]
        ] = self.dataset_train.get_attributes_sizes()

        # For subset of modalites:
        #   For each shared attribute:
        #       For each modality: The index of the attribute for this modality
        self.subsets_shared_attributes_modalities_indices: List[List[List[int]]] = [
            self.dataset_train.get_modalities_subset_shared_attributes_indices(subset)
            for subset in self.modalities_subsets_indices
        ]

        # FIXME: Replace with `self.modalities_subsets`
        self.subsets: Dict[str, List[Modality]] = self.modalities_subsets

        self.mm_vae: MultimodalVAE = self.init_model(
            flags=self.flags, modalities=self.modalities, subsets=self.subsets
        )

        if self._load_saved:
            self.load_checkpoint(self._trained_model_path)

        self.mm_vae = self.mm_vae.to(self._device)

        self.classifiers: Optional[Dict[str, torch.nn.Module]] = None
        if self.flags.use_classifier:
            self.classifiers = self.init_classifiers(self._device)

        self.optimizer: torch.optim.Optimizer = self.init_optimizer(self.mm_vae)

        self.rec_weights: Dict[str, float] = self.init_rec_weights(self.modalities)
        self.style_weights: Dict[str, float] = self.init_style_weights(self.modalities)

        self.test_samples: List[Dict[str, torch.Tensor]] = self.init_test_samples(
            num_images=10, dataset_test=self.dataset_test, device=self._device
        )
        self.paths_fid: Dict[str, str] = self.init_paths_fid()

        create_dir_structure_testing(self.flags, self.labels)
        self._initialized = True

    @abstractmethod
    def init_datasets(self) -> Tuple[MultimodalDataset, MultimodalDataset]:
        pass

    @abstractmethod
    def init_model(
        self,
        flags: argparse.Namespace,
        modalities: List[Modality],
        subsets: Dict[str, List[Modality]],
    ) -> MultimodalVAE:
        pass

    @abstractmethod
    def init_classifiers(self, device: torch.device) -> Dict[str, torch.nn.Module]:
        pass

    @abstractmethod
    def init_optimizer(self, mm_vae: MultimodalVAE) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def init_rec_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        pass

    @abstractmethod
    def init_style_weights(self, modalities: List[Modality]) -> Dict[str, float]:
        pass

    @abstractmethod
    def init_test_samples(
        self, num_images: int, dataset_test: MultimodalDataset, device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def mean_eval_metric(self, values):
        pass

    @abstractmethod
    def eval_label(self, values, labels, index=None):
        pass

    def load_checkpoint(self, checkpoint_file: str):
        self.mm_vae.load_state_dict(
            torch.load(checkpoint_file, map_location=self._device)
        )

    def _init_subsets(
        self,
    ) -> Tuple[Dict[str, List[Modality]], List[List[int]], List[str]]:
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """

        modality_names = [modality.name for modality in self.modalities]

        modality_indices = list(range(self.num_modalities))

        modalities_subsets_indices = list(
            map(
                list,
                chain.from_iterable(
                    combinations(modality_indices, n)
                    for n in range(self.num_modalities + 1)
                ),
            )
        )

        modalities_subsets_names: List[str] = list()
        modalities_subsets: Dict[str, List[Modality]] = dict()

        for subset_indices in modalities_subsets_indices:
            subset_names: List[str] = list()
            subset_modalities: List[Modality] = list()
            for modality_idx in subset_indices:
                subset_names.append(self.modalities[modality_idx].name)
                subset_modalities.append(self.modalities[modality_idx])
            subset_name = "_".join(subset_names)
            modalities_subsets_names.append(subset_name)
            modalities_subsets[subset_name] = subset_modalities

        return modalities_subsets, modalities_subsets_indices, modalities_subsets_names

    def init_paths_fid(self) -> Dict[str, str]:
        dir_real = os.path.join(self.flags.dir_gen_eval_fid, "real")
        dir_random = os.path.join(self.flags.dir_gen_eval_fid, "random")
        paths = {"real": dir_real, "random": dir_random}
        dir_cond = self.flags.dir_gen_eval_fid
        for k, name in enumerate(self.subsets):
            paths[name] = os.path.join(dir_cond, name)
        print(paths.keys())
        return paths

    @property
    def arg_parser(self) -> argparse.ArgumentParser:
        return self._arg_parser

    @abstractmethod
    def init_parser(self, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def eval_metric(self, ground_truth, prediction):
        pass

    def post_process_flags(self, flags: argparse.Namespace) -> argparse.Namespace:
        return flags

    def _init_argparse(self):

        # General
        self._arg_parser.add_argument("--run_name", type=str, help="Name of the run")
        self._arg_parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of parallel workers for data loading",
        )

        # TRAINING
        self._arg_parser.add_argument(
            "--batch_size", type=int, default=256, help="batch size for training"
        )
        self._arg_parser.add_argument(
            "--initial_learning_rate",
            type=float,
            default=0.001,
            help="starting learning rate",
        )
        self._arg_parser.add_argument(
            "--beta_1", type=float, default=0.9, help="default beta_1 val for adam"
        )
        self._arg_parser.add_argument(
            "--beta_2", type=float, default=0.999, help="default beta_2 val for adam"
        )
        self._arg_parser.add_argument(
            "--start_epoch",
            type=int,
            default=0,
            help="flag to set the starting epoch for training",
        )
        self._arg_parser.add_argument(
            "--end_epoch",
            type=int,
            default=100,
            help="flag to indicate the final epoch of training",
        )

        # DATA DEPENDENT
        self._arg_parser.add_argument(
            "--class_dim",
            type=int,
            default=20,
            help="dimension of common factor latent space",
        )

        # SAVE and LOAD
        self._arg_parser.add_argument(
            "--checkpoint_model_name",
            type=str,
            default="mm_vae",
            help="model save for vae_bimodal",
        )
        self._arg_parser.add_argument(
            "--load_saved",
            action="store_true",
            default=False,
            help="flag to indicate if a saved model will be loaded",
        )
        self._arg_parser.add_argument(
            "--trained_model_path",
            type=str,
            default="",
            help="Path to model save to be loaded",
        )

        self._arg_parser.add_argument(
            "--checkpoint_frequency",
            type=int,
            default=5,
            help="Save model checkpoint every `checkpoint-frequency` epoch",
        )

        # DIRECTORIES
        # classifiers
        self._arg_parser.add_argument(
            "--dir_classifier",
            type=str,
            default="resources/classifiers/",
            help="directory where classifier is stored",
        )
        # data
        self._arg_parser.add_argument(
            "--dir_data",
            type=str,
            default="data",
            help="directory where data is stored",
        )
        # experiments
        self._arg_parser.add_argument(
            "--dir_experiment",
            type=str,
            default="runs/tmp/",
            help="directory to save generated samples in",
        )
        # fid
        self._arg_parser.add_argument(
            "--dir_fid",
            type=str,
            default=None,
            help="directory to save generated samples for fid score calculation",
        )
        # fid_score
        self._arg_parser.add_argument(
            "--inception_state_dict",
            type=str,
            default="resources/inception_state_dict.pth",
            help="path to inception v3 state dict",
        )

        # EVALUATION
        self._arg_parser.add_argument(
            "--use_classifier",
            default=False,
            action="store_true",
            help="flag to indicate if generates samples should be classified",
        )
        self._arg_parser.add_argument(
            "--calc_nll",
            default=False,
            action="store_true",
            help="flag to indicate calculation of nll",
        )
        self._arg_parser.add_argument(
            "--eval_lr",
            default=False,
            action="store_true",
            help="flag to indicate evaluation of lr",
        )
        self._arg_parser.add_argument(
            "--calc_prd",
            default=False,
            action="store_true",
            help="flag to indicate calculation of prec-rec for gen model",
        )
        self._arg_parser.add_argument(
            "--save_figure",
            default=False,
            action="store_true",
            help="flag to indicate if figures should be saved to disk (in addition to tensorboard logs)",
        )
        self._arg_parser.add_argument(
            "--plotting_freq",
            type=int,
            default=10,
            help="frequency of plots (in number of epochs)",
        )
        self._arg_parser.add_argument(
            "--eval_freq",
            type=int,
            default=10,
            help="frequency of evaluation of latent representation of generative performance (in number of epochs)",
        )
        self._arg_parser.add_argument(
            "--eval_freq_fid",
            type=int,
            default=10,
            help="frequency of evaluation of latent representation of generative performance (in number of epochs)",
        )
        self._arg_parser.add_argument(
            "--num_samples_fid",
            type=int,
            default=10,
            help="number of samples the calculation of fid is based on",
        )
        self._arg_parser.add_argument(
            "--num_training_samples_lr",
            type=int,
            default=50,
            help="number of training samples to train the latent representation classifier",
        )
        self._arg_parser.add_argument(
            "--use_expert_classifier",
            default=False,
            action="store_true",
            help="Train expert classifier on latent representations",
        )
        self._arg_parser.add_argument(
            "--expert_classifier_epochs",
            type=int,
            default=2,
            help="Number of epochs to train the expert classifier for",
        )
        self._arg_parser.add_argument(
            "--expert_classifier_learning_rate",
            type=float,
            default=5e-3,
            help="Learning rate of experts classifier",
        )
        self._arg_parser.add_argument(
            "--expert_classifier_layers",
            type=int,
            default=4,
            help="Number of layers in the experts classifier",
        )

        # multimodal
        self._arg_parser.add_argument(
            "--method",
            type=str,
            default="poe",
            help="choose method for training the model",
        )
        self._arg_parser.add_argument(
            "--modality_jsd", type=bool, default=False, help="modality_jsd"
        )
        self._arg_parser.add_argument(
            "--modality_poe", type=bool, default=False, help="modality_poe"
        )
        self._arg_parser.add_argument(
            "--modality_moe", type=bool, default=False, help="modality_moe"
        )
        self._arg_parser.add_argument("--mopoe", type=bool, default=False, help="mopoe")
        self._arg_parser.add_argument(
            "--poe_unimodal_elbos",
            default=False,
            action="store_true",
            help="unimodal_klds",
        )
        self._arg_parser.add_argument(
            "--poe_num_subset_elbos",
            type=int,
            default=0,
            help="Number of subset elbo terms to compute. For <= 0, no subsets terms will be computed.",
        )
        self._arg_parser.add_argument(
            "--include_prior_expert",
            action="store_true",
            default=False,
            help="Include prior in product of experts",
        )
        self._arg_parser.add_argument(
            "--factorized_representation",
            action="store_true",
            default=False,
            help="factorized_representation",
        )
        self._arg_parser.add_argument(
            "--poe_no_product_prior",
            dest="poe_product_prior",
            default=True,
            action="store_false",
            help="If false, omit the standard normal prior that is part of the product distribution",
        )
        self._arg_parser.add_argument(
            "--latent_constant_variances",
            type=float,
            default=None,
            help="Fix the latent variances to multiple of identity matrices.",
        )
        self._arg_parser.add_argument(
            "--latent_variances_offset",
            type=float,
            default=None,
            help="Adds constant diagonal matrix to latent variances",
        )
        self._arg_parser.add_argument(
            "--anneal_latent_variance_by_epoch",
            type=int,
            default=None,
            help="If provided, the latent_variace_offset is linearly reduced to 0 by the epoch provided by this parameter.",
        )
        self._arg_parser.add_argument(
            "--expert_temperature",
            type=float,
            default=None,
            help="Temperature for expert distributions. Larger is closer to uniform.",
        )
        self._arg_parser.add_argument(
            "--poe_normalize_experts",
            default=False,
            action="store_true",
            help="Always set temperature to number of experts.",
        )
        self._arg_parser.add_argument(
            "--force_partition_latent_space_mode",
            type=str,
            default=None,
            choices=["variance", "concatenate"],
            help='Forces experts to partition the latent space. In "variance" mode, the experts will partition the latent representation by forcing high variance and zero means on unassigned coordinates. In "concatenate" mode, the experts outputs will be concatenated, which is equivalent to "variance" mode with infinite variance.',
        )
        self._arg_parser.add_argument(
            "--partition_latent_space_variance_offset",
            type=float,
            default=0,
            help="Constant variance value to add to variance coordinates not in the experts partition",
        )
        self._arg_parser.add_argument(
            "--no_stable_poe",
            default=True,
            dest="stable_poe",
            action="store_false",
            help="Use more stable poe method.",
        )
        self._arg_parser.add_argument(
            "--poe_variance_clipping",
            type=float,
            nargs=2,
            default=[float("inf"), float("inf")],
            help="Clip the variance values to fixed range [a, b]. If a or b is not clipped (=inf), then it is not limited in this direction.",
        )

        # LOSS TERM WEIGHTS
        self._arg_parser.add_argument(
            "--beta",
            type=float,
            default=5.0,
            help="default weight of sum of weighted divergence terms",
        )
        self._arg_parser.add_argument(
            "--beta_style",
            type=float,
            default=1.0,
            help="default weight of sum of weighted style divergence terms",
        )
        self._arg_parser.add_argument(
            "--beta_content",
            type=float,
            default=1.0,
            help="default weight of sum of weighted content divergence terms",
        )

        self._arg_parser.add_argument(
            "--tcmvae",
            default=False,
            action="store_true",
            help="Decompose ELBO KL term into Index-Code MI, TC and dimension wise KL, i.e. TCMVAE",
        )
        self._arg_parser.add_argument(
            "--elbo_add_tc",
            default=False,
            action="store_true",
            help="Use standard ELBO, but add the total-correlation term to the ELBO."
        )

        self._arg_parser.add_argument(
            "--index_code_mi_weight",
            type=float,
            default=1,
            help="TCMVAE: weight for Index-Code MI. Only used if --tcmvae is set",
        )

        self._arg_parser.add_argument(
            "--total_correlation_weight",
            type=float,
            default=1,
            help="TCMVAE: weight of total-correlation term. Only used if --tcmvae is set",
        )

        self._arg_parser.add_argument(
            "--dimension_wise_kld_weight",
            type=float,
            default=1,
            help="TCMVAE: weight of dimension-wise KL term. Only used if --tcmvae is set",
        )

        self._arg_parser.add_argument(
            "--total_correlation_weight_target_epoch",
            type=int,
            default=-1,
            help="Linearly increase the TC weight by epoch from 1 to full value. For TCMVAE or MAVE with TC weight."
        )

        self._arg_parser.add_argument(
            "--total_correlation_weight_wait_epochs",
            type=int,
            default=0,
            help="Wait given number of epochs before setting the total correlation weight to something other than 1. For TCMVAE or MAVE with TC weight."
        )


        self.init_parser(self._arg_parser)

    def _parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        flags = self._arg_parser.parse_args(args)

        flags = self._post_process_flags(flags)
        return flags

    def _post_process_flags(self, flags: argparse.Namespace) -> argparse.Namespace:
        if flags.method == "poe":
            flags.modality_poe = True
            # flags.poe_unimodal_elbos = True
        elif flags.method == "moe":
            flags.modality_moe = True
        elif flags.method == "jsd":
            flags.modality_jsd = True
        elif flags.method == "joint_elbo":
            flags.joint_elbo = True
        else:
            raise NotImplemented(f"Method '{flags.method}' not implemented!")

        # FIXME: This is illegal
        flags = create_dir_structure(flags)

        flags.num_features = len(self.alphabet)
        flags.device = self._device

        return self.post_process_flags(flags)
