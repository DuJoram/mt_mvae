import argparse
import shutil
import subprocess
import typing
from dataclasses import dataclass, field
from math import ceil


@dataclass
class ExperimentConfig:
    name: str = None
    params: typing.Dict = field(default_factory=dict)
    main_script: str = None
    max_time: str = "24:00"
    n_cpus: int = 8
    n_gpus: int = 1
    total_mem: int = 32768
    total_scratch: int = 16384
    runs_per_config: int = 3
    sweep: typing.Dict = field(default_factory=dict)
    _sweep_list: typing.List = field(default_factory=list, init=False)
    base_params: typing.Dict = None
    experiment_class: str = None

    def generate_value_sets(self) -> typing.Dict:
        self._sweep_list = list(self.sweep.items())
        param_lists = self._generate_value_sets(item_idx=0)
        param_lists_repeated = dict()
        for param in param_lists:
            param_lists_repeated[param] = list()

            for value in param_lists[param]:
                param_lists_repeated[param] += self.runs_per_config * [value]
        return param_lists_repeated

    def _generate_value_sets(self, item_idx: int = 0) -> typing.Dict[str, typing.List]:
        param, values = self._sweep_list[item_idx]
        if item_idx == (len(self._sweep_list) - 1):
            return {param: values}

        sub_value_sets = self._generate_value_sets(item_idx + 1)
        value_sets = dict()

        value_sets[param] = list()
        for value in values:
            for idx, sub_param in enumerate(sub_value_sets):

                value_sets[param] += [value] * len(sub_value_sets[sub_param])
                if sub_param not in value_sets:
                    value_sets[sub_param] = list()

                value_sets[sub_param] += sub_value_sets[sub_param]

        return value_sets

    def num_runs(self) -> int:
        return len(list(self.generate_value_sets().values())[0])

    def generate_sweep_bash_arrays(self) -> str:
        sweep_values = self.generate_value_sets()

        job_index_to_param_string = ""

        for param, param_values in sweep_values.items():
            bash_param_name = param.replace("-", "_")
            job_index_to_param_string += (
                f"job_index_to_{bash_param_name}=(0 "
                + " ".join(map(str, param_values))
                + ")\n"
            )

            job_index_to_param_string += f"value_{bash_param_name}=\\${{job_index_to_{bash_param_name}[\\$JOB_INDEX]}}\n\n"

        return job_index_to_param_string

    def generate_command_arguments_string(self) -> str:
        arguments = ""

        params = self.base_params.copy()
        params.update(self.params)

        for param_name in self.sweep:
            bash_param_name = param_name.replace("-", "_")

            if param_name in params:
                del params[param_name]

            arguments += f" --{param_name} \\${{value_{bash_param_name}}}"

        for param, param_value in params.items():
            arguments += f" --{param} {param_value}"

        return arguments


@dataclass
class PolyMNISTExperimentConfig(ExperimentConfig):
    main_script: str = "src/main_mmnist_vanilla.py"
    max_time: str = "24:00"
    experiment_class: str = "MMNIST"
    base_params: typing.Dict = field(
        default_factory=lambda: {
            "dataset-archive": "resources/data/PolyMNIST.zip",
            "num-mods": 5,
            "pretrained-classifier-paths": '"resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m0" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m1" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m2" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m3" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m4"',
            "dir_experiment": '"\\${DIR_EXPERIMENT}"',
            "inception_state_dict": "resources/pt_inception-2015-12-05-6726825d.pth",
            "method": "poe",
            "style_dim": 0,
            "class_dim": 512,
            "beta": 1,
            "likelihood": "laplace",
            "batch_size": 256,
            "initial_learning_rate": 0.0005,
            "eval_freq": 100,
            "plotting_freq": 100,
            "data_multiplications": 1,
            "num_hidden_layers": 1,
            "end_epoch": 1000,
            "eval_lr": "",
            "calc_nll": "",
            "checkpoint_frequency": 100,
        },
        init=False,
    )


@dataclass
class ThesisPolyMNISTExperimentConfig(ExperimentConfig):
    main_script: str = "src/main_mmnist_vanilla.py"
    max_time: str = "24:00"
    experiment_class: str = "MMNIST"
    base_params: typing.Dict = field(
        default_factory=lambda: {
            "dataset-archive": "resources/data/PolyMNIST.zip",
            "num-mods": 5,
            "pretrained-classifier-paths": '"resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m0" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m1" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m2" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m3" "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m4"',
            "dir_experiment": '"\\${DIR_EXPERIMENT}"',
            "inception_state_dict": "resources/pt_inception-2015-12-05-6726825d.pth",
            "method": "poe",
            "style_dim": 0,
            "class_dim": 128,
            "beta": 1,
            "likelihood": "laplace",
            "batch_size": 1024,
            "initial_learning_rate": 5e-3,
            "eval_freq": 100,
            "plotting_freq": 100,
            "data_multiplications": 1,
            "num_hidden_layers": 1,
            "end_epoch": 1000,
            "eval_lr": "",
            "calc_nll": "",
            "checkpoint_frequency": 100,
        },
        init=False,
    )


@dataclass
class MdSpritesExperimentConfig(ExperimentConfig):
    main_script: str = "src/main_mdsprites.py"
    experiment_class: str = "MdSprites"
    base_params: typing.Dict = field(
        default_factory=lambda: {
            "dsprites-archive": '"resources/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"',
            "pretrained-classifier": '"resources/trained_classifiers/trained_classifiers_dsprites/dsprites_classifier"',
            "dir_experiment": '"\\${DIR_EXPERIMENT}"',
            "inception_state_dict": "resources/pt_inception-2015-12-05-6726825d.pth",
            "method": "poe",
            "style_dim": 0,
            "class_dim": 100,
            "beta": 1,
            "likelihood": "bernoulli",
            "batch_size": 1024,
            "initial_learning_rate": 0.0005,
            "eval_freq": 100,
            "plotting_freq": 100,
            "end_epoch": 1000,
            "eval_lr": "",
            "calc_nll": "",
            "checkpoint_frequency": 200,
            "encoder_num_hidden_layers": 3,
            "decoder_num_hidden_layers": 5,
        },
        init=False,
    )


@dataclass
class ThesisMdSpritesExperimentConfig(ExperimentConfig):
    main_script: str = "src/main_mdsprites.py"
    experiment_class: str = "MdSprites"
    base_params: typing.Dict = field(
        default_factory=lambda: {
            "dsprites-archive": '"resources/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"',
            "pretrained-classifier": '"resources/trained_classifiers/trained_classifiers_dsprites/dsprites_classifier"',
            "dir_experiment": '"\\${DIR_EXPERIMENT}"',
            "inception_state_dict": "resources/pt_inception-2015-12-05-6726825d.pth",
            "method": "poe",
            "style_dim": 0,
            "class_dim": 64,
            "beta": 1,
            "likelihood": "bernoulli-logits",
            "batch_size": 1024,
            "initial_learning_rate": 0.0005,
            "eval_freq": 100,
            "plotting_freq": 100,
            "end_epoch": 1000,
            "eval_lr": "",
            "calc_nll": "",
            "checkpoint_frequency": 200,
            "encoder_num_hidden_layers": 2,
            "decoder_num_hidden_layers": 3,
        },
        init=False,
    )


EXPERIMENTS = [
    PolyMNISTExperimentConfig(
        name="poe-fixed-temps",
        sweep={"expert_temperature": [0.5, 1, 8, 32, 128]},
        params={"no_stable_poe": ""},
    ),
    PolyMNISTExperimentConfig(
        name="poe-reduced-latent-space",
        sweep={"class_dim": [2, 32, 64, 128, 256]},
        params={"no_stable_poe": ""},
    ),
    PolyMNISTExperimentConfig(
        name="poe-var-offset-annealing",
        sweep={"latent_constant_variances": [0, 1, 8, 32, 128]},
        params={"anneal_latent_variance_by_epoch": 700, "no_stable_poe": ""},
    ),
    PolyMNISTExperimentConfig(
        name="poe-partitioned-latent-space-variance",
        sweep={"expert_temperature": [1.0]},
        params={
            "force_partition_latent_space_mode": "variance",
            "class_dim": 510,
            "no_stable_poe": "",
        },
        runs_per_config=10,
    ),
    PolyMNISTExperimentConfig(
        name="poe-partitioned-latent-space-concatenate",
        sweep={"expert_temperature": [1.0]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "class_dim": 510,
            "no_stable_poe": "",
        },
        runs_per_config=5,
    ),
    PolyMNISTExperimentConfig(
        name="poe-constant-variance",
        sweep={"latent_constant_variances": [0.01, 1, 16, 64, 128]},
        params={"no_stable_poe": ""},
        max_time="48:00",
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable",
        sweep={"expert_temperature": [1.0]},
        runs_per_config=5,
    ),
    PolyMNISTExperimentConfig(
        name="poe-variance-clipping",
        sweep={"expert_temperature": [1.0]},
        params={"poe_variance_clipping": "-40 40", "no_stable_poe": ""},
        runs_per_config=5,
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-variance-clipping",
        sweep={"expert_temperature": [1.0]},
        params={"poe_variance_clipping": "-80 80"},
        runs_per_config=5,
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-beta-sweep",
        sweep={"beta": [0.5, 1, 2, 4, 8]},
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-subsampled-beta-sweep",
        sweep={"beta": [0.5, 1, 2, 4, 8]},
        params={"poe_num_subset_elbos": 5, "poe_unimodal_elbos": ""},
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-variance-clipping-beta-sweep",
        sweep={"beta": [0.5, 1, 2, 4, 8]},
        params={
            "poe_variance_clipping": "-50 50",
        },
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-variance-clipping-partitioned-latent-space-concatenate",
        sweep={"expert_temperature": [1.0]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "class_dim": 510,
            "poe_variance_clipping": "-50 50",
        },
        runs_per_config=5,
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-partitioned-latent-space-concatenate-beta-sweep",
        sweep={"beta": [0.5, 1, 2, 4, 8]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "class_dim": 510,
        },
        runs_per_config=3,
    ),
    PolyMNISTExperimentConfig(
        name="poe-stable-subsampled-partitioned-latent-space-concatenate-beta-sweep",
        sweep={"beta": [0.5, 1, 2, 4, 8]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "poe_num_subset_elbos": 5,
            "poe_unimodal_elbos": "",
            "class_dim": 510,
        },
        max_time="48:00",
    ),
    PolyMNISTExperimentConfig(
        name="testing", sweep={"end_epoch": [8]}, max_time="00:30", runs_per_config=1
    ),
    MdSpritesExperimentConfig(
        name="mdsprites-testing",
        params={
            "eval_freq": 10,
            "end_epoch": 100,
        },
        sweep={"beta": [0.5, 1, 2]},
        runs_per_config=2,
    ),
    MdSpritesExperimentConfig(
        name="mvae-p-mdsprites-beta-sweep",
        sweep={"beta": [2**-8, 2**-4, 2**-2, 1, 2, 4, 8]},
        params={
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 10,
            "shared-attributes": "shape scale posX",
            "encoder_num_hidden_layers": 2,
            "decoder_num_hidden_layers": 4,
        },
        max_time="30:00",
    ),
    MdSpritesExperimentConfig(
        name="mvae-mdsprites-beta-sweep",
        sweep={"beta": [2**-8, 2**-4, 2**-2, 1, 2, 4, 8]},
        params={
            "shared-attributes": "shape scale posX",
            "encoder_num_hidden_layers": 2,
            "decoder_num_hidden_layers": 4,
        },
    ),
    MdSpritesExperimentConfig(
        name="mvae-mdsprites-shared-attributes-sweep",
        sweep={
            "shared-attributes": [
                "'shape scale orientation posX posY'",
                "'shape scale orientation posX'",
                "'shape scale orientation'",
                "'shape scale'",
                "'shape'",
            ]
        },
        params={
            "batch_size": 512,
        },
        runs_per_config=3,
    ),
    MdSpritesExperimentConfig(
        name="mvae-mdsprites-partitioned-variance",
        sweep={
            "partition_latent_space_variance_offset": [0, 1, 10, 100, 1000],
            "class_dim": [10, 50, 100, 200],
        },
        params={
            "shared-attributes": "scale orientation posX",
            "force_partition_latent_space_mode": "variance",
        },
    ),
    MdSpritesExperimentConfig(
        name="mvae-p-mdsprites-partitioned-variance",
        sweep={
            "partition_latent_space_variance_offset": [0, 1, 10, 100, 1000],
            "class_dim": [10, 50, 100, 200],
        },
        params={
            "shared-attributes": "scale orientation posX",
            "force_partition_latent_space_mode": "variance",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 10,
        },
        max_time="30:00",
    ),
    MdSpritesExperimentConfig(
        name="mvae-mdsprites-partitioned-concatenate",
        sweep={"beta": [2**-8, 2**-4, 2**-2, 1, 2, 4, 8]},
        params={"force_partition_latent_space_mode": "concatenate"},
    ),
    MdSpritesExperimentConfig(
        name="mvae-p-mdsprites-partitioned-concatenate",
        sweep={"beta": [2**-8, 2**-4, 2**-2, 1, 2, 4, 8]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 10,
        },
        max_time="30:00",
    ),
    MdSpritesExperimentConfig(
        name="vae-mdsprites-beta-latent-size-sweep",
        sweep={"beta": [0.001, 0.01, 0.1, 1, 5, 10], "class_dim": [8, 16, 32, 64, 128]},
        params={
            "num_mods": 1,
            "encoder_num_hidden_layers": 2,
            "decoder_num_hidden_layers": 4,
            "poe_no_product_prior": "",
        },
        max_time="06:00",
    ),
    MdSpritesExperimentConfig(
        name="mave-p-mdsprites-convergence-test",
        sweep={"beta": [1]},
        params={
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 10,
            "end_epoch": 2000,
            "eval_freq": 250,
        },
        max_time="72:00",
    ),
    MdSpritesExperimentConfig(
        name="mave-mdsprites-convergence-test",
        sweep={"beta": [1]},
        params={"end_epoch": 2000, "eval_freq": 250},
        max_time="72:00",
    ),
    MdSpritesExperimentConfig(
        name="mvae-p-mdsprites-partitioned-variance-offset-sweep",
        sweep={"partition_latent_space_variance_offset": [0, 1, 10, 100, 1000]},
        params={
            "shared-attributes": "scale orientation posX",
            "force_partition_latent_space_mode": "variance",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 10,
        },
        max_time="30:00",
    ),
    MdSpritesExperimentConfig(
        name="mvae-mdsprites-partitioned-variance-offset-sweep",
        sweep={"partition_latent_space_variance_offset": [0, 1, 10, 100, 1000]},
        params={
            "shared-attributes": "scale orientation posX",
            "force_partition_latent_space_mode": "variance",
        },
        max_time="24:00",
    ),
    MdSpritesExperimentConfig(
        name="mvae-mdsprites-variance-offset-annealing",
        sweep={"latent_variances_offset": [0, 1, 4, 16, 64, 256]},
        params={"anneal_latent_variance_by_epoch": 150},
    ),
    MdSpritesExperimentConfig(
        name="mvae-p-mdsprites-variance-offset-annealing",
        sweep={"latent_variances_offset": [0, 1, 4, 16, 64, 256]},
        params={
            "anneal_latent_variance_by_epoch": 150,
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 10,
        },
    ),
    MdSpritesExperimentConfig(
        name="tcmvae-p-mdsprites-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8]},
        params={
            "batch_size": 512,
            "class_dim": 64,
            "tcmvae": "",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
        max_time="48:00",
    ),
    MdSpritesExperimentConfig(
        name="tcmvae-mdsprites-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8]},
        params={
            "batch_size": 1024,
            "class_dim": 64,
            "tcmvae": "",
        },
        max_time="48:00",
    ),
    PolyMNISTExperimentConfig(
        name="tcmvae-p-mmnist-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8]},
        params={
            "batch_size": 512,
            "tcmvae": "",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
            "class_dim": 128,
        },
        max_time="48:00",
    ),
    PolyMNISTExperimentConfig(
        name="tcmvae-mmnist-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8]},
        params={
            "batch_size": 1024,
            "tcmvae": "",
            "class_dim": 128,
        },
        max_time="48:00",
    ),
    PolyMNISTExperimentConfig(
        name="mvae-tc-mmnist-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8]},
        params={
            "batch_size": 512,
            "class_dim": 128,
            "elbo_add_tc": "",
        },
    ),
    PolyMNISTExperimentConfig(
        name="mvae-tc-p-mmnist-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8]},
        params={
            "batch_size": 512,
            "class_dim": 128,
            "elbo_add_tc": "",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-mvae-mmnist-variance-offset-sweep",
        sweep={
            "latent_variances_offset": [0, 2**0, 2**1, 2**2, 2**4, 2**6],
        },
        params={
            "anneal_latent_variance_by_epoch": 300,
            "eval_freq": 50,
            "checkpoint_frequency": 50,
        },
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-mvae-p-mmnist-variance-offset-sweep",
        sweep={
            "latent_variances_offset": [0, 2**0, 2**1, 2**2, 2**4, 2**6],
        },
        params={
            "anneal_latent_variance_by_epoch": 300,
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
            "eval_freq": 50,
            "checkpoint_frequency": 50,
        },
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-tcmvae-mdsprites-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8, 16]},
        params={
            "tcmvae": "",
            "total_correlation_weight_target_epoch": 200,
            "total_correlation_weight_wait_epochs": 100,
        },
        max_time="48:00",
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-tcmvae-p-mdsprites-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8, 16]},
        params={
            "tcmvae": "",
            "total_correlation_weight_target_epoch": 200,
            "total_correlation_weight_wait_epochs": 100,
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
        max_time="48:00",
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-tcmvae-mmnist-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8, 16]},
        params={
            "tcmvae": "",
            "total_correlation_weight_target_epoch": 200,
            "total_correlation_weight_wait_epochs": 100,
            "batch_size": 512,
        },
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-tcmvae-p-mmnist-total-correlation-weight-sweep",
        sweep={"total_correlation_weight": [0, 1, 2, 4, 8, 16]},
        params={
            "tcmvae": "",
            "total_correlation_weight_target_epoch": 200,
            "total_correlation_weight_wait_epochs": 100,
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
            "batch_size": 512,
        },
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-catmvae-mmnist-actually",
        sweep={
            "beta": [1],
        },
        params={
            "force_partition_latent_space_mode": "concatenate",
            "batch_size": 512,
        },
        runs_per_config=5,
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-catmvae-p-mmnist-actually",
        sweep={"beta": [1]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
            "batch_size": 512,
        },
        runs_per_config=5,
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-catmvae-mmnist",
        sweep={"beta": [1]},
        params={"force_partition_latent_space_mode": "concatenate"},
        runs_per_config=5,
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-catmvae-p-mmnist",
        sweep={"beta": [1]},
        params={
            "force_partition_latent_space_mode": "concatenate",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
        runs_per_config=5,
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-mvae-mmnist-beta-sweep",
        sweep={
            "beta": [2**-4, 2**-2, 1, 2, 4, 8],
        },
        params={
            "batch_size": 512,
        },
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-mvae-p-mmnist-beta-sweep",
        sweep={
            "beta": [2**(-4), 2**(-2), 1, 2, 4, 8],
        },
        params={
            "batch_size": 512,
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-mvae-mdsprites-beta-sweep",
        sweep={
            "beta": [2**-4, 2**-2, 1, 2, 4, 8],
        },
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-mvae-p-mdsprites-beta-sweep",
        sweep={
            "beta": [2**(-4), 2**(-2), 1, 2, 4, 8],
        },
        params={
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-vpmvae-mmnist-variance-offset-sweep",
        sweep={
            "partition_latent_space_variance_offset": [0, 1, 10, 100, 1000]
        },
        params={
            "force_partition_latent_space_mode": "variance",
            "batch_size": 512,
        }
    ),
    ThesisPolyMNISTExperimentConfig(
        name="thesis-vpmvae-p-mmnist-variance-offset-sweep",
        sweep={
            "partition_latent_space_variance_offset": [0, 1, 10, 100, 1000]
        },
        params={
            "force_partition_latent_space_mode": "variance",
            "batch_size": 512,
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        }
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-vpmvae-mdsprites-variance-offset-sweep",
        sweep={
            "partition_latent_space_variance_offset": [0, 1, 10, 100, 1000]
        },
        params={
            "force_partition_latent_space_mode": "variance",
        },
    ),
    ThesisMdSpritesExperimentConfig(
        name="thesis-vpmvae-p-mdsprites-variance-offset-sweep",
        sweep={
            "partition_latent_space_variance_offset": [0, 1, 10, 100, 1000]
        },
        params={
            "force_partition_latent_space_mode": "variance",
            "poe_unimodal_elbos": "",
            "poe_num_subset_elbos": 5,
        },
    ),
]


def generate_job_script_header(
    experiment: ExperimentConfig,
    num_runs: int,
    experiment_name_suffix: str = None,
    job_system: str = None,
    job_index_subset: typing.Optional[typing.List[int]] = None,
    max_jobs: int = None,
    dependency: str = None,
) -> str:
    mem_per_cpu = int(ceil(experiment.total_mem / experiment.n_cpus))
    scratch_per_cpu = int(ceil(experiment.total_scratch / experiment.n_cpus))
    experiment_name = experiment.name
    if experiment_name_suffix is not None:
        experiment_name += f"_{experiment_name_suffix}"
    header = ""

    workdir_init = """
if [[ -z "\\${WORK_DIR}" ]]; then
    WORK_DIR="/tmp"
fi

if [[ ! -d "\\${WORK_DIR}" ]]; then
    echo "ERROR: WORK_DIR \\${WORK_DIR} does not exist! Exiting..."
    exit -1
fi
    """

    job_array_indices = f"1-{num_runs}"
    if job_index_subset is not None:
        job_array_indices = ",".join(map(str, job_index_subset))

    if job_system == "lsf":
        gpu_option = ""
        if experiment.n_gpus > 0:
            gpu_option = f'#BSUB -R "rusage[ngpus_excl_p={experiment.n_gpus}]"'

        header = f"""#BSUB -n {experiment.n_cpus}
#BSUB -R "rusage[mem={mem_per_cpu},scratch={scratch_per_cpu}]"
#BSUB -W {experiment.max_time}
#BSUB -J "${{USER}}_{experiment_name}[{job_array_indices}]"
#BSUB -o {experiment_name}_%J.%I.out
#BSUB -e {experiment_name}_%J.%I.stderr.out
{gpu_option}

set -eo pipefail
shopt -s nullglob globstar

JOB_ID=\\${{LSB_JOBID}}
JOB_INDEX=\\${{LSB_JOBINDEX}}
JOB_NAME=\\${{LSB_JOBNAME}}
WORK_DIR=\\${{TMPDIR}}
SUBMISSION_DIR=\\${{LS_SUBCWD}}

{workdir_init}

echo "WORK_DIR: \\${{WORK_DIR}}"
cd \\${{WORK_DIR}}

echo "Copying resources..."
rsync -aq \\${{SUBMISSION_DIR}}/resources ./

echo "Extracting data..."
unzip -qn \\${{SUBMISSION_DIR}}/resources/data/PolyMNIST.zip -d resources/data/

echo "Copying sources..."
rsync -aq \\${{SUBMISSION_DIR}}/src ./
"""
    elif job_system == "slurm":
        partition = ""
        gpu_option = ""
        if experiment.n_gpus > 0:
            partition = "#SBATCH -p gpu\n"
            gpu_option = "#SBATCH --gres=gpu:1\n"
            gpu_option += "#SBATCH --exclude=gpu-biomed-[02-13,19-22],gpu-biomed-[16-17],gpu-biomed-18\n"

        dependency_option = ""
        if dependency is not None:
            dependency_option = f"#SBATCH -d {dependency}\n"

        max_jobs_suffix = ""
        if max_jobs is not None:
            max_jobs_suffix = f"%{max_jobs}"

        header = f"""#!/bin/bash
#SBATCH --cpus-per-task {experiment.n_cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={experiment.max_time}:00
#SBATCH --job-name="${{USER}}_{experiment_name}[1-{num_runs}]"
#SBATCH --array={job_array_indices}{max_jobs_suffix}
#SBATCH -o {experiment_name}_%A.%a.log
#SBATCH -e {experiment_name}_%A.%a.stderr.log
{dependency_option}{partition}{gpu_option}

set -eo pipefail
shopt -s nullglob globstar

JOB_ID=\\${{SLURM_ARRAY_JOB_ID}}
JOB_INDEX=\\${{SLURM_ARRAY_TASK_ID}}
JOB_NAME=\\${{SLURM_JOB_NAME}}
WORK_DIR=\\${{PWD}}
SUBMISSION_DIR=\\${{PWD}}

{workdir_init}
echo "WORK_DIR: \\${{WORK_DIR}}"
cd \\${{WORK_DIR}}

"""
    else:
        header = f"""
set -eo pipefail
shopt -s nullglob globstar

USER=joraml

if [[ -z "\${{JOB_ID}}" ]]; then
    JOB_ID=1234567
fi

echo "JOB_INDEX=\$JOB_INDEX"

if [[ -z "\${{JOB_INDEX}}" ]]; then
    echo "Setting job index to 1"
    JOB_INDEX={1 if job_index_subset is None else job_array_indices}
fi

WORK_DIR=\\${{PWD}}
SUBMISSION_DIR=\\${{PWD}}
"""
    return header


def generate_latent_representation_command(
    experiment: ExperimentConfig,
    job_id: str,
) -> typing.Tuple[str, str, str, str]:
    num_runs = experiment.num_runs()

    experiment_job_name = f"${{USER}}_{experiment.name}[1-{num_runs}]"

    job_index_to_params_string = experiment.generate_sweep_bash_arrays()

    bash_variables = f"""
DATA_DIR="resources/data"
EXPERIMENT_LOGDIR_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}"
EXPERIMENT_ARCHIVE_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}.tar.gz"
DIR_EXPERIMENT="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_latent_partitioning"
TARGET_LOG_FILE_DIR="\\${{SUBMISSION_DIR}}/runs"
TARGET_LOG_FILE="\\${{TARGET_LOG_FILE_DIR}}/\\${{EXPERIMENT_LOGDIR_NAME}}_latent_partitioning.tar.gz"
LOG_DIR="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_latent_partitioning"
"""

    post_copy_command = f"""
echo "Extracting \\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}} to \\${{PWD}}/runs/"
tar -xzf "\\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}}" -C runs/
"""

    cleanup_command = f"""echo "Backing up offline runs from \\${{WANDB_DIR}} to \\${{TARGET_LOG_FILE}}"
    tar -czf "\\${{TARGET_LOG_FILE}}" -C "runs/" "\\${{EXPERIMENT_LOGDIR_NAME}}_latent_partitioning"
    """

    python_call = f"""
python src/main_calc_latent_partitioning.py \\
    --num_workers {experiment.n_cpus} \\
    --run-path "runs/\\${{EXPERIMENT_LOGDIR_NAME}}/*" \\
    --log-dir "\\${{DIR_EXPERIMENT}}" \\
    --run-group "{experiment.name}" \\
    --run-name "{experiment.name}-{job_id}.\\${{JOB_INDEX}}" \\
    --plot-latents-projections \\
    --unimodal-datapaths-train "resources/data/MMNIST/train/m"{{0..4}} \\
    --unimodal-datapaths-test "resources/data/MMNIST/test/m"{{0..4}} \\
    --pretrained-classifier-paths "resources/trained_classifiers/trained_classifiers_polyMNIST/pretrained_img_to_digit_classifier_m"{{0..4}} \\
"""

    return bash_variables, post_copy_command, python_call, cleanup_command


def generate_experiment_command(
    experiment: ExperimentConfig,
) -> typing.Tuple[str, str, str, str]:
    num_runs = experiment.num_runs()

    experiment_job_name = f"${{USER}}_{experiment.name}[1-{num_runs}]"

    job_index_to_params_string = experiment.generate_sweep_bash_arrays()

    bash_variables = f"""
{job_index_to_params_string}

DATA_DIR="resources/data"
EXPERIMENT_LOGDIR_NAME="log_\\${{JOB_NAME}}_\\${{JOB_ID}}.\\${{JOB_INDEX}}"
TARGET_LOG_FILE_DIR="\\${{SUBMISSION_DIR}}/runs"
TARGET_LOG_FILE="\\${{TARGET_LOG_FILE_DIR}}/\\${{EXPERIMENT_LOGDIR_NAME}}.tar.gz"
DIR_EXPERIMENT="runs/\\${{EXPERIMENT_LOGDIR_NAME}}"
"""

    python_call = f"python {experiment.main_script} --run_name {experiment.name}-\\${{JOB_ID}}.\\${{JOB_INDEX}} --num_workers {experiment.n_cpus}"
    python_call += experiment.generate_command_arguments_string()

    cleanup_command = f"""echo "Backing up offline runs from \\${{WANDB_DIR}} to \\${{TARGET_LOG_FILE}}"
    tar -czf "\\${{TARGET_LOG_FILE}}" -C "runs/" "\\${{EXPERIMENT_LOGDIR_NAME}}"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}"
"""

    return bash_variables, "", python_call, cleanup_command


def generate_disentanglement_command(
    experiment: ExperimentConfig,
    job_id: str,
) -> typing.Tuple[str, str, str, str]:
    num_runs = experiment.num_runs()

    experiment_job_name = f"${{USER}}_{experiment.name}[1-{num_runs}]"

    job_index_to_params_string = experiment.generate_sweep_bash_arrays()

    experiment.max_time = "08:00"
    experiment.n_gpus = 0

    bash_variables = f"""
{job_index_to_params_string}
    
DATA_DIR="resources/data"
EXPERIMENT_LOGDIR_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}"
EXPERIMENT_ARCHIVE_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}.tar.gz"
DIR_EXPERIMENT="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_disentanglement"
TARGET_LOG_FILE_DIR="\\${{SUBMISSION_DIR}}/runs"
TARGET_LOG_FILE="\\${{TARGET_LOG_FILE_DIR}}/\\${{EXPERIMENT_LOGDIR_NAME}}_disentanglement.tar.gz"
LOG_DIR="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_disentanglement"
"""

    post_copy_command = f"""
mkdir -p \\${{PWD}}/runs/
echo "Extracting \\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}} to \\${{PWD}}/runs/"
tar -xzf "\\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}}" -C runs/
"""

    cleanup_command = f"""echo "Backing up offline runs from \\${{WANDB_DIR}} to \\${{TARGET_LOG_FILE}}"
    tar -czf "\\${{TARGET_LOG_FILE}}" -C "runs/" "\\${{EXPERIMENT_LOGDIR_NAME}}_disentanglement"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}_disentanglement"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}"
    """

    disent_experiment = (
        "MdSprites"
        if isinstance(experiment, MdSpritesExperimentConfig)
        or isinstance(experiment, ThesisMdSpritesExperimentConfig)
        else ""
    )

    experiment.params.update(
        {
            "disent-experiment": disent_experiment,
            "disent-num-workers": experiment.n_cpus,
            "disent-training-experiment-dir": '"runs/\\${EXPERIMENT_LOGDIR_NAME}/"',
            "disent-checkpoints-epochs": "last",
            "disent-num-training-samples": 100_000,
            "disent-num-test-samples": 10_000,
            "disent-num-diff-samples": 64,
            "disent-num-redundant-classifiers": 50,
            "disent-run-group": f"{experiment.name}",
            "disent-run-name": f'"{experiment.name}-{job_id}.\\${{JOB_INDEX}}"',
        }
    )

    python_call = f'python src/main_evaluate_disentanglement.py --run_name "{experiment.name}-\\${{JOB_ID}}.\\${{JOB_INDEX}}" --num_workers {experiment.n_cpus}'
    python_call += experiment.generate_command_arguments_string()

    return bash_variables, post_copy_command, python_call, cleanup_command


def generate_coherence_command(
    experiment: ExperimentConfig,
    job_id: str,
) -> typing.Tuple[str, str, str, str]:
    num_runs = experiment.num_runs()

    experiment_job_name = f"${{USER}}_{experiment.name}[1-{num_runs}]"

    job_index_to_params_string = experiment.generate_sweep_bash_arrays()

    experiment.max_time = "08:00"

    bash_variables = f"""
{job_index_to_params_string}

DATA_DIR="resources/data"
EXPERIMENT_LOGDIR_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}"
EXPERIMENT_ARCHIVE_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}.tar.gz"
DIR_EXPERIMENT="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_coherence"
TARGET_LOG_FILE_DIR="\\${{SUBMISSION_DIR}}/runs"
TARGET_LOG_FILE="\\${{TARGET_LOG_FILE_DIR}}/\\${{EXPERIMENT_LOGDIR_NAME}}_coherence.tar.gz"
LOG_DIR="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_coherence"
"""

    post_copy_command = f"""
mkdir -p \\${{PWD}}/runs/
echo "Extracting \\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}} to \\${{PWD}}/runs/"
tar -xzf "\\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}}" -C runs/
"""

    cleanup_command = f"""echo "Backing up offline runs from \\${{WANDB_DIR}} to \\${{TARGET_LOG_FILE}}"
    tar -czf "\\${{TARGET_LOG_FILE}}" -C "runs/" "\\${{EXPERIMENT_LOGDIR_NAME}}_coherence"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}_coherence"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}"
    """

    experiment.params.update(
        {
            "coher-experiment": experiment.experiment_class,
            "coher-training-experiment-dir": '"runs/\\${EXPERIMENT_LOGDIR_NAME}/"',
            "coher-checkpoints-epochs": "all",
            "coher-run-group": f"{experiment.name}",
            "coher-run-name": f'"{experiment.name}-{job_id}.\\${{JOB_INDEX}}"',
            "use_classifier": "",
        }
    )

    python_call = f'python src/main_evaluate_coherence.py --run_name "{experiment.name}-\\${{JOB_ID}}.\\${{JOB_INDEX}}" --num_workers {experiment.n_cpus}'
    python_call += experiment.generate_command_arguments_string()

    return bash_variables, post_copy_command, python_call, cleanup_command


def generate_total_correlation_command(
    experiment: ExperimentConfig,
    job_id: str,
) -> typing.Tuple[str, str, str, str]:
    num_runs = experiment.num_runs()

    experiment_job_name = f"${{USER}}_{experiment.name}[1-{num_runs}]"

    job_index_to_params_string = experiment.generate_sweep_bash_arrays()

    experiment.max_time = "04:00"

    bash_variables = f"""
{job_index_to_params_string}

DATA_DIR="resources/data"
EXPERIMENT_LOGDIR_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}"
EXPERIMENT_ARCHIVE_NAME="log_{experiment_job_name}_{job_id}.\\${{JOB_INDEX}}.tar.gz"
DIR_EXPERIMENT="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_total_correlation"
TARGET_LOG_FILE_DIR="\\${{SUBMISSION_DIR}}/runs"
TARGET_LOG_FILE="\\${{TARGET_LOG_FILE_DIR}}/\\${{EXPERIMENT_LOGDIR_NAME}}_total_correlation.tar.gz"
LOG_DIR="runs/\\${{EXPERIMENT_LOGDIR_NAME}}_total_correlation"
"""

    post_copy_command = f"""
mkdir -p \\${{PWD}}/runs/
echo "Extracting \\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}} to \\${{PWD}}/runs/"
tar -xzf "\\${{SUBMISSION_DIR}}/runs/\\${{EXPERIMENT_ARCHIVE_NAME}}" -C runs/
"""

    cleanup_command = f"""echo "Backing up offline runs from \\${{WANDB_DIR}} to \\${{TARGET_LOG_FILE}}"
    tar -czf "\\${{TARGET_LOG_FILE}}" -C "runs/" "\\${{EXPERIMENT_LOGDIR_NAME}}_total_correlation"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}_total_correlation"
    rm -rf "runs/\\${{EXPERIMENT_LOGDIR_NAME}}"
    """

    experiment.params.update(
        {
            "tc-experiment": experiment.experiment_class,
            "tc-num-workers": 8,
            "tc-training-experiment-dir": '"runs/\\${EXPERIMENT_LOGDIR_NAME}/"',
            "tc-checkpoints-epochs": "all",
            "tc-run-group": f"{experiment.name}",
            "tc-run-name": f'"{experiment.name}-{job_id}.\\${{JOB_INDEX}}"',
        }
    )

    python_call = f'python src/main_evaluate_total_correlation.py --run_name "{experiment.name}-\\${{JOB_ID}}.\\${{JOB_INDEX}}" --num_workers {experiment.n_cpus}'
    python_call += experiment.generate_command_arguments_string()

    return bash_variables, post_copy_command, python_call, cleanup_command


def main():
    experiments = {experiment.name: experiment for experiment in EXPERIMENTS}

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list-experiments", action="store_true")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        choices=list(experiments),
        help="Experiment name",
    )

    parser.add_argument(
        "-p",
        "--latent-partitioning",
        default=False,
        action="store_true",
        help="Compute latent partitioning logs for existing experiment",
    )

    parser.add_argument(
        "-d",
        "--disentanglement",
        default=False,
        action="store_true",
        help="Compute disentanglement metric",
    )

    parser.add_argument(
        "-c",
        "--coherence",
        default=False,
        action="store_true",
        help="Compute coherence metrics",
    )

    parser.add_argument(
        "-t",
        "--total-correlation",
        default=False,
        action="store_true",
        help="Estimate total correlation",
    )

    parser.add_argument("--job-id", type=str, default=None, help="LSB or SLURM job ID")

    parser.add_argument(
        "--dry-run",
        default=False,
        action="store_true",
        help="Print generated script but don't execute",
    )

    parser.add_argument(
        "--no-submit", default=True, dest="submit", action="store_false"
    )

    parser.add_argument(
        "--bash",
        default=False,
        action="store_true",
        help="Use 'bash' as job system instead of LSF/SLURM",
    )

    parser.add_argument(
        "--dependency",
        default=None,
        help="Run job only after other job has terminated. Only compatible with SLURM, uses same syntax.",
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum number of jobs to run in parallel",
    )

    parser.add_argument("--job-index-subset", type=int, nargs="+", default=None)

    args = parser.parse_args()

    if args.list_experiments:
        for experiment in experiments:
            print(experiment)
        exit(0)

    if args.experiment is None:
        print("ERROR: Missing experiment name!")
        parser.print_help()
        exit(0)

    if args.experiment not in experiments:
        print(f'ERROR: Experiment "{args.experiment}" is not defined')
        exit(-1)

    job_system = ""
    if args.bash:
        job_system = "bash"
    else:
        if shutil.which("bsub") is not None:
            job_system = "lsf"
        elif shutil.which("sbatch") is not None:
            job_system = "slurm"
        else:
            args.submit = False
            print("Failed to determine job system. Nothing will be run.")

    experiment = experiments[args.experiment]

    num_runs = experiment.num_runs()
    if args.latent_partitioning:
        (
            bash_variables,
            post_copy_command,
            python_call,
            cleanup_command,
        ) = generate_latent_representation_command(experiment, args.job_id)
        experiment_name_suffix = "latent_partitioning"
    elif args.disentanglement:
        (
            bash_variables,
            post_copy_command,
            python_call,
            cleanup_command,
        ) = generate_disentanglement_command(experiment, job_id=args.job_id)
        experiment_name_suffix = "disentanglement"
    elif args.coherence:
        (
            bash_variables,
            post_copy_command,
            python_call,
            cleanup_command,
        ) = generate_coherence_command(experiment, job_id=args.job_id)
        experiment_name_suffix = "coherence"

    elif args.total_correlation:
        (
            bash_variables,
            post_copy_command,
            python_call,
            cleanup_command,
        ) = generate_total_correlation_command(experiment, job_id=args.job_id)
        experiment_name_suffix = "total_correlation"
    else:
        (
            bash_variables,
            post_copy_command,
            python_call,
            cleanup_command,
        ) = generate_experiment_command(experiment)
        experiment_name_suffix = None

    job_script_header = generate_job_script_header(
        experiment=experiment,
        experiment_name_suffix=experiment_name_suffix,
        num_runs=num_runs,
        job_index_subset=args.job_index_subset,
        job_system=job_system,
        max_jobs=args.max_jobs,
        dependency=args.dependency,
    )
    use_venv = True
    job_system = job_system

    command = generate_command(
        experiment=experiment,
        job_script_header=job_script_header,
        python_call=python_call,
        bash_variables=bash_variables,
        post_copy_command=post_copy_command,
        cleanup_command=cleanup_command,
        job_system=job_system,
        use_venv=use_venv,
    )

    if not args.submit or args.dry_run:
        print("GENERATED SCRIPT (not executed)")
        for line_number, line in enumerate(command.split("\n")):
            print(f"{line_number - 2:03d} {line}")
        return

    subprocess.run([command], shell=True, check=False)


def generate_command(
    experiment: ExperimentConfig,
    job_script_header: str,
    python_call: str,
    bash_variables: str,
    cleanup_command: str = "",
    post_copy_command: str = "",
    job_system: str = None,
    use_venv: bool = False,
):

    activate_env_cmd = """# activate conda
eval "\\$(conda shell.bash hook)"
conda activate mt_mvae
echo "CONDA_PREFIX: \\${CONDA_PREFIX}"
"""

    command = "bash"
    if job_system == "lsf":
        command = "bsub"
    elif job_system == "slurm":
        command = "sbatch"

        if use_venv:
            activate_env_cmd = ""
            env_suffix = "gpu"
            if experiment.n_gpus == 0:
                activate_env_cmd = "module load python/3.9.6\n"
                env_suffix = "cpu"

            activate_env_cmd += f"""
source ~/venvs/mt_mvae_{env_suffix}/bin/activate
"""

    return f"""#!/usr/bin/env bash
{command} <<EOF
{job_script_header}

{activate_env_cmd}

{bash_variables}

{post_copy_command}


export WANDB_DIR="\\${{DIR_EXPERIMENT}}"
export WANDB_MODE='offline'
export WANDB_RUN_GROUP="{experiment.name}"
export WANDB_CACHE_DIR="\\${{DIR_EXPERIMENT}}"

mkdir -p "\\${{DIR_EXPERIMENT}}"
mkdir -p "\\${{TARGET_LOG_FILE_DIR}}"

function cleanup {{
    {cleanup_command}
}}

# Setup EXIT trap to store logs
trap cleanup EXIT

# Setup traps to ignore other signals since they will be sent to all processes
trap "echo 'Received USR2; ignoring'" USR2
trap "echo 'Received INT; ignoring'" INT
trap "echo 'Received QUIT; ignoring'" QUIT
trap "echo 'Received TERM; ignoring'" TERM

{python_call}

EOF
"""


if __name__ == "__main__":
    main()
