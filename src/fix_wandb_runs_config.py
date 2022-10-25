import json
import os
import typing

import torch
import wandb

import experiments.mmnist


def main():
    api = wandb.Api()
    runs = api.runs("liebeskind/mvae")

    for run in runs:
        metadata = json.load(run.file("wandb-metadata.json").download(replace=True))
        experiment = experiments.mmnist.MMNISTExperiment(args=metadata["args"])
        run_args = experiment.flags
        new_config = run_args.__dict__
        new_config.update(run.config)
        new_config = namespace_dict_to_config_dict(new_config)
        run.config.update(new_config)
        run.update()

    os.remove("wandb-metadata.json")


def namespace_dict_to_config_dict(args: typing.Dict) -> typing.Dict:
    remove_keys = list()
    for key in args:

        if isinstance(args[key], dict):
            args[key] = namespace_dict_to_config_dict(args[key])

        elif isinstance(args[key], torch.Tensor):
            args[key] = args[key].cpu().item()

        elif isinstance(args[key], torch.device):
            args[key] = str(args[key])

        elif isinstance(args[key], str):
            try:
                args[key] = float(args[key])
            except:
                pass

        elif args[key] is None:
            remove_keys.append(key)

    for key in remove_keys:
        del args[key]

    return args


if __name__ == "__main__":
    main()
