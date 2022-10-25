import itertools

import wandb

from typing import List, Dict, Union

import argparse

import numpy as np

def stderr(x: Union[np.array, List[float]]):
    return np.std(x)/np.sqrt(len(x))

def minmax(x: Union[np.array, List[float]]):
    mean = np.mean(x)
    return mean - np.min(x), np.max(x) - mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group", type=str, required=True)
    parser.add_argument("--num-mods", type=int, default=5)
    parser.add_argument("--sweep-param", type=str, required=True)
    parser.add_argument(
        "--error-type",
        choices=["stddev", "var", "minmax", "stderr"],
        default="stderr",
    )
    parser.add_argument("--wandb-entity", type=str, default="liebeskind")
    parser.add_argument("--wandb-project", type=str, default="mvae")

    args = parser.parse_args()


    errors = {
        "stddev": np.std,
        "stderr": stderr,
        "var": np.var,
        "minmax": minmax,
    }

    errfn = errors[args.error_type]

    api = wandb.Api()

    runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}", filters={"group": args.run_group})

    num_mods = args.num_mods

    sweep_values: List[float] = list()

    loo_values: Dict[float, List[float]] = dict()


    loo_keys = list(map(lambda x: ("_".join(map(lambda e: f"m{e}", x[0])), f"m{x[1]}"), [ (filter(lambda jdx: jdx != idx, range(num_mods)), idx) for idx in range(num_mods) ]))

    for run in runs:
        if "hidden" in run.tags or run.state != "finished":
            continue

        if "Generation/digit" not in run.summary:
            continue

        sweep_value = float(run.config[args.sweep_param])
        sweep_values.append(sweep_value)
        if sweep_value not in loo_values:
            loo_values[sweep_value] = list()

        for cond, gen in loo_keys:
            loo_values[sweep_value].append(run.summary["Generation/digit"][cond][gen])

    sweep_values = sorted(set(sweep_values))

    for sweep_value in sweep_values:
        print(f"{sweep_value}: {np.mean(loo_values[sweep_value])}, +- {errfn(loo_values[sweep_value])}")



if __name__ == "__main__":
    main()