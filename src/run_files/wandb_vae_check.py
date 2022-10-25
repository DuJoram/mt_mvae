from typing import Dict, List

import numpy as np
import wandb
from matplotlib import pyplot as plt
from wandb.wandb_run import Run

plt.rcParams["text.usetex"] = True


def main():
    run_group = "vae-mdsprites-beta-latent-size-sweep"

    beta_values: List[float] = list()
    class_dim_values: List[float] = list()

    api = wandb.Api()

    runs = api.runs("liebeskind/mvae", filters={"group": run_group})

    train_runs: Dict[str, Run] = dict()
    disent_runs: Dict[str, Run] = dict()

    for run in runs:
        if (
            run.job_type == "train" or run.job_type is None
        ) and run.state == "finished":
            train_runs[run.name] = run
        elif run.job_type == "disentanglement":
            disent_runs[run.name] = run

    values_refs: Dict[str, Dict[str, List[float]]] = dict()

    for run_name, train_run in train_runs.items():
        disent_run = disent_runs[run_name]

        if (
            "disentanglement/higgins/test/individuals/joint" not in disent_run.summary
            or "disentanglement"
            not in disent_run.summary["disentanglement/higgins/test/individuals/joint"]
        ):
            continue

        disentanglement = disent_run.summary[
            "disentanglement/higgins/test/individuals/joint"
        ]["disentanglement"]

        beta = float(train_run.config["beta"])
        beta_key = str(beta)
        class_dim = float(train_run.config["class_dim"])
        class_dim_key = str(class_dim)

        if beta not in beta_values:
            beta_values.append(beta)

        if class_dim not in class_dim_values:
            class_dim_values.append(class_dim)

        if beta_key not in values_refs:
            values_refs[beta_key] = dict()

        if class_dim_key not in values_refs[beta_key]:
            values_refs[beta_key][class_dim_key] = list()

        values_refs[beta_key][class_dim_key].append(disentanglement)

    beta_values = list(reversed(sorted(beta_values)))
    class_dim_values = sorted(class_dim_values)

    mat_shape = len(beta_values), len(class_dim_values)

    mat = np.zeros(mat_shape, dtype=float)

    for beta_idx, beta in enumerate(beta_values):
        beta_key = str(beta)
        for class_dim_idx, class_dim in enumerate(class_dim_values):
            class_dim_key = str(class_dim)

            if (
                beta_key not in values_refs
                or class_dim_key not in values_refs[beta_key]
            ):
                continue

            mat[beta_idx, class_dim_idx] = np.mean(values_refs[beta_key][class_dim_key])

    fig = plt.figure(dpi=196, figsize=(10, 10))
    fig.suptitle("VAE Disentanglement")

    ax = fig.subplots()
    cax = ax.imshow(mat, cmap="Blues")
    ax.set_xlabel("latent size")
    ax.set_ylabel(r"\(\beta\)")

    ax.set_xticks(
        np.arange(0, len(class_dim_values)), labels=map(str, class_dim_values)
    )
    ax.set_yticks(np.arange(0, len(beta_values)), labels=map(str, beta_values))

    cbar = fig.colorbar(cax, ticks=np.linspace(0, 1, 5))
    cbar.ax.set_yticklabels(list(map(str, np.linspace(0, 1, 5))))

    fig.savefig(f"vae_beta_latent_disent.png")


if __name__ == "__main__":
    main()
