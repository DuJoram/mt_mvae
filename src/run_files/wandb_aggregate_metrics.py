import argparse
from typing import Dict, List

import matplotlib as mpl
import wandb
from matplotlib import pyplot as plt
from wandb.wandb_run import Run

mpl.use("pgf")
plt.rcParams.update({
    "text.latex.preamble": "\n".join([
        r'\usepackage{mathpazo}',
    ]),
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.size": 11,
    "font.serif": "Palatino",
    "font.sans-serif": "sans-serif",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.serif": "Palatino",
    "font.sans-serif": "sans-serif",
    "figure.dpi": 300,
    # "font.size": 11,
    # "axes.titlesize": 14,
    # "axes.labelsize": 10,
    # "xtick.labelsize": 8,
    # "ytick.labelsize": 8,
    # "legend.fontsize": 10,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.pad_inches": 1,
    "figure.figsize": (2.5, 2),
    "axes.linewidth": 0.3,
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.75,
    "xtick.major.width": 0.4,
    "xtick.minor.width": 0.2,
    "ytick.major.width": 0.4,
    "ytick.minor.width": 0.2,
    "lines.markersize": 2.5,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric1",
        type=str,
        default="total-correlation",
        choices=["coherence", "disentanglement", "total-correlation"],
    )
    parser.add_argument(
        "--metric2",
        type=str,
        default="coherence",
        choices=["coherence", "disentanglement", "total-correlation"],
    )

    parser.add_argument(
        "--total-correlation-subset",
        type=str,
        default="full",
        help="Subset key of total correlation subset to use",
    )

    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(
        "liebeskind/mvae",
        filters={
            "$or": [
                # {"group": "tcmvae-"}
                # {"group": "mvae-mdsprites-convergence-test"},
                # {"group": "mvae-p-mdsprites-convergence-test"},
                # {"group": "mvae-mdsprites-partitioned-variance-offset-sweep"},
                # {"group": "mvae-p-mdsprites-partitioned-variance-offset-sweep"},
                # {"group": "mvae-p-mdsprites-beta-sweep"},
                # {"group": "mvae-mdsprites-beta-sweep"},
                {"group": "tcmvae-p-mdsprites-total-correlation-weight-sweep"}
            ]
        },
    )

    metric_names = {
        "coherence": "Coherence",
        "disentanglement": "Disentanglement",
        "total-correlation": "Total Correlation",
    }

    value_ranges = {
        "coherence": r"\([0, 1]\)",
        "disentanglement": r"\([0, 1]\)",
        "total-correlation": r"\([0, \infty)\)",
    }

    arrows = {
        "coherence": r"\(\rightarrow\)",
        "disentanglement": r"\(\rightarrow\)",
        "total-correlation": r"\(\leftarrow\)",
    }

    subset_key = (
        None
        if args.total_correlation_subset == "full"
        else args.total_correlation_subset
    )

    m1_runs: Dict[str, Run] = dict()
    m2_runs: Dict[str, Run] = dict()

    for run in runs:
        if run.state == "finished":
            pass

        if not "hidden" in run.tags:
            continue

        run_metric = ""
        if (
            "Generation/loo" in run.summary
            and "all_attributes" in run.summary["Generation/loo"]
        ):
            run_metric = "coherence"
        elif (
            "disentanglement/higgins/test/individuals/joint" in run.summary
            and "disentanglement"
            in run.summary["disentanglement/higgins/test/individuals/joint"]
        ):
            run_metric = "disentanglement"
        elif "total_correlation/total_correlation/train/" in run.summary:
            run_metric = "total-correlation"

        if run_metric == args.metric1:
            m1_runs[run.name] = run
        elif run_metric == args.metric2:
            m2_runs[run.name] = run

    values_m1: List[float] = list()
    values_m2: List[float] = list()

    for run_name, m1_run in m1_runs.items():
        if run_name not in m2_runs:
            continue

        m2_run = m2_runs[run_name]

        if args.metric1 == "coherence":
            m1 = m1_run.summary["Generation/loo"]["all_attributes"]
        elif args.metric1 == "disentanglement":
            m1 = m1_run.summary["disentanglement/higgins/test/individuals/joint"][
                "disentanglement"
            ]
        elif args.metric1 == "total-correlation":
            if subset_key is None:
                longest = 0
                for key in m1_run.summary[
                    "total_correlation/total_correlation/train/"
                ].keys():
                    if len(key) > longest:
                        subset_key = key
                        longest = len(key)

            m1 = m1_run.summary["total_correlation/total_correlation/train/"][
                subset_key
            ]

        if args.metric2 == "coherence":
            m2 = m2_run.summary["Generation/loo"]["all_attributes"]
        elif args.metric2 == "disentanglement":
            m2 = m2_run.summary["disentanglement/higgins/test/individuals/joint"][
                "disentanglement"
            ]
        elif args.metric2 == "total-correlation":
            if subset_key is None:
                longest = 0
                for key in m2_run.summary[
                    "total_correlation/total_correlation/train/"
                ].keys():
                    if len(key) > longest:
                        subset_key = key
                        longest = len(key)

            m2 = m2_run.summary["total_correlation/total_correlation/train/"][
                subset_key
            ]

        values_m1.append(m1)
        values_m2.append(m2)

    fig = plt.figure()  # plt.figure(dpi=196, figsize=(7, 6))
    ax = fig.subplots()

    ax.scatter(values_m1, values_m2)

    ax.axhline(y=0.177, linestyle='dashed', color='slategray')

    ax.set_xlabel(
        f"{metric_names[args.metric1]} {value_ranges[args.metric1]} {arrows[args.metric1]}"
    )
    ax.set_ylabel(
        f"{metric_names[args.metric2]} {value_ranges[args.metric2]} {arrows[args.metric2]}"
    )
    ax.grid(which='major', color='gray')

    # fig.suptitle(f"{metric_names[args.metric1]} vs {metric_names[args.metric2]}")
    fig.tight_layout()
    fig.savefig(
        f"aggregated_{args.metric1.replace('-','_')}_vs_{args.metric2.replace('-','_')}.pdf"
    )


if __name__ == "__main__":
    main()
