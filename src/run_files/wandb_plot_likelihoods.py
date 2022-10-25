import argparse
import math
import typing

from typing import Dict, List

import numpy as np
import pandas as pd
import wandb

import itertools
import numpy

import matplotlib as mpl
from matplotlib import pyplot as plt

LEGEND_BBOX_TO_ANCHOR = (1.04, 0.5)
LEGEND_LOC = "center left"
MARKERS = ['o', 'X', 'D']

# mpl.use("pgf")
mpl.rcParams.update({
    "text.latex.preamble": '\n'.join([
        r'\usepackage{mathpazo}',
        r'\usepackage[OT1]{fontenc}',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage{amsmath,amssymb,amsfonts,mathrsfs,bm}'
    ]),
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.serif": "Palatino",
    "font.sans-serif": "sans-serif",
    "figure.dpi": 300,
#    "font.size": 11,
#    "axes.titlesize": 14,
#    "axes.labelsize": 10,
#    "xtick.labelsize": 8,
#    "ytick.labelsize": 8,
#    "legend.fontsize": 10,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.pad_inches": 1,
    "figure.figsize": (3, 1.75),
    "axes.xmargin": 0,
    "lines.markersize": 2.5,
})

plt.rcParams.update({
    "text.latex.preamble": '\n'.join([
        r'\usepackage{mathpazo}',
        r'\usepackage[OT1]{fontenc}',
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage{amsmath,amssymb,amsfonts,mathrsfs,bm}'
    ]),
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "text.usetex": True,
    "font.size": 11,
    "font.serif": "Palatino",
    "font.sans-serif": "sans-serif",
    "figure.dpi": 300,
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
    "figure.figsize": (3, 1.75),
    "axes.xmargin": 0,
    "axes.linewidth": 0.3,
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.75,
    "xtick.major.width": 0.4,
    "xtick.minor.width": 0.2,
    "ytick.major.width": 0.4,
    "ytick.minor.width": 0.2,
    "lines.markersize": 5
})
# rcp = {
#     "text.latex.preamble": '\n'.join([
#         r'\usepackage[sc]{mathpazo}',
#         r'\usepackage[OT1]{fontenc}',
#         r'\usepackage[utf8]{inputenc}',
#         r'\usepackage{amsmath,amssymb,amsfonts,mathrsfs,bm}'
#     ]),
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "text.usetex": True,
#     "font.size": 11,
#     "font.serif": "Palatino",
#     "font.sans-serif": "sans-serif",
#     "figure.dpi": 300,
#     "axes.titlesize": 14,
#     "axes.labelsize": 10,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 10,
#     "savefig.dpi": 300,
#     "savefig.pad_inches": 0.1,
# }
# mpl.rcParams.update(
#     rcp
# )
#
# plt.rcParams.update(
#     rcp
# )

def get_cond_indices(num_mods: int, cond_subset_sizes: typing.Union[int, typing.Iterable] = None):
    if cond_subset_sizes is None:
        cond_subset_sizes = range(1, num_mods)
    if isinstance(cond_subset_sizes, int):
        cond_subset_sizes = [cond_subset_sizes]

    subsets = list(map(
        list,
        itertools.chain.from_iterable((
            itertools.combinations(range(num_mods), r=r)
            for r in cond_subset_sizes
        ))
    ))

    cond_keys = list()
    for subset in subsets:
        for idx in range(num_mods):
            if idx not in subset:
                cond_keys.append((subset, idx))

    return cond_keys

def get_cond_keys(num_mods: int, cond_subset_sizes: typing.Union[int, typing.Iterable] = None):
    return list(map(lambda x: "_".join(map(lambda y: f"m{y}", x[0])) + ".m" + str(x[1]), get_cond_indices(num_mods=num_mods, cond_subset_sizes=cond_subset_sizes)))

def get_rec_indices(num_mods: int, cond_subset_sizes: typing.Union[int, typing.Iterable] = None):
    if cond_subset_sizes is None:
        cond_subset_sizes = range(num_mods + 1)
    if isinstance(cond_subset_sizes, int):
        cond_subset_sizes = [cond_subset_sizes]

    subsets = list(map(
        list,
        itertools.chain.from_iterable((
            itertools.combinations(range(num_mods), r=r)
            for r in cond_subset_sizes
        ))
    ))

    cond_keys = list()
    for subset in subsets:
        for idx in range(num_mods):
            if idx in subset:
                cond_keys.append((subset, idx))

    return cond_keys

def get_rec_keys(num_mods: int, cond_subset_sizes: typing.Union[int, typing.Iterable] = None):
    return list(map(lambda x: "_".join(map(lambda y: f"m{y}", x[0])) + ".m" + str(x[1]), get_rec_indices(num_mods=num_mods, cond_subset_sizes=cond_subset_sizes)))

def get_joint_indices(num_mods: int, subset_sizes: typing.Union[int, typing.Iterable] = None):
    if subset_sizes is None:
        subset_sizes = range(num_mods + 1)
    if isinstance(subset_sizes, int):
        subset_sizes = [subset_sizes]

    joint_indices = list(filter(
        lambda x: len(x) > 0,
        itertools.chain.from_iterable(
            itertools.combinations(range(num_mods), r=r)
            for r in subset_sizes
        )
    ))

    return joint_indices

def get_joint_keys(num_mods: int, subset_sizes: typing.Union[int, typing.Iterable] = None):
    return list(map(lambda x: "_".join(map(lambda y: f"m{y}", x)) + ".joint", get_joint_indices(num_mods=num_mods, subset_sizes=subset_sizes)))


def get_cond_tex_str(num_mods: int, cond_subset_sizes: typing.Union[int, typing.Iterable] = None):
    return list(map(
        lambda x: rf"\mathbf{{X}}^{{({x[1]})}}\mid " + ", ".join(map(lambda y: rf"\mathbf{{X}}^{{({y})}}", x[0])),
        get_cond_indices(num_mods=num_mods, cond_subset_sizes=cond_subset_sizes)
    ))

def get_rec_tex_str(num_mods: int, cond_subset_sizes: typing.Union[int, typing.Iterable] = None):
    return list(map(
        lambda x: rf"\mathbf{{X}}^{{({x[1]})}}\mid " + ", ".join(map(lambda y: rf"\mathbf{{X}}^{{({y})}}", x[0])),
        get_rec_indices(num_mods=num_mods, cond_subset_sizes=cond_subset_sizes)
    ))

def get_joint_tex_str(num_mods: int, subset_sizes: typing.Union[int, typing.Iterable] = None):
    return list(map(
        lambda x: ", ".join(map(lambda y: rf"\mathbf{{X}}^{{({y})}}", x)),
        get_joint_indices(num_mods=num_mods, subset_sizes=subset_sizes)
    ))



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-group",
        type=str,
        required=True,
        help="Name of run group"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="thesis",
        help="W&B project name"
    )

    parser.add_argument(
        "--grouping-param",
        type=str,
        required=True,
        help="Parameter to group runs by"
    )

    parser.add_argument(
        "--grouping-param-alias",
        type=str,
        default=None,
        required=False,
        help="Parameter string for plot rendering"
    )

    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"liebeskind/{args.project}", filters={"group": args.run_group})

    grouping_values = list()
    step_values: Dict[Dict[List[float]]] = dict()
    likelihoods: Dict[Dict[List[List[float]]]] = dict()

    keys_cond = None
    keys_rec = None
    keys_joint = None

    dfs: Dict[float, List[pd.DataFrame]] = dict()
    dfs_joint: Dict[float, pd.DataFrame] = dict()

    num_mods = None
    for run in runs:
        if "hidden" in run.tags or run.state != "finished":
            continue

        if num_mods is None:
            num_mods = run.config["num_mods"]
            keys_cond = get_cond_keys(num_mods)
            keys_rec = get_rec_keys(num_mods)
            keys_joint = get_joint_keys(num_mods)

        grouping_value = float(run.config[args.grouping_param])
        if grouping_value not in dfs:
            dfs[grouping_value] = list()

        if grouping_value not in grouping_values:
            grouping_values.append(grouping_value)

        run_history = run.history(
            keys=list(map(lambda key: "Likelihoods/" + key, keys_cond + keys_rec + keys_joint))
        )
        if run_history.size > 0:
            dfs[grouping_value].append(run_history.set_index('_step'))

    for grouping_value, dfs_group in dfs.items():
        dfs_cat = pd.concat(dfs_group)
        dfs_grouped = dfs_cat.groupby(dfs_cat.index)
        means = dfs_grouped.mean().add_suffix("_mean")
        vars = dfs_grouped.var().add_suffix("_var")
        stds = dfs_grouped.std().add_suffix("_std")
        stderrs = (dfs_grouped.std()/math.sqrt(len(dfs_group))).add_suffix("_stderr")
        joint = means.join(vars).join(stds).join(stderrs)
        dfs_joint[grouping_value] = joint.reindex(sorted(joint.columns), axis=1)




    grouping_values = sorted(grouping_values)
    # for grouping_value in grouping_values:
    #     dfs_joint[grouping_value].to_csv(f"likelihoods_{args.run_group}_{args.grouping_param}_{grouping_value}.csv")


    cmap = mpl.cm.get_cmap("cividis", len(grouping_values))

    grouping_param_name = args.grouping_param_alias if args.grouping_param_alias is not None else args.grouping_param

    cond_keys = get_cond_keys(num_mods=num_mods, cond_subset_sizes=range(1, num_mods))
    cond_indices = get_cond_indices(num_mods=num_mods, cond_subset_sizes=range(1, num_mods))

    for modality_idx in range(num_mods):
        fig = plt.figure()
        ax = fig.subplots()
        lines = list()
        for line_idx, grouping_value in enumerate(grouping_values):
            iters = None

            values = list()

            for cond_key, (_, gen_idx) in zip(cond_keys, cond_indices):
                if gen_idx != modality_idx:
                    continue

                for df in dfs[grouping_value]:
                    if iters is None:
                        iters = df["Likelihoods/" + cond_key].index

                    values.append(df["Likelihoods/" + cond_key].values)
            values = np.stack(values)

            means = np.mean(values, axis=0)
            stderrs = np.std(values, axis=0)/np.sqrt(values.shape[0])

            lines += ax.plot(iters, means, color=cmap(line_idx), label=f"{grouping_param_name} = {grouping_value}")
            ax.fill_between(iters, means-stderrs, means+stderrs, color=cmap(line_idx), alpha=0.25)

        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(rf"$\log p(\mathbf X^{{({modality_idx+1})}}\mid \mathbf X^{{(\setminus {modality_idx+1})}})$")
        fig.tight_layout()
        fig.savefig(f"{args.run_group}-cond-gen-m{modality_idx}.pdf")

        fig_legend = plt.figure()
        legend_labels = [line.get_label() for line in lines]
        fig_legend.legend(lines, legend_labels, loc="center", ncol=len(grouping_values))
        fig_legend.tight_layout()
        fig_legend.savefig(f"{args.run_group}-cond-gen-m{modality_idx}-legend.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    rec_keys = get_rec_keys(num_mods=num_mods, cond_subset_sizes=range(1, num_mods))
    rec_indices = get_rec_indices(num_mods=num_mods, cond_subset_sizes=range(1, num_mods))

    for modality_idx in range(num_mods):
        fig = plt.figure()
        ax = fig.subplots()
        lines = list()
        for line_idx, grouping_value in enumerate(grouping_values):
            iters = None

            values = list()

            for rec_key, (_, gen_idx) in zip(rec_keys, rec_indices):
                if gen_idx != modality_idx:
                    continue

                for df in dfs[grouping_value]:
                    if iters is None:
                        iters = df["Likelihoods/" + rec_key].index

                    values.append(df["Likelihoods/" + rec_key].values)
            values = np.stack(values)

            means = np.mean(values, axis=0)
            stderrs = np.std(values, axis=0)/np.sqrt(values.shape[0])

            lines += ax.plot(iters, means, color=cmap(line_idx), label=f"{grouping_param_name} = {grouping_value}")
            ax.fill_between(iters, means-stderrs, means+stderrs, color=cmap(line_idx), alpha=0.25)

        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(rf"$\log p(\mathbf X^{{({modality_idx+1})}}\mid \mathbf X^{{(\mathbb M)}})$")
        fig.tight_layout()
        fig.savefig(f"{args.run_group}-rec-m{modality_idx}.pdf")

        fig_legend = plt.figure()
        legend_labels = [line.get_label() for line in lines]
        fig_legend.legend(lines, legend_labels, loc="center", ncol=len(grouping_values))
        fig_legend.tight_layout()
        fig_legend.savefig(f"{args.run_group}-rec-m{modality_idx}-legend.pdf", bbox_inches='tight', pad_inches=0.1)

        plt.close(fig)
        plt.close(fig_legend)

    joint_keys = get_joint_keys(num_mods=num_mods, subset_sizes=[1, num_mods])
    joint_indices = get_joint_indices(num_mods=num_mods, subset_sizes=[1, num_mods])

    for joint_key, joint_index in zip(joint_keys, joint_indices):
        fig = plt.figure()
        ax = fig.subplots()
        lines = list()
        for line_idx, grouping_value in enumerate(grouping_values):
            iters = None
            values = list()
            for df in dfs[grouping_value]:
                if iters is None:
                    iters = df["Likelihoods/" + joint_key].index

                values.append(df["Likelihoods/" + joint_key].values)

            values = np.stack(values)

            means = np.mean(values, axis=0)
            stderrs = np.std(values, axis=0)/np.sqrt(values.shape[0])

            lines += ax.plot(iters, means, color=cmap(line_idx), label=f"{grouping_param_name} = {grouping_value}")
            ax.fill_between(iters, means-stderrs, means+stderrs, color=cmap(line_idx), alpha=0.25)

        ax.set_xlabel("Training Iteration")
        if len(joint_index) == 1:
            ax.set_ylabel(rf"$\log p(\mathbf X^{{({joint_index[0] + 1})}})$")
            # fig.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR)
            fig.tight_layout()
            fig.savefig(f"{args.run_group}-joint-m{joint_index[0]}.pdf")

            fig_legend = plt.figure()
            legend_labels = [line.get_label() for line in lines]
            fig_legend.legend(lines, legend_labels, loc="center", ncol=len(grouping_values))
            fig_legend.tight_layout()
            fig_legend.savefig(f"{args.run_group}-joint-m{joint_index[0]}-legend.pdf", bbox_inches='tight', pad_inches=0.1)
        else:
            ax.set_ylabel(rf"$\log p(\mathbf X^{{(\mathbb[M])}})$")
            # fig.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR)
            fig.tight_layout()
            fig.savefig(f"{args.run_group}-joint-m-all.pdf")

            fig_legend = plt.figure()
            legend_labels = [line.get_label() for line in lines]
            fig_legend.legend(lines, legend_labels, loc="center", ncol=len(grouping_values))
            fig_legend.tight_layout()
            fig_legend.savefig(f"{args.run_group}-joint-m-all-legend.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        plt.close(fig_legend)

    # for key in keys_cond:
    #     fig = plt.figure()
    #     ax = fig.subplots()
    #     key_mean = "Likelihoods/" + key + "_mean"
    #     key_stderr = "Likelihoods/" + key + "_stderr"
    #     for idx, grouping_value in enumerate(grouping_values):
    #         df = dfs_joint[grouping_value]
    #         iters = df.index
    #         means = df[key_mean].values
    #         stderrs = df[key_stderr].values
    #         ax.plot(iters, means, label=f"{grouping_param_name} = {grouping_value}", color=cmap(idx))
    #         ax.fill_between(iters, means-stderrs, means+stderrs, color=cmap(idx), alpha=0.2)

    #     ax.set_xlabel(f"Training Iteration")
    #     ax.set_ylabel(rf"Log-Likelihood $\log p_\theta({key})$")
    #     ax.legend()
    #     fig.savefig(f"{args.run_group}-cond-gen-{key}.pdf")
    #     plt.close(fig)



if __name__ == "__main__":
    main()