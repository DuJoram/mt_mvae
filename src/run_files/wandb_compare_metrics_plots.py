#%% imports
import argparse
from typing import Dict, List, Union, Optional

import matplotlib as mpl
import numpy as np
import wandb
from matplotlib import pyplot as plt
import matplotlib.ticker
from wandb.wandb_run import Run

# mpl.use("pgf")
mpl.rcParams.update({
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
    "axes.xmargin": 0,
    "figure.figsize": (3, 1.75),
    "axes.linewidth": 0.3,
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.75,
    "xtick.major.width": 0.4,
    "xtick.minor.width": 0.2,
    "ytick.major.width": 0.4,
    "ytick.minor.width": 0.2,
    "lines.markersize": 2.5,
})

LEGEND_BBOX_TO_ANCHOR = (1.04, 0.5)
LEGEND_LOC = "center left"
MARKERS = ['o', 'd', 'X']

def minmax(X: np.array, axis: Optional[int] = None, keepdims: bool = True) -> np.array:
    means = np.mean(X, axis=axis)
    mins = np.min(X, axis=axis) - means
    maxs = np.max(X, axis=axis) - means
    ret = np.abs(np.array([mins, maxs]).T[:, :, None])
    return ret


def stderr(X: np.array, axis: Optional[int] = None, keepdims: bool = False) -> np.array:
    n = X.shape[axis] if axis is not None else np.prod(np.shape(X))
    return np.std(X, axis=axis, keepdims=keepdims)[:, None] / np.sqrt(n)


def numpy_deviation_wrapper(fn):
    def fn_wrapper(X: np.array, axis: Optional[int]) -> np.array:
        return fn(X, axis=axis)[:, None]

    return fn_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group", type=str, help="W&B run name")

    parser.add_argument(
        "--disentanglement-subset-name", type=str, help="Name of subset to use"
    )

    parser.add_argument(
        "--coherence-attribute",
        type=str,
        default="all_attributes",
        help='Name of attribute to filter for. (Use "all_attributes" to use the joint coherence)',
    )

    parser.add_argument(
        "--sweep-parameter",
        type=str,
        required=True,
        default="",
    )

    parser.add_argument(
        "--sweep-parameter-alias",
        type=str,
        default=None,
        help="Alias for use in plots (may use TeX formatting)",
    )

    parser.add_argument(
        "--grouping-parameter",
        type=str,
        default=None,
        help="Parameters to group by",
    )

    parser.add_argument(
        "--deviation-type",
        type=str,
        choices=["stddev", "var", "minmax", "stderr"],
        default="stderr",
    )

    parser.add_argument(
        "--plot-runs",
        action="store_true",
        default=False,
        help="Plots runs individually",
    )

    parser.add_argument(
        "--metric1",
        type=str,
        choices=["coherence", "disentanglement", "total-correlation"],
        default="coherence",
    )

    parser.add_argument(
        "--metric2",
        type=str,
        choices=["coherence", "disentanglement", "total-correlation"],
        default=None,
    )

    parser.add_argument(
        "--log-base",
        type=float,
        default=2.0,
        help="Log base of x-axis",
    )

    parser.add_argument(
        "--subs",
        type=int,
        default=8,
        help="Subdivisions per major tick on x-axis"
    )

    parser.add_argument(
        "--x-scale",
        type=str,
        default="sym-log",
        choices=["sym-log", "log"],
    )

    parser.add_argument(
        "--total-correlation-subset", type=str, default="full", help="Subset key to use"
    )

    args = parser.parse_args()
    grouping_parameter = args.grouping_parameter
    has_grouping_parameter = grouping_parameter is not None
    sweep_param = args.sweep_parameter
    sweep_param_alias = (
        args.sweep_parameter_alias
        if args.sweep_parameter_alias is not None
        else sweep_param
    )

    print(f"sweep_param_alias: {sweep_param_alias}")

    if args.deviation_type == "stddev":
        bars_fn = numpy_deviation_wrapper(np.std)
    elif args.deviation_type == "var":
        bars_fn = numpy_deviation_wrapper(np.var)
    elif args.deviation_type == "minmax":
        bars_fn = minmax
    elif args.deviation_type == "stderr":
        bars_fn = stderr
    else:
        raise ValueError("Invalid choice for deviation type!")

    api = wandb.Api()
    runs = api.runs("liebeskind/thesis", filters={"group": args.run_group})

    train_runs: Dict[str, Run] = dict()
    m1_runs: Dict[str, Run] = dict()
    m2_runs: Dict[str, Run] = dict()

    disent_baseline = None
    coher_baseline = None

    for run in runs:
        if "hidden" in run.tags:
            continue
        if run.state == "finished":
            pass

        if coher_baseline is None:
            if run.config['dataset'] == 'MdSprites':
                coher_baseline = 0.1771
            else:
                coher_baseline = 0.1

        if disent_baseline is None:
            if (
                "shared_attributes" in run.config and run.config["shared_attributes"] is not None
                and len(run.config["shared_attributes"]) > 0
            ):
                disent_baseline = 1 / len(run.config["shared_attributes"])

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
        elif args.metric2 is not None and run_metric == args.metric2:
            m2_runs[run.name] = run

    values: Dict[str, Union[Dict[str, List[List[float]]], List[List[float]]]] = dict()

    sweep_values: List[float] = list()
    if has_grouping_parameter:
        grouping_values: List[float] = list()

    subset_key = (
        None
        if args.total_correlation_subset == "full"
        else args.total_correlation_subset
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

    for run_name, m1_run in m1_runs.items():
        if args.metric2 is not None:
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

        if args.metric2 is not None:
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
        else:
            m2 = 0

        sweep_value = float(m1_run.config[sweep_param])
        sweep_value_key = str(sweep_value)

        if sweep_value not in sweep_values:
            sweep_values.append(sweep_value)

        if has_grouping_parameter:
            grouping_value = float(m1_run.config[grouping_parameter])
            grouping_value_key = str(grouping_value)

            if sweep_value_key not in values:
                values[sweep_value_key] = dict()

            if grouping_value_key not in values[sweep_value_key]:
                values[sweep_value_key][grouping_value_key] = list()

            if grouping_value not in grouping_values:
                grouping_values.append(grouping_value)

            if "joint" not in values[sweep_value_key]:
                values[sweep_value_key]["joint"] = list()

            values[sweep_value_key]["joint"].append([m1, m2])
            values[sweep_value_key][grouping_value_key].append([m1, m2])

        else:
            if sweep_value_key not in values:
                values[sweep_value_key] = list()

            values[sweep_value_key].append([m1, m2])

    sweep_values = sorted(sweep_values)

    if has_grouping_parameter:
        grouping_values = sorted(grouping_values)

        for grouping_value in grouping_values:
            grouping_key = str(grouping_value)

            for sweep_value in sweep_values:
                sweep_key = str(sweep_value)
            pass

    else:
        means: List[np.array] = list()
        bars: List[np.array] = list()
        points: List[np.array] = list()
        for sweep_value in sweep_values:
            sweep_key = str(sweep_value)
            point_values = np.array(values[sweep_key])
            points.append(point_values)
            mean = np.mean(point_values, axis=0)
            means.append(mean)
            bar = bars_fn(point_values, axis=0)
            bars.append(bar)

        means_x = np.array(means)[:, 0]
        bars_x = np.array(bars)[:, 0]

        axis_maxes = np.max(np.concatenate(points), axis=0)
        axis_mins = np.min(np.concatenate(points), axis=0)
        x_range = axis_maxes[0] - axis_mins[0]
        x_lim_low = max(axis_mins[0] - x_range * 0.05, 0)
        x_lim_high = axis_maxes[0] + x_range * 0.05
        x_range = x_lim_high - x_lim_low

        cmap_sweep_values = mpl.cm.get_cmap("cividis", len(points) + 1)

        if args.metric2 is not None:
            means_y = np.array(means)[:, 1]
            bars_y = np.array(bars)[:, 1]

            fig_scatter = plt.figure()
            ax_scatter = fig_scatter.subplots()

            bars = list()
            for idx in range(len(means_x)):
                if args.plot_runs:
                    ax_scatter.scatter(
                        [points[idx][:, 0]],
                        [points[idx][:, 1]],
                        color=cmap_sweep_values(idx),
                        alpha=0.5,
                    )
                bars.append(ax_scatter.errorbar(
                    [means_x[idx]],
                    [means_y[idx]],
                    xerr=bars_x[idx],
                    yerr=bars_y[idx],
                    color=cmap_sweep_values(idx),
                    fmt="o",
                    linewidth=1,
                    capsize=4,
                    label=f"{sweep_param_alias} = {sweep_values[idx]}",
                ))

            ax_scatter.set_xlabel(
                f"{metric_names[args.metric1]} {value_ranges[args.metric1]} {arrows[args.metric1]}"
            )
            ax_scatter.set_ylabel(
                f"{metric_names[args.metric2]} {value_ranges[args.metric2]} {arrows[args.metric2]}"
            )


            y_range = axis_maxes[1] - axis_mins[1]

            y_lim_low = max(axis_mins[1] - y_range * 0.05, 0)
            y_lim_high = axis_maxes[1] + y_range * 0.05

            y_range = y_lim_high - y_lim_low

            ax_scatter.set_xlim(x_lim_low, x_lim_high)
            ax_scatter.set_ylim(y_lim_low, y_lim_high)

            fig_scatter.tight_layout()
            fig_scatter.savefig(
                f"{run.group}_{args.metric1.replace('-','_')}_vs_{args.metric2.replace('-','_')}_{args.deviation_type}.pdf"
            )

            fig_legend = plt.figure()
            legend_labels = [bar.get_label() for bar in bars]
            fig_legend.legend(bars, legend_labels, loc="center", ncol=len(sweep_values))
            fig_legend.tight_layout()
            fig_legend.savefig(
                f"{run.group}_{args.metric1.replace('-','_')}_vs_{args.metric2.replace('-','_')}_{args.deviation_type}-legend.pdf",
                bbox_inches='tight',
                pad_inches=0.1,
            )

            plt.close(fig_scatter)
            plt.close(fig_legend)

        #%%
        fig_lines = plt.figure()
        ax_lines1 = fig_lines.subplots()

        # ax_lines1.set_xlim(sweep_values[0], sweep_values[-1])

        if args.x_scale == 'sym-log':
            ax_lines1.set_xscale(mpl.scale.SymmetricalLogScale(ax_lines1.get_xaxis(), linthresh=1, linscale=0.5, base=args.log_base))
        elif args.x_scale == 'log':
            # ax_lines1.set_xscale('log', base=args.log_base)  # mpl.scale.LogScale(ax_lines1.get_xaxis(), subs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], base=args.log_base))
            # ax_lines1.set_xscale('log', base=args.log_base)
            # ax_lines1.get_xaxis().set_minor_locator(mpl.ticker.LogLocator(base=2, numticks=100000))
            # ax_lines1.get_xaxis().set_major_locator(mpl.ticker.LogLocator(base=2, numdecs=4))
            # ax_lines1.get_xaxis().set_minor_locator(mpl.ticker.MultipleLocator(5))
            # ax_lines1.set_xscale(mpl.scale.LogScale(ax_lines1.get_xaxis(), subs=np.linspace(0,1,args.subs)'all', base=2))
            ax_lines1.set_xscale('log', base=args.log_base)
            # ax_lines1.get_xaxis().set_minor_locator(mpl.ticker.LogLocator(base=args.log_base, subs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], numticks=1000))
            # ax_lines1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
            ax_lines1.get_xaxis().set_major_locator(mpl.ticker.LogLocator(base=args.log_base, subs=[1]))
        # ax_lines1.set_xscale(mpl.scale.SymmetricalLogScale(ax_lines1.get_xaxis(), subs=np.arange(sweep_values[0], sweep_values[-1]), linscale=0.5, base=args.log_base))
        # ax_lines1.grid(axis='x', which='minor')  # , color='gray', linewidth=0.25, markevery=1)  # , linewidth=0.25, markevery=0.01)
        ax_lines1.grid(axis='x', which='major', color='black')
        ax_lines1.grid(axis='y', which='major')


        points = np.array(points)
        runs_per_sweep_value = points.shape[1]
        cmap_runs = mpl.cm.get_cmap("cividis", 16)
        c1_idx = 2
        c2_idx = 13

        # ax_lines1.grid(axis='y', color=cmap_runs(c1_idx))
        # ax_lines2.grid(axis='y', color=cmap_runs(c2_idx))

        if args.metric2 is not None:
            ax_lines2 = ax_lines1.twinx()

        for run_idx in range(runs_per_sweep_value):
            if args.plot_runs:
                ax_lines1.scatter(sweep_values, points[:, run_idx, 0], color=cmap_runs(c1_idx), alpha=0.3)

                if args.metric2 is not None:
                    ax_lines2.scatter(sweep_values, points[:, run_idx, 1], color=cmap_runs(c2_idx), alpha=0.3)

        line1 = ax_lines1.plot(sweep_values, means_x, color=cmap_runs(c1_idx), label=f"{metric_names[args.metric1]}", marker=MARKERS[0])
        ax_lines1.fill_between(sweep_values, means_x - bars_x.flatten(), means_x + bars_x.flatten(), color=cmap_runs(c1_idx), alpha=0.4)

        baseline_line = None
        if args.metric1 == 'coherence':
            baseline_line = ax_lines1.plot([sweep_values[0],sweep_values[-1]], [coher_baseline, coher_baseline], linestyle='dashed', color='slategray', label=f"Coherence Baseline")


        if args.metric2 is not None:
            line2 = ax_lines2.plot(sweep_values, means_y, color=cmap_runs(c2_idx), label=f"{metric_names[args.metric2]}", marker=MARKERS[1])
            ax_lines2.fill_between(sweep_values, means_y - bars_y.flatten(), means_y + bars_y.flatten(), color=cmap_runs(c2_idx), alpha=0.4)

            if args.metric2 == 'coherence':
                baseline_line = ax_lines2.plot([sweep_values[0],sweep_values[-1]], [coher_baseline, coher_baseline], linestyle='dashed', color='slategray', label=f"Coherence Baseline")

        ax_lines1.set_xlabel(f"{sweep_param_alias}")
        ax_lines1.set_ylabel(
            f"{metric_names[args.metric1]} {value_ranges[args.metric1]} {arrows[args.metric1]}"
        )

        if args.metric2 is not None:
            legend_labels = [line1[0].get_label(), line2[0].get_label()]
            lines = line1 + line2

            if baseline_line is not None:
                legend_labels = [ baseline_line[0].get_label() ] + legend_labels
                lines = baseline_line + lines

            # fig_lines.legend(lines, legend_labels, loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR)


            ax_lines2.set_ylabel(
                f"{metric_names[args.metric2]} {value_ranges[args.metric2]} {arrows[args.metric2]}"
            )

            fig_lines.tight_layout()
            fig_lines.savefig(
                f"{run.group}_lines_{args.metric1.replace('-','_')}_and_{args.metric2.replace('-','_')}_vs_{args.sweep_parameter}_{args.deviation_type}.pdf"
            )

            fig_legend = plt.figure()
            fig_legend.legend(lines, legend_labels, loc="center", ncol=len(sweep_values))
            fig_legend.tight_layout()
            fig_legend.savefig(
                f"{run.group}_lines_{args.metric1.replace('-','_')}_and_{args.metric2.replace('-','_')}_vs_{args.sweep_parameter}_{args.deviation_type}-legend.pdf",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close(fig_legend)

        else:
            fig_lines.tight_layout()
            fig_lines.savefig(
                f"{run.group}_lines_{args.metric1.replace('-','_')}_vs_{args.sweep_parameter}_{args.deviation_type}.pdf"
            )
        plt.close(fig_lines)

        #%%


        fig = plt.figure()
        ax = fig.subplots()
        sweep_values_labels = list(map(lambda x: f"{x:5.3f}", sweep_values))
        width = 0.25
        x_pos = np.arange(len(sweep_values))
        if args.metric2 is not None:
            x_pos_m1 = x_pos - width / 1.8
            x_pos_m2 = x_pos + width / 1.8
        else:
            x_pos_m1 = x_pos

        bar_m1 = ax.bar(x_pos_m1, means_x, width=width, color="tab:blue")
        if bars_x.shape[1] == 1:
            bars_x = bars_x.flatten()
            if args.metric2 is not None:
                bars_y = bars_y.flatten()
        else:
            bars_x = bars_x.T[0]
            if args.metric2 is not None:
                bars_y = bars_y.T[0]

        ax.set_xlim(-0.75, len(sweep_values) - 0.25)
        color = "tab:blue"
        ax.errorbar(
            x_pos_m1, means_x, yerr=bars_x, fmt="none", ecolor="k", alpha=0.5
        )
        ax.set_xticks(x_pos, labels=sweep_values_labels)
        ax.set_ylabel(
            f"{metric_names[args.metric1]} {value_ranges[args.metric1]} {arrows[args.metric1]}",
            color=color,
        )
        if args.metric1 == "disentanglement" and disent_baseline is not None:
            ax.plot(
                [-1, len(sweep_values)],
                [disent_baseline, disent_baseline],
                color=color,
                label="baseline",
            )

        ax.set_xlabel(sweep_param_alias)
        ax.tick_params(axis="y", color=color)
        ax.set_ylim(x_lim_low, x_lim_high)
        ax.bar_label(bar_m1, padding=3)

        color = "tab:orange"
        if args.metric2 is not None:
            ax2 = ax.twinx()
            ax2.set_xlim(-0.75, len(sweep_values) - 0.25)
            ax2.set_ylim(y_lim_low, y_lim_high)
            bar_m2 = ax2.bar(x_pos_m2, means_y, width=width, color=color)
            ax2.errorbar(
                x_pos_m2, means_y, yerr=bars_y, fmt="none", ecolor="k", alpha=0.5
            )
            ax2.bar_label(bar_m2, padding=3)
            ax2.set_ylim(y_lim_low, y_lim_high)
            ax2.set_ylabel(
                f"{metric_names[args.metric2]} {value_ranges[args.metric2]} {arrows[args.metric2]}",
                color=color,
            )
            ax2.tick_params(axis="y", color=color)

            if args.metric2 == "disentanglement" and disent_baseline is not None:
                ax2.plot(
                    [-1, len(sweep_values)],
                    [disent_baseline, disent_baseline],
                    color=color,
                    label="baseline",
                )

            fig.suptitle(
                f"{metric_names[args.metric1]} and {metric_names[args.metric2]}, {sweep_param_alias} sweep",
                fontsize=16,
            )
            ax.set_title(f"Run Group: {run.group}, bars: {args.deviation_type}")
            fig.tight_layout()
            fig.savefig(
                f"{run.group}_{args.metric1.replace('-','_')}_{args.metric2.replace('-','_')}_vs_{sweep_param}_{args.deviation_type}.pdf"
            )
        else:
            fig.suptitle(
                f"{metric_names[args.metric1]} vs {sweep_param_alias} sweep"
            )
            fig.tight_layout()
            fig.savefig(
                f"{run.group}_{args.metric1.replace('-','_')}_vs_{sweep_param}_{args.deviation_type}.pdf"
            )
        # fig.savefig(
        #     f"{run.group}_{args.metric1.replace('-','_')}_{args.metric2.replace('-','_')}_vs_sweep_param_{args.deviation_type}.pgf"
        # )



if __name__ == "__main__":
    main()
