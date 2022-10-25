from typing import List, Dict, Iterable, Sized, Union

import wandb

import argparse

import numpy as np

def stderr(x: Union[np.array, List[float]]):
    return np.std(x)/np.sqrt(len(x))

def minmax(x: Union[np.array, List[float]]):
    mean = np.mean(x)
    return mean - np.min(x), np.max(x) - mean

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        help="Run groups, separated by colons. Runs within groups may be separated using spaces or commas.",
    )

    parser.add_argument(
        "--group-aliases",
        type=str,
        nargs="+",
        default=None,
        help="space,comma or colon separated names of groups.",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="List of W&B summary metrics",
    )

    parser.add_argument(
        "--runs-per-group",
        type=int,
        required=False,
        default=-1,
        help="Limit number of runs per group",
    )

    parser.add_argument(
        "--error-type",
        choices=["stddev", "var", "minmax", "stderr"],
        default="stderr",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="liebeskind",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="thesis",
    )

    parser.add_argument(
        "--filter-params",
        type=str,
        nargs="*",
        default=None
    )

    args = parser.parse_args()

    run_groups: List[List[str]] = list(
        map(
            lambda x: list(filter(lambda y: len(y) > 0, x.split(","))),
            (",".join(args.groups)).split(":"),
        )
    )

    if args.filter_params is not None:
        filter_params = list(map(lambda x: x.split("="), args.filter_params))
    else:
        filter_params = list()

    group_aliases: List[str] = list(
        filter(
            lambda x: len(x) > 0,
            (
                ",".join(args.group_aliases)
                .replace(":", ",")
                .replace(";", ",")
                .replace(" ", ",")
            ).split(","),
        )
    )

    assert len(group_aliases) == len(run_groups)

    metrics_results: Dict[str, Dict[str, List[float]]] = dict()

    api = wandb.Api()

    metric_name_cell_width = len(" Metric Name ")

    for metric_name in args.metrics:
        metrics_results[metric_name] = dict()
        metric_name_cell_width = max(metric_name_cell_width, len(metric_name) + 2)


    cell_width = 6 + 1 + 6 + 2 + 2

    for group_alias in group_aliases:
        cell_width = max(cell_width, len(group_alias) + 2)

        for metric_name in args.metrics:
            metrics_results[metric_name][group_alias] = list()

    for group_alias, run_group_names in zip(group_aliases, run_groups):
        for run_group in run_group_names:
            runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}", filters={"group": run_group})
            for run in runs:
                if "hidden" in run.tags or run.state != "finished":
                    continue

                filtered = False
                for param_key, param_value in filter_params:
                    if param_key not in run.config or run.config[param_key] is None:
                        continue
                    else:
                        param_true_value = run.config[param_key]
                        if isinstance(param_true_value, str) and param_true_value == param_value:
                            continue
                        elif isinstance(param_true_value, int) and param_true_value == int(param_value):
                            continue
                        elif isinstance(param_true_value, float) and param_true_value == float(param_value):
                            continue

                        filtered = True
                        break

                if filtered:
                    continue


                for metric in args.metrics:
                    if metric == "coherence":
                        if "Generation/loo" in run.summary and "all_attributes" in run.summary["Generation/loo"]:
                            metrics_results[metric][group_alias].append(float(run.summary["Generation/loo"]["all_attributes"]))
                    elif metric == "disentanglement":
                        if "disentanglement/higgins/test/individuals/joint" in run.summary and "disentanglement" in run.summary["disentanglement/higgins/test/individuals/joint"]:
                            metrics_results[metric][group_alias].append(float(run.summary["disentanglement/higgins/test/individuals/joint"]["disentanglement"]))

                    elif metric == "total-correlation":
                        if "total_correlation/total_correlation/train/" in run.summary:
                            metric_value = 0
                            metric_name_len = 0
                            for subset_name, value in run.summary["total_correlation/total_correlation/train/"].items():
                                if len(subset_name) > metric_name_len:
                                    metric_name_len = len(subset_name)
                                    metric_value = value
                            metrics_results[metric][group_alias].append(float(metric_value))

                    else:
                        if metric not in run.summary:
                            continue

                        if group_alias not in metrics_results[metric]:
                            metrics_results[metric][group_alias] = list()

                        metrics_results[metric][group_alias].append(float(run.summary[metric]))

    if args.runs_per_group > 1:
        for metric in args.metrics:
            for group_alias in group_aliases:
                if len(metrics_results[metric][group_alias]) > args.runs_per_group:
                    metrics_results[metric][group_alias] = np.random.choice(metrics_results[metric][group_alias], args.runs_per_group)

    errs: Dict[str, callable] = {
        "stderr": stderr,
        "stddev": np.std,
        "var": np.var,
        "minmax": minmax
    }

    metrics_results_strings: Dict[str, Dict[str]] = dict()
    metrics_results_tex: Dict[str, Dict[str]] = dict()

    for metric, groups in metrics_results.items():
        if metric not in metrics_results_strings:
            metrics_results_strings[metric] = dict()
            metrics_results_tex[metric] = dict()
        for group, values in groups.items():
            result_string = f"{np.mean(values):.2f} ("
            result_tex = "\\(" + result_string[:-1]
            error = errs[args.error_type](values)
            if args.error_type == "minmax":
                result_string += f"+{error[0]:.2f}, -{error[1]:.2f}"
                result_tex += f"+{error[0]:.2f}, -{error[1]:.2f}"
            else:
                result_string += f"±{error:.2f}"
                result_tex += f"\\pm {error:.2f}"
            result_string += ")"
            result_tex += "\\)"
            metrics_results_strings[metric][group] = result_string
            metrics_results_tex[metric][group] = result_tex
            cell_width = max(cell_width, len(result_string))



    top_rule    = "╔" + metric_name_cell_width*"═" + "╦═" + "═╦═".join(len(group_aliases) * [ cell_width * "═" ]) + "═╗"
    header_rule = "╠" + metric_name_cell_width*"═" + "╬═" + "═┼═".join(len(group_aliases) * [ cell_width * "═" ]) + "═╣"
    inner_rule  = "╠" + metric_name_cell_width*"─" + "╬─" + "─┼─".join(len(group_aliases) * [ cell_width * "─" ]) + "─╣"
    bot_rule    = "╚" + metric_name_cell_width*"═" + "╩═" + "═╩═".join(len(group_aliases) * [ cell_width * "═" ]) + "═╝"
    print(top_rule)
    print("║ " + "Metric Name".ljust(metric_name_cell_width - 1, " ") + "║ " + " │ ".join(map(lambda x: x.ljust(cell_width, " "), group_aliases)) + " ║")
    print(header_rule)
    for metric in args.metrics:
        print("║ " + metric.ljust(metric_name_cell_width - 1, " ") + "║ " + " │ ".join(map(lambda x: x.rjust(cell_width, " "), [metrics_results_strings[metric][group_alias] for group_alias in group_aliases])) + " ║")

        if metric != args.metrics[-1]:
            print(inner_rule)

    print(bot_rule)

    print("\nTeX:\n")

    print(" & ".join(["Metric Name"] + group_aliases) + r"\\\toprule")
    for metric in args.metrics:
        print(metric + " & " + " & ".join([metrics_results_tex[metric][group_alias] for group_alias in group_aliases]) + r"\\")



if __name__ == "__main__":
    main()
