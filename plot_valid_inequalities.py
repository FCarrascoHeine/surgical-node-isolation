"""Plot disaggregated and VI LP improvements over the aggregated LP."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


def instance_family(name):
    match = re.search(r"(?:_|\d)(dir|c)$", Path(name).stem)
    return match.group(1) if match else "other"


def read_comparisons(filename):
    """Match optimal LP results by instance, repetition, seed, threads, and time limit."""
    groups = defaultdict(dict)
    with Path(filename).open(newline="") as file:
        for row in csv.DictReader(file):
            formulation = row.get("formulation")
            if row.get("mode") != "relaxation" or formulation not in ("3", "2", "3_VI"):
                continue
            key = tuple(row.get(field, "") for field in (
                "instance", "repetition", "solver_seed", "threads", "time_limit_seconds",
            ))
            if formulation in groups[key]:
                raise ValueError(f"Duplicate formulation {formulation} for {key}")
            groups[key][formulation] = row

    comparisons, skipped = [], []
    for key, models in groups.items():
        name, repetition, seed, threads, time_limit = key
        identity = dict(instance=name, family=instance_family(name), repetition=repetition,
                        solver_seed=seed, threads=threads, time_limit_seconds=time_limit)
        values = {}
        for formulation in ("3", "2", "3_VI"):
            row = models.get(formulation, {})
            if (row.get("status") != "OPTIMAL" or row.get("validation_passed") == "False"
                    or row.get("solution_type", "relaxation") != "relaxation"):
                continue
            try:
                value = float(row["objective_value"])
                if math.isfinite(value):
                    values[formulation] = value
            except (KeyError, ValueError):
                pass
        if len(values) != 3:
            missing = ", ".join(f for f in ("3", "2", "3_VI") if f not in values)
            skipped.append({**identity, "reason": f"Missing optimal LP results: {missing}"})
            continue
        baseline = values["3"]
        result = {**identity, "lp_3": baseline, "lp_2": values["2"], "lp_3_vi": values["3_VI"],
                  "cuts": models["3_VI"].get("cuts", ""),
                  "max_cuts": models["3_VI"].get("max_cuts", ""),
                  "separation_complete": models["3_VI"].get("separation_complete", "")}
        for formulation, label in (("2", "disaggregated"), ("3_VI", "vi")):
            delta = values[formulation] - baseline
            if abs(delta) <= 1e-9 * max(1.0, abs(baseline), abs(values[formulation])):
                delta = 0.0
            result[f"{label}_absolute"] = delta
            result[f"{label}_percent"] = 100 * delta / abs(baseline) if baseline else None
        gap = result["disaggregated_absolute"]
        result["gap_closed_percent"] = 100 * result["vi_absolute"] / gap if gap > 0 else None
        comparisons.append(result)
    comparisons.sort(key=lambda row: [int(s) if s.isdigit() else s
                                     for s in re.split(r"(\d+)", row["instance"] + row["repetition"])])
    return comparisons, skipped


def save_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparisons(rows, output, metric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    families = [family for family in ("c", "dir", "other")
                if any(row["family"] == family for row in rows)]
    count = max(sum(row["family"] == family for row in rows) for family in families)
    colors = {"disaggregated": "#516273", "vi": "#287A8E"}
    fig, axes = plt.subplots(1, len(families), squeeze=False, sharex=True,
                             figsize=(7.5 * len(families), max(5, 0.7 * count + 2.5)))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.20, wspace=0.75)
    values = [row[f"{model}_{metric}"] for row in rows for model in colors
              if row[f"{model}_{metric}"] is not None]
    scale = max([abs(value) for value in values] or [1.0]) or 1.0
    for ax, family in zip(axes[0], families):
        subset = [row for row in rows if row["family"] == family]
        for index, row in enumerate(subset):
            for offset, model in ((-0.17, "disaggregated"), (0.17, "vi")):
                position, value = index + offset, row[f"{model}_{metric}"]
                if value is None:
                    ax.text(0, position, "undefined: LP(3) = 0", va="center", fontsize=8)
                    continue
                hatch = "///" if model == "vi" and row["separation_complete"] == "False" else None
                if value:
                    ax.barh(position, value, height=0.28, color=colors[model],
                            hatch=hatch, edgecolor="white")
                else:
                    ax.scatter([0], [position], color=colors[model], s=22, zorder=3)
                label = f"{value:.3f}%" if metric == "percent" else f"{value:,.2f}"
                ax.annotate(label, (value, position), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=9, color=colors[model])
        names = [re.sub(r"_?(dir|c)$", "", Path(row["instance"]).stem)
                 + f" [r{row['repetition']}]" for row in subset]
        ax.set_yticks(range(len(subset)), names)
        ax.set_ylim(len(subset) - 0.4, -0.6)
        ax.set_xlim(min([0.0] + values) - 0.035 * scale, max([0.0] + values) + 0.30 * scale)
        ax.set_title(f"_{family} instances" if family != "other" else "Instances", loc="left")
        ax.set_xlabel("LP bound increase (%)" if metric == "percent" else "LP bound increase (original units)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E3E8EE")
        ax.axvline(0, color="#8997A8", linewidth=0.8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("LP improvement over the aggregated formulation", x=0.04, ha="left", fontsize=17)
    fig.legend(handles=[Patch(color=colors["disaggregated"], label="Disaggregated (2)"),
                        Patch(color=colors["vi"], label="Aggregated + VI (3_VI)")],
               loc="upper left", bbox_to_anchor=(0.035, 0.93), ncol=2, frameon=False)
    formula = "100 × (LP − LP(3)) / |LP(3)|" if metric == "percent" else "LP − LP(3)"
    caps = ", ".join(sorted({row["max_cuts"] for row in rows if row["max_cuts"]}))
    fig.text(0.04, 0.09, f"{formula}.  VI cut cap: {caps or 'unspecified'}.", fontsize=9)
    fig.text(0.04, 0.04, "Higher is better. Hatched bars: incomplete VI separation.", fontsize=9)
    stem = "lp_improvement" if metric == "percent" else "lp_improvement_absolute"
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="results/valid_inequalities.csv")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--metric", choices=("percent", "absolute"), default="percent")
    args = parser.parse_args(argv)
    try:
        rows, skipped = read_comparisons(args.csv)
        if not rows:
            raise ValueError("No complete optimal results for formulations 3, 2, and 3_VI")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "lp_comparison.csv", rows, list(rows[0]))
    save_csv(output / "excluded.csv", skipped,
             ["instance", "family", "repetition", "solver_seed", "threads", "time_limit_seconds", "reason"])
    plot_comparisons(rows, output, args.metric)
    print(f"Compared {len(rows)} instances/repetitions; excluded {len(skipped)} incomplete comparisons.")
    return rows


if __name__ == "__main__":
    main()
