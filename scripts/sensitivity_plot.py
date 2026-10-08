#!/usr/bin/env python3
"""
sensitivity_plot.py: plot (AR - DFL) / DFL against the repair-cost scale s.

Reads evaluation.json from each scale_<s>/<network>/ directory under --base
(default experiments/sensitivity, written by
scripts/slurm/sensitivity/sensitivity_sweep.sh). The gap is computed from the
mean per-scenario costs of AlwaysRepair (AR) and DFL. With the default --out it
writes to experiments/sensitivity/:
  1. sensitivity_curve.png          toy and case14 on one graph
  2. sensitivity_curve_toy.png      toy (5-bus) only
  3. sensitivity_curve_case14.png   case14 only
  4. sensitivity_curve.txt          summary table
Files 2-4 take their names from --out (my_figure.png gives my_figure_toy.png,
my_figure_case14.png and my_figure.txt).

Usage (from the repository root):
    python3 scripts/sensitivity_plot.py
    python3 scripts/sensitivity_plot.py --base /abs/path/to/sensitivity
    python3 scripts/sensitivity_plot.py --out my_figure.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# config.py defaults ($ and $/MWh), used only by x_ratio()
GEN_REPAIR_BASE    = 550.0
LINE_REPAIR_BASE   = 350.0
LOAD_SHEDDING_COST = 1000.0

NETWORKS = [
    ("toy",    "Toy (5-bus)",  "steelblue"),
    ("case14", "Case 14",      "chocolate"),
]


def discover_scales(base: Path):
    scales = []
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("scale_"):
            try:
                scales.append(float(d.name[len("scale_"):]))
            except ValueError:
                pass
    return sorted(scales)


def x_ratio(scale):
    """Mean of the two repair-cost bases at scale s over the shedding cost (0.45 s, in MWh).

    The plots put s itself on the x-axis and do not call this function.
    """
    return scale * (GEN_REPAIR_BASE + LINE_REPAIR_BASE) / 2.0 / LOAD_SHEDDING_COST


def read_gap(base: Path, scale: float, network: str):
    """Return (AR - DFL) / DFL in percent, or None if the file or a cost list is missing.

    The directory is looked up as scale_<str(scale)>, e.g. scale_10.0 (as written
    by sensitivity_sweep.sh); a folder named scale_10 is not read.
    """
    ef = base / f"scale_{scale}" / network / "evaluation.json"
    if not ef.exists():
        return None
    data = json.loads(ef.read_text())
    dfl = data.get("Pseudo_DFL", {})
    ar  = data.get("AlwaysRepair", {})
    dfl_costs = dfl.get("costs", []) if isinstance(dfl, dict) else []
    ar_costs  = ar.get("costs",  []) if isinstance(ar,  dict) else []
    if not dfl_costs or not ar_costs:
        return None
    mean_dfl = sum(dfl_costs) / len(dfl_costs)
    mean_ar  = sum(ar_costs)  / len(ar_costs)
    if mean_dfl == 0:
        return None
    return (mean_ar - mean_dfl) / mean_dfl * 100.0


def _draw_curve(ax, base, scales, net_key, net_label, color):
    """Draw one network's curve (x = scale s) on ax. Returns (xs, ys) for the plotted points."""
    xs, ys = [], []
    for scale in scales:
        gap = read_gap(base, scale, net_key)
        if gap is not None:
            xs.append(scale)
            ys.append(gap)
    if not xs:
        print(f"  [skip] {net_key}: no evaluation.json found", file=sys.stderr)
        return xs, ys
    ax.plot(xs, ys, marker="o", label=net_label, color=color, linewidth=1.8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:+.1f}%", (x, y),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color=color)
    return xs, ys


def _style_ax(ax, title=None):
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Repair Cost Scale")
    ax.set_ylabel("(AR − DFL) / DFL  (%)")
    if title:
        ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="experiments/sensitivity",
                        help="Root sensitivity results directory")
    parser.add_argument("--out", default="experiments/sensitivity/sensitivity_curve.png",
                        help="Output path for the combined figure")
    args = parser.parse_args()

    base = Path(args.base)
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    scales = discover_scales(base)
    if not scales:
        print(f"No scale_* directories found under {base}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(scales)} scales: {scales}")

    # ------------------------------------------------------------------
    # 1. Combined figure with both networks
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for net_key, net_label, color in NETWORKS:
        _draw_curve(ax, base, scales, net_key, net_label, color)
    _style_ax(ax, title="DFL improvement over AlwaysRepair vs cost ratio")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out.resolve()}")

    # ------------------------------------------------------------------
    # 2. One figure per network
    # ------------------------------------------------------------------
    for net_key, net_label, color in NETWORKS:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_curve(ax, base, scales, net_key, net_label, color)
        _style_ax(ax, title=f"DFL improvement over AlwaysRepair vs cost ratio — {net_label}")
        fig.tight_layout()
        ind_path = out.with_name(out.stem + f"_{net_key}" + out.suffix)
        fig.savefig(ind_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {ind_path.resolve()}")

    # ------------------------------------------------------------------
    # 3. Text summary table
    # ------------------------------------------------------------------
    # paper/make_paper_figures.py and paper/reproduce_paper_numbers.py parse this table:
    # a header row, a separator row, then one row per scale ('---' = no result).
    col_w = 10
    header = f"{'scale':>6}" + "".join(f"  {n[0]:>{col_w}}" for n in NETWORKS)
    sep    = "-" * len(header)

    print(f"\nSensitivity table:  (AR−DFL)/DFL  (%)")
    print(header)
    print(sep)
    lines = [header, sep]
    for scale in scales:
        row = f"{scale:>6}"
        for net_key, _, _ in NETWORKS:
            gap = read_gap(base, scale, net_key)
            row += f"  {gap:>+8.2f}%" if gap is not None else f"  {'---':>{col_w}}"
        print(row)
        lines.append(row)

    txt_path = out.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Table  → {txt_path.resolve()}")


if __name__ == "__main__":
    main()
