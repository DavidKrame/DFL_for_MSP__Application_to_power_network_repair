#!/usr/bin/env python3
"""Regenerate the paper's figures from the results stored in ``experiments/``.

Run from the repository root:

    python3 paper/make_paper_figures.py

Figures are written to ``figures/``, or to the folder given with ``--out``
(``case14_scale55`` is case14 at s = 55):

    failure_probability_profiles.png            failure-probability profiles
    sensitivity/cost_vs_scale_{toy,case14}.png   DFL vs AlwaysRepair across repair-cost scales
    paper/toy/convergence_multiseed.png          sample complexity, 5-bus, no budget
    paper/case14_scale55/convergence_multiseed.png
    paper/toy/convergence_multiseed_budget.png   sample complexity, 5-bus, phi = 0.3
    paper/case14_scale55/convergence_multiseed_budget.png
    paper/toy/sddip_convergence.png              SDDiP training trace, 5-bus, no budget

The case14 s = 45 sample-complexity curves are also drawn
(``paper/case14/convergence_multiseed*.png``); the paper reports that
configuration in its tables only.

The sensitivity figures read ``experiments/sensitivity/sensitivity_curve.txt``,
the table written by ``scripts/sensitivity_plot.py``.

In the code and in the result files the 5-bus network is called ``toy`` and
the DFL policy is stored under the key ``Pseudo_DFL``.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
OUT = REPO / "figures"

sys.path.insert(0, str(REPO))
import config  # noqa: E402  (failure-model constants shared with the experiments)

UNBUDGETED = "paper_unbudgeted"
BUDGETED = "paper_budget0.3"

NET_COLORS = {"toy": "#1f77b4", "case14": "#ff7f0e", "case14_scale55": "#ff7f0e"}

TICK = 15
LABEL = 17
LEGEND = 14
VALUE = 14

plt.rcParams.update({
    "font.size": TICK,
    "axes.labelsize": LABEL,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,
    "legend.fontsize": LEGEND,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(fig, rel: str, dpi: int = 200) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {_shown(path)}")


# ---------------------------------------------------------------------------
# Sample-complexity curves
# ---------------------------------------------------------------------------

def _sweep(variant: str, net: str) -> list[dict]:
    """Five-seed DFL mean and across-seed std per training size, plus the baseline means."""
    rows = []
    for size_dir in (EXP / variant / "dfl_scaling_multiseed" / net).glob("dfl_train*"):
        ev = size_dir / "evaluation.json"
        if not ev.exists():
            continue
        d = _load(ev)
        dfl = d.get("Pseudo_DFL") or {}
        if dfl.get("mean_cost") is None:
            continue
        rows.append({
            "n": int(size_dir.name.replace("dfl_train", "")),
            "mean": float(dfl["mean_cost"]),
            "std": float(dfl.get("std_cost") or 0.0),
            "ar": (d.get("AlwaysRepair") or {}).get("mean_cost"),
            "nr": (d.get("NoRepair") or {}).get("mean_cost"),
            "mpc": (d.get("NearSightedMPC") or {}).get("mean_cost"),
        })
    return sorted(rows, key=lambda r: r["n"])


def plot_convergence(variant: str, net: str, rel: str) -> None:
    """Plot the DFL mean cost against n_train with a band of +/- one across-seed std.

    Each baseline is a horizontal line at its mean over the sweep; the baselines
    do not depend on n_train, so every size stores the same value.
    """
    rows = _sweep(variant, net)
    color = NET_COLORS[net]
    ns = np.array([r["n"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    stds = np.array([r["std"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, means / 1e6, color=color, lw=2.4, marker="o", ms=5, zorder=3,
            label="DFL (mean ± std)")
    ax.fill_between(ns, (means - stds) / 1e6, (means + stds) / 1e6,
                    color=color, alpha=0.2, zorder=2)

    levels = [float(np.max(means + stds))]
    for key, label, col, ls in [("ar", "AlwaysRepair", "gray", "--"),
                                ("nr", "NoRepair", "black", ":"),
                                ("mpc", "NearSightedMPC", "#16a085", "-.")]:
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            v = float(np.mean(vals))
            levels.append(v)
            ax.axhline(v / 1e6, color=col, ls=ls, lw=1.8, label=f"{label}  (${v:,.0f})")

    # The axis fits the data with a small margin; the legend goes in the
    # emptiest region inside the plot and may cross a reference line.
    lo = min(float(np.min(means - stds)), *levels)
    hi = max(levels)
    pad = hi - lo
    ax.set_ylim((lo - 0.05 * pad) / 1e6, (hi + 0.08 * pad) / 1e6)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:,.0f}k"))
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"$n_{\mathrm{train}}$")
    ax.set_ylabel("Mean cost ($M)")
    ax.legend(loc="best", framealpha=0.85)
    fig.tight_layout()
    _save(fig, rel)


# ---------------------------------------------------------------------------
# Repair-cost sensitivity
# ---------------------------------------------------------------------------

def _sensitivity_table() -> dict:
    """Read the (AR - DFL) / DFL table written by scripts/sensitivity_plot.py.

    Returns {network: (scales, gaps in %)}; cells marked '---' (no result) are left out.
    """
    lines = (EXP / "sensitivity" / "sensitivity_curve.txt").read_text().splitlines()
    nets = lines[0].split()[1:]
    table = {n: ([], []) for n in nets}
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        scale = float(parts[0])
        for net, cell in zip(nets, parts[1:]):
            if cell != "---":
                table[net][0].append(scale)
                table[net][1].append(float(cell.rstrip("%")))
    return table


def plot_sensitivity(net: str, label: str, color: str, rel: str) -> None:
    xs, ys = _sensitivity_table()[net]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", ms=7, color=color, lw=2.2, label=label)
    # Points below s = 20 crowd together near the left edge and their labels
    # would overlap, so values are written only from s = 20 on.
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x < 20:
            continue
        last = i == len(xs) - 1
        ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                    xytext=(-10, 8) if last else (10, -10),
                    ha="right" if last else "left", va="bottom" if last else "top",
                    fontsize=VALUE - 1, color=color)
    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_xlabel("Repair cost scale $s$")
    ax.set_ylabel("(AR − DFL) / DFL  (%)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.margins(y=0.1)
    fig.tight_layout()
    _save(fig, rel)


# ---------------------------------------------------------------------------
# SDDiP trace
# ---------------------------------------------------------------------------

def plot_sddip(rel: str) -> None:
    # "upper_bounds" holds the cost of the single forward pass of each
    # iteration, not a statistical upper bound, hence the legend label.
    conv = _load(EXP / UNBUDGETED / "toy" / "SDDiP" / "sddip_solution.json")["convergence"]
    it = np.arange(1, len(conv["upper_bounds"]) + 1)
    ub = np.array(conv["upper_bounds"]) / 1e6
    lb = np.array(conv["lower_bounds"]) / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(it, ub, "b-o", lw=1.5, ms=4, label="Forward-pass cost")
    axes[0].plot(it, lb, "r-s", lw=2, ms=4, label="Lower bound")
    axes[0].fill_between(it, lb, ub, alpha=0.2, color="gray")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost ($M)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(it, np.array(conv["gaps"]) * 100, "g-^", lw=1.5, ms=4)
    # Reference line only: SDDiP stops on its iteration and time limits, not on the gap.
    axes[1].axhline(1.0, color="red", ls="--", alpha=0.7, label="1% target")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Relative gap (%)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, rel)


# ---------------------------------------------------------------------------
# Failure-probability profiles
# ---------------------------------------------------------------------------

# The two vulnerability classes of the experiments (config.py).
CLASSES = {
    "Standard": {"bounds": (config.FAILURE_P_MAX_STD_LOW, config.FAILURE_P_MAX_STD_HIGH),
                 "color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Elevated": {"bounds": (config.FAILURE_P_MAX_ELV_LOW, config.FAILURE_P_MAX_ELV_HIGH),
                 "color": "#D55E00", "linestyle": "--", "marker": "s"},
}


def failure_probability(t: np.ndarray, p_max: float) -> np.ndarray:
    """Log-linear logit failure probability, evaluated at continuous t.

    Same formula as ModelParameters._compute_lambda, which evaluates it at the
    integer stages 1..T only; here it is drawn from t = 0 so that the peak at
    t* is visibly inside the plot.
    """
    t_star, k = float(config.FAILURE_T_STAR), float(config.FAILURE_K)
    delta = (t_star + k) * np.log(t_star + k) - t_star
    b = np.log(p_max / (1.0 - p_max)) / delta
    a = (t_star + k) * b
    return 1.0 / (1.0 + np.exp(-(a * np.log(t + k) - b * t)))


def plot_failure_profiles(rel: str) -> None:
    """Draw each class at its midpoint p_max, shade its p_max range, mark stages 1..T."""
    horizon = int(config.T)
    t_dense = np.linspace(0.0, float(horizon), 800)
    stages = np.arange(1, horizon + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(6.7, 4.0), constrained_layout=True)
    for name, spec in CLASSES.items():
        low, high = spec["bounds"]
        mid = (low + high) / 2.0
        c = spec["color"]
        lo, hi = failure_probability(t_dense, low), failure_probability(t_dense, high)
        # Shade the pointwise range of the class. The curves of one class cross
        # later in the horizon, so the envelope uses min/max rather than
        # assuming the larger p_max stays on top.
        ax.fill_between(t_dense, np.minimum(lo, hi), np.maximum(lo, hi),
                        color=c, alpha=0.18, lw=0)
        ax.plot(t_dense, failure_probability(t_dense, mid), color=c, lw=2.0,
                ls=spec["linestyle"],
                label=rf"{name}: $p_{{\max}}\sim\mathcal{{U}}[{low:.2f},{high:.2f}]$")
        ax.plot(stages, failure_probability(stages, mid), ls="none",
                marker=spec["marker"], ms=3.5, mfc="white", mec=c, mew=1.0)
    t_star = float(config.FAILURE_T_STAR)
    ax.axvline(t_star, color="0.25", lw=1.2, ls=":", zorder=1)
    ax.text(t_star + 0.15, 0.955, rf"$t^*={config.FAILURE_T_STAR}$", color="0.25",
            ha="left", va="top", fontsize=12)
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, horizon + 1))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=11)
    ax.set_xlabel(r"Stage $t$", fontsize=12)
    ax.set_ylabel(r"Conditional failure probability $\lambda^t[k]$", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=11.5)
    _save(fig, rel, dpi=300)


def _shown(path: Path) -> Path:
    """Return ``path`` relative to the repository when it lies inside it."""
    return path.relative_to(REPO) if path.is_relative_to(REPO) else path


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="Redraw the figures from experiments/.")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output folder (default: figures/ of the repository)")
    OUT = parser.parse_args().out.resolve()
    print("Writing figures to", _shown(OUT))
    plot_failure_profiles("failure_probability_profiles.png")
    plot_sensitivity("toy", "5-bus", "steelblue", "sensitivity/cost_vs_scale_toy.png")
    plot_sensitivity("case14", "Case 14", "chocolate", "sensitivity/cost_vs_scale_case14.png")
    for net in ["toy", "case14", "case14_scale55"]:
        plot_convergence(UNBUDGETED, net, f"paper/{net}/convergence_multiseed.png")
        plot_convergence(BUDGETED, net, f"paper/{net}/convergence_multiseed_budget.png")
    plot_sddip("paper/toy/sddip_convergence.png")


if __name__ == "__main__":
    main()
