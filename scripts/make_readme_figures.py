#!/usr/bin/env python3
"""Draw the README figures that the paper does not have.

Run from anywhere:

    python3 scripts/make_readme_figures.py

Figures are written to ``docs/images/``, or to the folder given with ``--out``:

    toy_network.png          the synthetic 5-bus network (``toy``) with its data
    budget_cost_bars.png     budgeted mean cost of every method on the 5-bus network
                             and on case14 at s = 55, with SDDiP's simulated mean
                             and lower bound, read from ``experiments/``

The generators, loads and lines of the network, with their capacities, costs,
demands, susceptances and flow limits, are read from
``network_data.create_network("toy")``. Only the bus positions and the side on
which each generator or load is drawn are set here.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "images"

sys.path.insert(0, str(REPO))
from network_data import create_network  # noqa: E402

# Font sizes of paper/make_paper_figures.py. The labels use the axis-label
# size so that they stay readable when the figure is shown about 600 px wide.
TICK = 15
LABEL = 17

plt.rcParams.update({"font.size": LABEL})

INK = "#1a1a1a"
LINE = "#7a7a7a"
BUS = "#1f77b4"   # the 5-bus color of NET_COLORS in paper/make_paper_figures.py
# Okabe-Ito green and vermillion, which color-blind readers can tell apart
GEN = "#009E73"
LOAD = "#D55E00"

# Lengths in inches: the canvas is drawn at one data unit per inch.
R_BUS = 0.27
R_GEN = 0.25
STUB = 0.30       # gap between a bus and its generator
ARROW = 0.75      # length of a load arrow

# Bus positions, numbered from 1 as in the README. The loop 1-2-4-3 is a
# square and bus 5 hangs off bus 4.
BUS_XY = {1: (0.0, 0.0), 2: (3.4, 0.0), 3: (0.0, -2.8), 4: (3.4, -2.8), 5: (7.2, -2.8)}
# Direction in which the generator or load at each bus is drawn
OUTWARD = {1: (0, 1), 2: (0, 1), 3: (0, -1), 4: (0, -1), 5: (0, -1)}


def _num(x: float) -> str:
    return f"{x:g}"


def _gen_symbol(ax, center, r, lw=2.5) -> list:
    """Circle with a sine wave, the usual symbol of a generator."""
    cx, cy = center
    t = np.linspace(0.0, 2.0 * np.pi, 60)
    circle = ax.add_patch(Circle(center, r, fc="white", ec=GEN, lw=lw, zorder=3))
    wave, = ax.plot(cx + 0.6 * r * (t / np.pi - 1.0), cy + 0.35 * r * np.sin(t),
                    color=GEN, lw=0.8 * lw, solid_capstyle="round", zorder=4)
    return [circle, wave]


def _load_arrow(ax, start, end, lw=2.5, head=22):
    return ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head,
                                        lw=lw, color=LOAD, shrinkA=0, shrinkB=0, zorder=2))


def _key(ax, x, y) -> None:
    """Draw a framed key with its symbols centered on x and its first row at height y."""
    step = 0.42
    rows = ["generator: capacity, cost", "load: demand", "line: susceptance $b$, flow limit"]
    parts = []
    for i, text in enumerate(rows):
        yi = y - i * step
        if i == 0:
            parts += _gen_symbol(ax, (x, yi), 0.14, lw=1.8)
        elif i == 1:
            parts.append(_load_arrow(ax, (x - 0.2, yi), (x + 0.2, yi), lw=2.0, head=16))
        else:
            parts += ax.plot([x - 0.2, x + 0.2], [yi, yi], color=LINE, lw=3, zorder=1)
        parts.append(ax.text(x + 0.36, yi, text, ha="left", va="center",
                             fontsize=TICK, color=INK))

    # The frame keeps the key symbols from being read as part of the network.
    renderer = ax.figure.canvas.get_renderer()
    box = Bbox.union([p.get_window_extent(renderer) for p in parts])
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(box.get_points())
    pad = 0.18
    ax.add_patch(FancyBboxPatch((x0 - pad, y0 - pad), x1 - x0 + 2 * pad, y1 - y0 + 2 * pad,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                fc="white", ec="#cfcfcf", lw=1.0, zorder=0))


def draw_toy_network(path: Path) -> None:
    """Draw the 5-bus network with its data and save it to ``path``."""
    net = create_network("toy")
    # The layout is made for this network: five buses, and no bus with both a
    # generator and a load (the two would be drawn on the same side).
    assert net.n_buses == len(BUS_XY)
    assert not set(net.gen_buses) & set(net.load_buses)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-2, 10)
    ax.set_ylim(-6, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    for j, (a, b) in enumerate(net.line_connections):
        (x0, y0), (x1, y1) = BUS_XY[a + 1], BUS_XY[b + 1]
        ax.plot([x0, x1], [y0, y1], color=LINE, lw=3, zorder=1)
        lo, hi = net.line_flow_min[j], net.line_flow_max[j]
        limit = f"±{_num(hi)} MW" if np.isclose(lo, -hi) else f"{_num(lo)} to {_num(hi)} MW"
        label = (rf"$\mathbf{{L{j + 1}}}$" + "\n"
                 + f"$b$ = {_num(net.line_susceptances[j])} MW/rad\n{limit}")
        # The white box hides the line behind the label, so the line runs into it.
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center",
                multialignment="center", color=INK, linespacing=1.3, zorder=3,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))

    for g, bus in enumerate(net.gen_buses):
        (x, y), (dx, dy) = BUS_XY[bus + 1], OUTWARD[bus + 1]
        d = R_BUS + STUB + R_GEN
        cx, cy = x + dx * d, y + dy * d
        ax.plot([x + dx * R_BUS, cx - dx * R_GEN], [y + dy * R_BUS, cy - dy * R_GEN],
                color=GEN, lw=2.5, zorder=2)
        _gen_symbol(ax, (cx, cy), R_GEN)
        label = (rf"$\mathbf{{G{g + 1}}}$" + "\n"
                 + f"{_num(net.gen_pmax[g])} MW\n\\${_num(net.gen_cost[g])}/MWh")
        ax.text(cx + R_GEN + 0.15, cy, label, ha="left", va="center", color=INK,
                linespacing=1.3, zorder=3)

    for i, bus in enumerate(net.load_buses):
        (x, y), (dx, dy) = BUS_XY[bus + 1], OUTWARD[bus + 1]
        tip = (x + dx * (R_BUS + ARROW), y + dy * (R_BUS + ARROW))
        _load_arrow(ax, (x + dx * R_BUS, y + dy * R_BUS), tip)
        ax.text(tip[0], tip[1] + dy * 0.1, f"{_num(net.load_nominal[i])} MW",
                ha="center", va="bottom" if dy > 0 else "top", color=INK, zorder=3)

    for k, (x, y) in BUS_XY.items():
        ax.add_patch(Circle((x, y), R_BUS, fc=BUS, ec="white", lw=2, zorder=5))
        # va="center" counts the room below the baseline, which digits do not
        # use, and would set the numbers slightly high in the circle.
        ax.text(x, y, str(k), ha="center", va="center_baseline", color="white",
                fontweight="bold", zorder=6)

    _key(ax, 4.4, 0.5)

    # The axes fill the canvas, so bbox_inches="tight" would keep its empty
    # margins. Crop to the drawn elements instead.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    drawn = [a.get_window_extent(renderer) for a in (*ax.texts, *ax.patches, *ax.lines)]
    crop = Bbox.union(drawn).transformed(fig.dpi_scale_trans.inverted()).padded(0.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches=crop, facecolor="white")
    plt.close(fig)
    print(f"  saved {_shown(path)}")


# Method colours of the paper figures
METHOD_COLORS = {"NoRepair": "#E06555", "AlwaysRepair": "#F2A03D", "MPC": "#3BA287",
                 "SAA": "#9B7FB8", "DFL": "#5B8FC2", "SDDiP": "#8C8C8C"}
BUDGETED = "paper_budget0.3"
PANELS = [("toy", "toy", 100, "5-bus ($s$ = 100)"), ("case14_scale55", "case14", 55, "case14 ($s$ = 55)")]


def _budget_lane(folder: str, network_name: str, scale: float) -> dict:
    """(mean, std) of every method in the budgeted lane, in USD.

    NoRepair, AlwaysRepair, Near-sighted MPC, SAA and DFL are scored on the
    500 common test scenarios, DFL at the best training size of its sweep (the
    lowest five-seed mean, as in the results tables). DFL's spread is the
    across-scenario std averaged over its five seeds. SDDiP has no score on the
    common test set: its mean is that of its own 500 policy simulations,
    rebuilt from the saved averages as in paper/reproduce_paper_numbers.py, and
    its lower bound is the final training bound. Neither has a saved std.
    """
    sys.path.insert(0, str(REPO / "exact_dp"))
    from exact_dp_toy import reconstruct_saved_sddip_mean
    from model_parameters import create_model_parameters

    exp = REPO / "experiments" / BUDGETED
    sizes = [(json.loads(f.read_text())["Pseudo_DFL"]["mean_cost"], f.parent)
             for f in (exp / "dfl_scaling_multiseed" / folder).glob("dfl_train*/evaluation.json")]
    best_dir = min(sizes)[1]
    ev = json.loads((best_dir / "evaluation.json").read_text())
    seed_std = np.mean([json.loads(f.read_text())["Pseudo_DFL"]["std_cost"]
                        for f in sorted(best_dir.glob("seed_*/evaluation.json"))])
    saa = json.loads((exp / folder / "SAA" / "solution.json").read_text())["out_of_sample_evaluation"]
    sddip = json.loads((exp / folder / "SDDiP" / "sddip_solution.json").read_text())
    network = create_network(network_name)
    params = create_model_parameters(network, seed=42, t=14, tau=1, gen_repair_base=550.0 * scale,
                                     line_repair_base=350.0 * scale, repair_budget_fraction=0.3)
    return {
        "NoRepair": (ev["NoRepair"]["mean_cost"], ev["NoRepair"]["std_cost"]),
        "AlwaysRepair": (ev["AlwaysRepair"]["mean_cost"], ev["AlwaysRepair"]["std_cost"]),
        "MPC": (ev["NearSightedMPC"]["mean_cost"], ev["NearSightedMPC"]["std_cost"]),
        "SAA": (saa["mean_cost"], saa["std_cost"]),
        "DFL": (ev["Pseudo_DFL"]["mean_cost"], seed_std),
        "SDDiP mean": (reconstruct_saved_sddip_mean(sddip, network, params), 0.0),
        "SDDiP LB": (sddip["lower_bound"], 0.0),
        "n_train": int(best_dir.name.replace("dfl_train", "")),
    }


def draw_budget_bars(path: Path) -> None:
    """Budgeted mean costs of the methods, with SDDiP's simulated mean and lower bound as references."""
    order = ["NoRepair", "AlwaysRepair", "MPC", "SAA", "DFL", "SDDiP mean", "SDDiP LB"]
    x = np.array([0, 1, 2, 3, 4, 5.4, 6.4])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 6.4), sharey=True)
    for ax, (folder, network_name, scale, label) in zip(axes, PANELS):
        stats = _budget_lane(folder, network_name, scale)
        print(f"  {label.replace('$', '')}, DFL at {stats['n_train']:,} scenarios:")
        for m in order:
            print(f"    {m:13s} mean {stats[m][0]:>12,.0f}   std {stats[m][1]:>10,.0f}")
        means = np.array([stats[m][0] for m in order]) / 1e6
        errs = np.array([stats[m][1] for m in order]) / 1e6
        colors = [METHOD_COLORS[m.split()[0]] for m in order]
        bars = ax.bar(x, means, color=colors, edgecolor="black", lw=0.8)
        bars[-1].set_facecolor("white")
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor(METHOD_COLORS["SDDiP"])
        ax.errorbar(x[:5], means[:5], yerr=errs[:5], fmt="none", ecolor="black",
                    capsize=5, lw=1.8, capthick=1.8)
        for xi, m, e in zip(x, means, errs):
            ax.text(xi, m + e + 0.06, f"{m:.2f}", ha="center", va="bottom", fontsize=TICK - 1, fontweight="bold")
        ax.axvline(4.7, color="0.4", ls=":", lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["NoRepair", "AlwaysRepair", "MPC", "SAA", "DFL", "SDDiP\nmean", "SDDiP\nlower bound"],
                           rotation=20, fontsize=TICK)
        ax.set_xlabel(label, fontsize=LABEL, labelpad=8)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 4.75)
    axes[0].set_ylabel("Mean total cost (million USD)", fontsize=LABEL)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"  saved {_shown(path)}")


def _shown(path: Path) -> Path:
    """Return ``path`` relative to the repository when it lies inside it."""
    return path.relative_to(REPO) if path.is_relative_to(REPO) else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the README figures.")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output folder (default: docs/images/ of the repository)")
    out = parser.parse_args().out.resolve()
    print("Writing figures to", _shown(out))
    draw_toy_network(out / "toy_network.png")
    draw_budget_bars(out / "budget_cost_bars.png")


if __name__ == "__main__":
    main()
