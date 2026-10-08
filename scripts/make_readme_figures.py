#!/usr/bin/env python3
"""Draw the README figures that the paper does not have.

Run from anywhere:

    python3 scripts/make_readme_figures.py

Figures are written to ``docs/images/``, or to the folder given with ``--out``:

    toy_network.png          the synthetic 5-bus network (``toy``) with its data
    mean_cost_by_method.png  mean test cost of each method in the six reported
                             configurations, read from ``experiments/``

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
METHODS = [("NoRepair", "NoRepair", "#E06555"), ("AlwaysRepair", "AlwaysRepair", "#F2A03D"),
           ("NearSightedMPC", "Near-sighted MPC", "#3BA287"), ("SAA", "SAA", "#9B7FB8"),
           ("Pseudo_DFL", "DFL", "#5B8FC2")]
LANES = [("toy", "5-bus\n$s = 100$"), ("case14", "case14\n$s = 45$"), ("case14_scale55", "case14\n$s = 55$")]
VARIANTS = [("paper_unbudgeted", "Without a budget"), ("paper_budget0.3", "With a budget ($\\varphi = 0.3$)")]


def _lane_costs(variant: str, lane: str) -> dict:
    """Mean test cost of each method at the best training size of the DFL sweep.

    The best size is the one with the lowest five-seed DFL mean, as in the
    results tables. SAA comes from its own folder, the other methods from the
    aggregated ``evaluation.json`` of that size.
    """
    exp = REPO / "experiments" / variant
    sizes = []
    for f in (exp / "dfl_scaling_multiseed" / lane).glob("dfl_train*/evaluation.json"):
        sizes.append((json.loads(f.read_text())["Pseudo_DFL"]["mean_cost"], f))
    best = json.loads(min(sizes)[1].read_text())
    costs = {key: best[key]["mean_cost"] for key, _, _ in METHODS if key != "SAA"}
    saa = json.loads((exp / lane / "SAA" / "solution.json").read_text())
    costs["SAA"] = saa["out_of_sample_evaluation"]["mean_cost"]
    return costs


def draw_cost_bars(path: Path) -> None:
    """Grouped bars of the mean test cost, one panel per variant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    width = 0.16
    for ax, (variant, title) in zip(axes, VARIANTS):
        for g, (lane, _) in enumerate(LANES):
            costs = _lane_costs(variant, lane)
            for m, (key, label, color) in enumerate(METHODS):
                x = g + (m - 2) * width
                ax.bar(x, costs[key] / 1e6, width, color=color, edgecolor="black",
                       lw=1.6 if key == "Pseudo_DFL" else 0.6, label=label if g == 0 else None)
            change = (costs["Pseudo_DFL"] - costs["AlwaysRepair"]) / costs["AlwaysRepair"] * 100
            ax.text(g + 2 * width, costs["Pseudo_DFL"] / 1e6 + 0.06, f"{change:+.1f}%",
                    ha="center", va="bottom", fontsize=TICK - 2, fontweight="bold")
        ax.set_xticks(range(len(LANES)))
        ax.set_xticklabels([lab for _, lab in LANES], fontsize=TICK)
        ax.set_title(title, fontsize=LABEL)
        ax.tick_params(axis="y", labelsize=TICK)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean cost per test scenario (million USD)", fontsize=LABEL - 2)
    axes[0].set_ylim(0, 4.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), fontsize=TICK, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
    draw_cost_bars(out / "mean_cost_by_method.png")


if __name__ == "__main__":
    main()
