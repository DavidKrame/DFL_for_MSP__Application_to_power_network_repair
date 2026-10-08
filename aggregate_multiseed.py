"""
aggregate_multiseed.py: aggregate per-seed DFL results and optionally plot them.

Discovers every network, training size and seed under --root and writes one
aggregated evaluation.json per (network, n_train), overwriting the existing
file on every run. Each seed_<s>/ folder is a single-seed Pseudo_DFL_main.py
run (scripts/slurm/{unbudgeted,budgeted}/dfl_multiseed.sh). The runs share the
training data and test set and differ in the seed that initializes the
RepairNetwork weights; under a repair budget the Monte Carlo perturbations of
the training loss differ as well, because they come from NumPy's global
random state, which is not seeded.

Usage
-----
  python3 aggregate_multiseed.py              # aggregate the default root
  python3 aggregate_multiseed.py --plot       # aggregate + convergence figures
  python3 aggregate_multiseed.py --root experiments/paper_budget0.3/dfl_scaling_multiseed --plot

Reads:  <root>/<net>/dfl_train<N>/seed_<s>/evaluation.json
Writes: <root>/<net>/dfl_train<N>/evaluation.json   (per size, overwritten)
        <root>/<net>/convergence_multiseed.png       (only with --plot)
        <root>/convergence_multiseed.png             (only with --plot, all networks)
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = "experiments/paper_unbudgeted/dfl_scaling_multiseed"

NET_LABELS = {"toy": "Toy", "case14": "Case14"}
COLORS     = {"toy": "#1f77b4",     "case14": "#ff7f0e"}
NET_ORDER  = ["toy", "case14"]  # display order; other networks follow alphabetically


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(output_dir: Path, seeds: list[int]) -> None:
    """Write output_dir/evaluation.json from the seed_<s>/evaluation.json files.

    The Pseudo_DFL entry holds the mean and population std (ddof=0) of the
    per-seed DFL mean costs, the per-seed means, and the seed averages of the
    cost components and rollout time. The other method entries are copied from
    the first seed's file, since only DFL depends on the initialization seed.
    Timing is the first seed's, with training_s and data_generation_s averaged
    over the seeds. Nothing is written if no seed has a DFL mean cost.
    """
    per_seed_means: dict[int, float] = {}
    first_rd = None
    ref_timing: dict = {}
    train_times: list[float] = []
    datagen_times: list[float] = []
    # Per-seed DFL cost components and rollout time; their averages feed the
    # cost-share and computation-time plots of compare_methods.py
    opf_costs:    list[float] = []
    repair_costs: list[float] = []
    shed_mwhs:    list[float] = []
    infer_times:  list[float] = []

    for s in seeds:
        seed_json = output_dir / f"seed_{s}" / "evaluation.json"
        if not seed_json.exists():
            continue
        with open(seed_json) as f:
            rd = json.load(f)

        dfl = rd.get("Pseudo_DFL") or {}
        mc = dfl.get("mean_cost")
        if mc is not None:
            per_seed_means[s] = float(mc)

        if first_rd is None:
            first_rd = rd

        t = rd.get("timing", {})
        if t.get("training_s", 0) > 0:
            train_times.append(float(t["training_s"]))
        if t.get("data_generation_s", 0) > 0:
            datagen_times.append(float(t["data_generation_s"]))
        if not ref_timing:
            ref_timing = dict(t)

        if dfl.get("mean_opf_cost")    is not None: opf_costs.append(float(dfl["mean_opf_cost"]))
        if dfl.get("mean_repair_cost") is not None: repair_costs.append(float(dfl["mean_repair_cost"]))
        if dfl.get("mean_load_shed_mwh") is not None: shed_mwhs.append(float(dfl["mean_load_shed_mwh"]))
        if dfl.get("mean_time_per_scenario_s") is not None: infer_times.append(float(dfl["mean_time_per_scenario_s"]))

    if not per_seed_means:
        return

    means_arr  = np.array(list(per_seed_means.values()))
    grand_mean = float(means_arr.mean())
    grand_std  = float(means_arr.std())

    agg_timing = dict(ref_timing)
    if train_times:
        agg_timing["training_s"] = round(float(np.mean(train_times)), 2)
    if datagen_times:
        agg_timing["data_generation_s"] = round(float(np.mean(datagen_times)), 2)
    agg_timing["n_seeds"] = len(per_seed_means)

    agg = {k: v for k, v in first_rd.items() if k not in ("Pseudo_DFL", "timing")}
    agg["timing"] = agg_timing
    agg["Pseudo_DFL"] = {
        "mean_cost":  grand_mean,
        "std_cost":   grand_std,
        "n_seeds":    len(per_seed_means),
        "seed_means": {str(k): v for k, v in per_seed_means.items()},
        **({"mean_opf_cost":            round(float(np.mean(opf_costs)),    2)} if opf_costs    else {}),
        **({"mean_repair_cost":         round(float(np.mean(repair_costs)), 2)} if repair_costs else {}),
        **({"mean_load_shed_mwh":       round(float(np.mean(shed_mwhs)),    4)} if shed_mwhs    else {}),
        **({"mean_time_per_scenario_s": round(float(np.mean(infer_times)),  6)} if infer_times  else {}),
    }

    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(
        f"  {output_dir.parent.name}/{output_dir.name} "
        f"— {len(per_seed_means)} seed(s): "
        f"mean=${grand_mean:,.0f}  std=${grand_std:,.0f}"
    )


def run_aggregation(root: Path) -> None:
    for net_dir in sorted(root.iterdir()):
        if not net_dir.is_dir():
            continue
        size_dirs = sorted(
            (d for d in net_dir.iterdir() if d.is_dir() and d.name.startswith("dfl_train")),
            key=lambda d: int(d.name.replace("dfl_train", "")),
        )
        if not size_dirs:
            continue
        print(f"\n{net_dir.name}")
        for size_dir in size_dirs:
            seeds = sorted(
                int(s.name.replace("seed_", ""))
                for s in size_dir.iterdir()
                if s.is_dir() and s.name.startswith("seed_")
                and (s / "evaluation.json").exists()
            )
            if seeds:
                aggregate(size_dir, seeds)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _load_network_results(root: Path, net: str) -> list[dict]:
    net_dir = root / net
    if not net_dir.exists():
        return []
    results = []
    for size_dir in net_dir.iterdir():
        if not size_dir.is_dir() or not size_dir.name.startswith("dfl_train"):
            continue
        eval_path = size_dir / "evaluation.json"
        if not eval_path.exists():
            continue
        n_train = int(size_dir.name.replace("dfl_train", ""))
        with open(eval_path) as f:
            d = json.load(f)
        dfl  = d.get("Pseudo_DFL", {})
        mean = dfl.get("mean_cost")
        std  = dfl.get("std_cost")
        if mean is None:
            continue
        results.append({
            "n_train": n_train,
            "mean":    float(mean),
            "std":     float(std) if std is not None else 0.0,
            "n_seeds": int(dfl.get("n_seeds", 1)),
            "ar_mean": (d.get("AlwaysRepair") or {}).get("mean_cost"),
            "nr_mean": (d.get("NoRepair")     or {}).get("mean_cost"),
            "mpc_mean": (d.get("NearSightedMPC") or {}).get("mean_cost"),
        })
    return sorted(results, key=lambda r: r["n_train"])


def _plot_network(ax, results: list[dict], net: str) -> None:
    import matplotlib.ticker as mticker
    # A scale-tagged network such as case14_scale55 reuses its base network's
    # color (case14); anything else falls back to gray, so an unlisted network
    # directory still plots.
    color = COLORS.get(net) or COLORS.get(net.split("_scale")[0], "#7f7f7f")
    ns    = np.array([r["n_train"] for r in results])
    means = np.array([r["mean"]    for r in results])
    stds  = np.array([r["std"]     for r in results])

    ax.plot(ns, means, color=color, lw=2, marker="o", ms=4, zorder=3,
            label="DFL (mean ± std)")
    ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.2, zorder=2)

    # With a fixed test set the baselines do not depend on n_train; each
    # reference line is their mean over the training sizes.
    ar_vals = [r["ar_mean"] for r in results if r["ar_mean"] is not None]
    if ar_vals:
        v = float(np.mean(ar_vals))
        ax.axhline(v, color="gray", ls="--", lw=1.5, label=f"AlwaysRepair  (${v:,.0f})")

    nr_vals = [r["nr_mean"] for r in results if r["nr_mean"] is not None]
    if nr_vals:
        v = float(np.mean(nr_vals))
        ax.axhline(v, color="black", ls=":", lw=1.5, label=f"NoRepair  (${v:,.0f})")

    mpc_vals = [r["mpc_mean"] for r in results if r.get("mpc_mean") is not None]
    if mpc_vals:
        v = float(np.mean(mpc_vals))
        ax.axhline(v, color="#16a085", ls="-.", lw=1.5,
                   label=f"NearSightedMPC  (${v:,.0f})")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("n_train", fontsize=11)
    ax.set_ylabel("Mean cost ($)", fontsize=11)
    seeds_str = "/".join(str(s) for s in sorted({r["n_seeds"] for r in results}))
    label = NET_LABELS.get(net, net)
    ax.set_title(f"{label}\n({seeds_str} seed(s))",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")


def plot_convergence(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nets      = sorted(
        (d.name for d in root.iterdir() if d.is_dir()),
        key=lambda n: (NET_ORDER.index(n) if n in NET_ORDER else len(NET_ORDER), n),
    )
    results   = {net: _load_network_results(root, net) for net in nets}
    available = [net for net in nets if results[net]]

    if not available:
        print("  No aggregated evaluation.json files found — skipping plot.")
        return

    # Per-network figures
    for net in available:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_network(ax, results[net], net)
        fig.suptitle("DFL sample complexity — mean cost ± std across init seeds",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        out = root / net / "convergence_multiseed.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {net} plot saved → {out}")

    # Combined figure (all networks side by side)
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), sharey=False)
    if len(available) == 1:
        axes = [axes]
    for ax, net in zip(axes, available):
        _plot_network(ax, results[net], net)
    fig.suptitle("DFL sample complexity — mean cost ± std across init seeds",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = root / "convergence_multiseed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined plot saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Aggregate multi-seed DFL results and optionally plot")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help=f"Root results directory (default: {DEFAULT_ROOT})")
    p.add_argument("--plot", action="store_true",
                   help="Generate convergence figure after aggregation")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        p.error(f"Root directory not found: {root}")

    run_aggregation(root)

    if args.plot:
        plot_convergence(root)


if __name__ == "__main__":
    main()
