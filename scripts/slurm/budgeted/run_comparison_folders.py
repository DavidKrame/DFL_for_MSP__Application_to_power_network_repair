"""
run_comparison_folders.py: run compare_methods.py on every multiseed DFL result
with a repair budget, at the aggregated and at the per-seed level.

For every network folder <net> and training size N found under --root:
  - Aggregated level : <root>/<net>/dfl_train<N>/comparison/
  - Per-seed level   : <root>/<net>/dfl_train<N>/seed_<s>/comparison/

The aggregated level reads the evaluation.json that aggregate_multiseed.py
(dfl_aggregate.sh) writes for each size; a size or seed without evaluation.json
is skipped. The network and its repair-cost scale s come from the folder name:
case14_scale55 is case14 at s = 55, and a plain name takes s from PAPER_SCALES.

--repair-budget-fraction is passed on to compare_methods.py, which uses it for
the rollouts it simulates for its plots. Pass the phi of the runs (0.3 for
experiments/paper_budget0.3, as compare.sh does); without the flag,
compare_methods.py uses REPAIR_BUDGET_FRACTION from config.py (None, no budget).

SAA and SDDiP are optional. With --saa-root / --sddip-root the script uses
<saa-root>/<net>/SAA/ and <sddip-root>/<net>/SDDiP/ (lower-case saa/ and sddip/
also work), the folders that saa.sh and sddip.sh write.

Run it from the repository root: compare_methods.py and the default --root are
relative paths.

Usage
-----
  # All levels, DFL and the baselines from evaluation.json only
  python3 scripts/slurm/budgeted/run_comparison_folders.py --repair-budget-fraction 0.3

  # With SAA and SDDiP results
  python3 scripts/slurm/budgeted/run_comparison_folders.py \\
      --repair-budget-fraction 0.3 \\
      --saa-root   experiments/paper_budget0.3 \\
      --sddip-root experiments/paper_budget0.3

  # Aggregated level only (skip per-seed)
  python3 scripts/slurm/budgeted/run_comparison_folders.py --repair-budget-fraction 0.3 \\
      --aggregated-only

  # One network folder, as in each compare.sh array task
  python3 scripts/slurm/budgeted/run_comparison_folders.py --repair-budget-fraction 0.3 \\
      --network toy
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Repair-cost scale s (--repair-base-scale) of each plain network folder; must
# match SCALES in dfl_multiseed.sh in this folder. Folders such as case14_scale55
# carry their scale in the name.
PAPER_SCALES = {"toy": 100, "case14": 45}


def _resolve_network_scale(net_dir_name):
    """Map a result-folder name to (network name, repair-cost scale s).

    A folder like ``case14_scale55`` holds network ``case14`` evaluated at a
    non-default repair-cost scale (55); the scale is encoded in the folder name
    so one tree can hold several scales of the same network side by side. A
    plain folder name takes its scale from PAPER_SCALES, or 1.0 (the config.py
    default of REPAIR_BASE_SCALE) for a network not listed there.
    """
    if "_scale" in net_dir_name:
        base, _, s = net_dir_name.partition("_scale")
        scale = int(s) if s.isdigit() else float(s)
        return base, scale
    return net_dir_name, PAPER_SCALES.get(net_dir_name, 1.0)


# Seed and size of the DFL test set (EVAL_SEED and N_EVAL in dfl_multiseed.sh).
# compare_methods.py regenerates these test scenarios for its plots. It is not
# passed --seed, so its default data seed (42) must equal SEED in dfl_multiseed.sh.
EVAL_SEED = 1000
N_EVAL    = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_method_dir(root, net, subdirs):
    """Return the first existing root/net/<sub> (sub in subdirs) as a str, or None."""
    if root is None:
        return None
    for sub in subdirs:
        p = Path(root) / net / sub
        if p.exists():
            return str(p)
    return None


def _run_compare(dfl_dir, net, scale, saa_dir, sddip_dir, budget=None):
    """Run compare_methods.py on one DFL directory and return its exit code.

    compare_methods.py writes the comparison/ folder inside dfl_dir. A budget
    other than None is passed on as --repair-budget-fraction.
    """
    cmd = [
        sys.executable, "compare_methods.py",
        "--network",           net,
        "--dfl-dir",           str(dfl_dir),
        "--output-dir",        str(dfl_dir),
        "--eval-seed",         str(EVAL_SEED),
        "--n-eval",            str(N_EVAL),
        "--repair-base-scale", str(scale),
    ]
    if saa_dir:
        cmd += ["--saa-dir", saa_dir]
    if sddip_dir:
        cmd += ["--sddip-dir", sddip_dir]
    if budget is not None:
        cmd += ["--repair-budget-fraction", str(budget)]

    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  WARNING: compare_methods.py exited {rc} for {dfl_dir}")
    return rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate comparison folders for all multiseed DFL results"
    )
    p.add_argument("--root", default="experiments/paper_budget0.3/dfl_scaling_multiseed",
                   help="Root of multiseed results (default: experiments/paper_budget0.3/dfl_scaling_multiseed)")
    p.add_argument("--network", default=None,
                   help="Restrict to one network (e.g. toy). Default: all.")
    p.add_argument("--saa-root", default=None,
                   help="Root whose <net>/SAA/ subdirs contain SAA results "
                        "(e.g. experiments/paper_budget0.3)")
    p.add_argument("--sddip-root", default=None,
                   help="Root whose <net>/SDDiP/ subdirs contain SDDiP results "
                        "(e.g. experiments/paper_budget0.3)")
    p.add_argument("--repair-budget-fraction", type=float, default=None,
                   help="Per-stage budget fraction phi forwarded to compare_methods.py")
    p.add_argument("--aggregated-only", action="store_true",
                   help="Only run aggregated-level comparisons (skip per-seed)")
    p.add_argument("--seeds-only", action="store_true",
                   help="Only run per-seed comparisons (skip aggregated)")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    net_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )
    if args.network:
        net_dirs = [d for d in net_dirs if d.name == args.network]
        if not net_dirs:
            sys.exit(f"Network '{args.network}' not found under {root}")

    total_ok = total_fail = total_skip = 0

    for net_dir in net_dirs:
        net   = net_dir.name
        real_net, scale = _resolve_network_scale(net)

        saa_dir   = _find_method_dir(args.saa_root,   net, ["SAA",   "saa"])
        sddip_dir = _find_method_dir(args.sddip_root, net, ["SDDiP", "sddip"])

        size_dirs = sorted(
            (d for d in net_dir.iterdir()
             if d.is_dir() and d.name.startswith("dfl_train")),
            key=lambda d: int(d.name.replace("dfl_train", "")),
        )

        print(f"\n{'='*62}")
        print(f"Network : {net}  |  scale={scale}  |  {len(size_dirs)} n_train sizes")
        if saa_dir:   print(f"  SAA dir   : {saa_dir}")
        if sddip_dir: print(f"  SDDiP dir : {sddip_dir}")
        print("="*62)

        for size_dir in size_dirs:
            n_train = size_dir.name

            # ── Aggregated level ──────────────────────────────────────────
            if not args.seeds_only:
                agg_json = size_dir / "evaluation.json"
                if agg_json.exists():
                    print(f"\n[AGG]  {net}/{n_train}")
                    rc = _run_compare(size_dir, real_net, scale, saa_dir, sddip_dir, args.repair_budget_fraction)
                    if rc == 0: total_ok   += 1
                    else:       total_fail += 1
                else:
                    print(f"\n[AGG]  {net}/{n_train}  — no evaluation.json, skipping"
                          " (run aggregate_multiseed.py first)")
                    total_skip += 1

            # ── Per-seed level ────────────────────────────────────────────
            if not args.aggregated_only:
                seed_dirs = sorted(
                    (d for d in size_dir.iterdir()
                     if d.is_dir() and d.name.startswith("seed_")),
                    key=lambda d: int(d.name.replace("seed_", "")),
                )
                for seed_dir in seed_dirs:
                    if not (seed_dir / "evaluation.json").exists():
                        print(f"\n[SEED] {net}/{n_train}/{seed_dir.name}"
                              "  — no evaluation.json, skipping")
                        total_skip += 1
                        continue
                    print(f"\n[SEED] {net}/{n_train}/{seed_dir.name}")
                    rc = _run_compare(seed_dir, real_net, scale, saa_dir, sddip_dir, args.repair_budget_fraction)
                    if rc == 0: total_ok   += 1
                    else:       total_fail += 1

    print(f"\n{'='*62}")
    print(f"Done.  OK={total_ok}  Failed={total_fail}  Skipped={total_skip}")
    print("="*62)


if __name__ == "__main__":
    main()
