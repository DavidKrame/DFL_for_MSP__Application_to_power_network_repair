#!/usr/bin/env python3
"""
inject_mpc_baseline.py
======================
Add the Near-sighted MPC baseline (key 'NearSightedMPC') to existing
evaluation.json files without changing the stored results of the other methods.

For each target folder it:
  1. Rebuilds the network, model parameters and test scenarios of the original
     run. The network is --network. The repair-cost bases, the model and eval
     seeds (42 and 1000 if absent) and the budget fraction phi (no budget if
     absent) come from the folder's config_snapshot.json; a folder without a
     snapshot that holds the repair-cost bases is skipped. n_eval is the number
     of stored NoRepair costs. Everything else (T, tau, failure model, shedding
     cost) is taken from config.py.
  2. Re-evaluates NoRepair and AlwaysRepair and checks that they reproduce the
     stored mean costs (within --tol dollars, or 1e-6 relative if larger). A
     mismatch means the rebuilt test set or parameters differ; the folder is
     then skipped and nothing is written.
  3. Evaluates Near-sighted MPC with window --window on the same scenarios.
  4. Copies evaluation.json to evaluation.json.bak (only if no backup exists
     yet) and rewrites evaluation.json with a 'NearSightedMPC' entry; all
     other entries are kept unchanged.
A folder that already has the entry is skipped unless --force is given.

Folders that share the network, repair-cost bases, seeds, n_eval, window and
budget fraction get the same MPC result, so it is computed once per such group
and reused for the rest of the invocation. This cache lives in memory only; the
data_cache MPC file of Pseudo_DFL_main.py --with-mpc is neither read nor written.

In a dfl_scaling_multiseed tree, run it on the seed_<s>/ folders (the top-level
dfl_train<N>/ folders hold only the aggregated evaluation.json, without a
config_snapshot.json), then rerun aggregate_multiseed.py, which copies the
entry from the first seed. --network is the base network name (case14 for a
case14_scale55 folder); the repair-cost scale is in the snapshot. The
dfl_scaling.sh and dfl_multiseed.sh scripts in scripts/slurm/{unbudgeted,budgeted}/
pass --with-mpc, so their folders already hold the entry; replacing it (e.g.
with another --window) needs --force.

Usage:
  python3 inject_mpc_baseline.py --network toy --window 3 \\
      experiments/paper_unbudgeted/dfl_scaling_multiseed/toy/dfl_train*/seed_*
  python3 aggregate_multiseed.py --root experiments/paper_unbudgeted/dfl_scaling_multiseed
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from network_data       import create_network
from model_parameters   import create_model_parameters
from scenario_generator import create_scenario_generator
from Pseudo_DFL_solver  import (NoRepairRollout, AlwaysRepairRollout,
                                NearSightedMPCRollout)
import solver_config


def _load(path):
    with open(path) as f:
        return json.load(f)


def _close(a, b, tol):
    """True if |a - b| <= max(tol, 1e-6 * |b|); False if either value is None."""
    if a is None or b is None:
        return False
    return abs(a - b) <= max(tol, 1e-6 * abs(b))


def process_folder(folder, network, window, lp_method, tol, cache, force=False):
    """Add NearSightedMPC to folder/evaluation.json (steps 1-4 of the module docstring).

    cache maps the rebuild key to (MPC result, (NoRepair mean, AlwaysRepair mean))
    and is shared across the folders of one invocation.
    """
    folder = Path(folder)
    eval_path = folder / 'evaluation.json'
    cfg_path  = folder / 'config_snapshot.json'

    if not eval_path.exists():
        print(f"  SKIP {folder}: no evaluation.json")
        return
    ev = _load(eval_path)
    if 'NearSightedMPC' in ev and not force:
        print(f"  SKIP {folder}: already has NearSightedMPC (use --force to overwrite)")
        return

    nr_stored = (ev.get('NoRepair')     or {}).get('mean_cost')
    ar_stored = (ev.get('AlwaysRepair') or {}).get('mean_cost')
    n_eval    = len((ev.get('NoRepair') or {}).get('costs') or [])
    if n_eval == 0:
        print(f"  SKIP {folder}: cannot determine n_eval from NoRepair costs")
        return

    cfg = _load(cfg_path) if cfg_path.exists() else {}
    gen_base   = cfg.get('GEN_REPAIR_BASE')
    line_base  = cfg.get('LINE_REPAIR_BASE')
    model_seed = cfg.get('DEFAULT_TRAIN_SEED', 42)
    eval_seed  = cfg.get('DEFAULT_EVAL_SEED', 1000)
    budget_frac = cfg.get('REPAIR_BUDGET_FRACTION')
    if gen_base is None or line_base is None:
        print(f"  SKIP {folder}: config_snapshot.json missing repair-cost bases")
        return

    key = (network, gen_base, line_base, model_seed, eval_seed, n_eval, window,
           budget_frac)
    if key in cache:
        mpc_res, (nr_re, ar_re) = cache[key]
    else:
        net = create_network(network)
        params = create_model_parameters(
            net, seed=model_seed,
            gen_repair_base=gen_base, line_repair_base=line_base,
            repair_budget_fraction=budget_frac)
        scen = create_scenario_generator(
            params, seed=eval_seed).generate_scenarios(n_eval)
        print(f"  [compute] network={network} scale_bases=({gen_base},{line_base}) "
              f"budget_frac={budget_frac} model_seed={model_seed} "
              f"eval_seed={eval_seed} n_eval={n_eval} W={window}")
        nr_re = NoRepairRollout(net, params).evaluate(scen, verbose=False)['mean_cost']
        ar_re = AlwaysRepairRollout(net, params).evaluate(scen, verbose=False)['mean_cost']
        mpc_res = NearSightedMPCRollout(
            net, params, window=window, lp_method=lp_method).evaluate(scen, verbose=True)
        cache[key] = (mpc_res, (nr_re, ar_re))

    if not _close(nr_re, nr_stored, tol) or not _close(ar_re, ar_stored, tol):
        print(f"  ABORT {folder}: reproduced baselines do not match stored "
              f"(NoRepair re={nr_re:,.2f} vs {nr_stored}; "
              f"AlwaysRepair re={ar_re:,.2f} vs {ar_stored}). "
              f"Test set mismatch — NOT injecting.")
        return

    bak = folder / 'evaluation.json.bak'
    if not bak.exists():
        shutil.copy2(eval_path, bak)

    ev['NearSightedMPC'] = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in mpc_res.items()
    }
    with open(eval_path, 'w') as f:
        json.dump(ev, f, indent=2)
    print(f"  OK {folder}: injected NearSightedMPC mean={mpc_res['mean_cost']:,.1f} "
          f"(W={window}); baselines verified, backup -> {bak.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', help='evaluation.json folders to update')
    ap.add_argument('--network', default='toy', help='network name (default: toy)')
    ap.add_argument('--window', type=int, default=3, help='MPC window W (default: 3)')
    ap.add_argument('--lp-method', default=solver_config.LP_SOLVER,
                    help="per-stage DC-OPF backend: scipy method or 'gurobi' "
                         "(default follows solver_config / ED_OPF_SOLVER)")
    ap.add_argument('--tol', type=float, default=1.0,
                    help='abs tolerance ($) for matching stored baselines (default: 1.0)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing NearSightedMPC key (e.g. after a change of '
                         'the repair budget); the original evaluation.json is '
                         'preserved in the .bak the first time it is touched')
    args = ap.parse_args()

    cache = {}
    print(f"Injecting NearSightedMPC (W={args.window}) into {len(args.folders)} folder(s):")
    for folder in args.folders:
        process_folder(folder, args.network, args.window, args.lp_method,
                       args.tol, cache, force=args.force)
    print("Done.")


if __name__ == '__main__':
    main()
