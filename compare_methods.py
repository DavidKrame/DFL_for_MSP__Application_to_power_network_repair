"""
compare_methods.py
==================
Command-line tool that builds the cross-method comparison for one network
(plots, comparison_summary.json and a DFL convergence curve) from existing DFL,
SAA and SDDiP result directories.

Mean costs come from the stored evaluation and solution files. In an aggregated
multi-seed DFL directory, the DFL std is the std of the per-seed mean costs, not
of the test-scenario costs. The heatmap, repair-rate and stage-cost plots
simulate the methods on the first min(50, --n-eval) test scenarios, regenerated
from --seed and --eval-seed, so --network, --repair-base-scale,
--repair-budget-fraction and the seeds must match the runs being compared. DFL
is simulated only if --dfl-dir contains model.pt (an aggregated dfl_train<N>/
directory has none). SAA and SDDiP are simulated only if their solution file
holds the mean repair decisions, which are rounded to a fixed schedule.

A directory that is not passed is searched for under --output-dir: the
dfl_train* run with the lowest DFL mean cost, the SAA run with the largest
sample (SAA/ or SAA/sc*/) and the SDDiP run with the most iterations (SDDiP/ or
SDDiP/iter*/).

Usage
-----
  python compare_methods.py \\
      --network toy \\
      --dfl-dir   experiments/paper_unbudgeted/dfl_scaling_multiseed/toy/dfl_train5000 \\
      --saa-dir   experiments/paper_unbudgeted/toy/SAA \\
      --sddip-dir experiments/paper_unbudgeted/toy/SDDiP \\
      --repair-base-scale 100 --eval-seed 1000 --n-eval 500 \\
      --output-dir experiments/paper_unbudgeted/dfl_scaling_multiseed/toy

Output
------
  <output-dir>/comparison/
      cost_bars_with_bounds.png      mean cost +/- std per method, SDDiP LB/UB lines
      cost_share_pct.png             dispatch, shedding and repair shares of the cost
      cost_share_pct_normalized.png
      computation_time.png           (if any timing data was found)
      repair_heatmap_comparison.png
      repair_rate_by_stage.png
      stage_cost_profile.png
      comparison_summary.json
      dfl_convergence.png            (if <output-dir> holds two or more dfl_train*/ runs)
  repair_heatmap_<method>.png in each method's own directory: the DFL directory
  for DFL and the baselines, the SAA and SDDiP directories when their solution
  file contains the repair decisions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')

from network_data       import create_network
from model_parameters   import create_model_parameters
from scenario_generator import create_scenario_generator

from Pseudo_DFL_model   import RepairNetwork, load_repair_network
from Pseudo_DFL_training import PseudoDFLTrainer
from Pseudo_DFL_solver  import (PseudoDFLRollout, AlwaysRepairRollout,
                                 NoRepairRollout, NearSightedMPCRollout, _cost_stats)
from compare_visualization import create_comparison_folder
import config as _config
from config import (DEFAULT_EVAL_SEED, DEFAULT_N_EVAL, NN_HIDDEN_DIMS,
                    NEAR_SIGHTED_MPC_WINDOW, REPAIR_BUDGET_FRACTION)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _autodiscover_dirs(base_dir):
    """
    Find the best DFL run and the SAA/SDDiP result directories in base_dir.

    Returns
    -------
    dfl_dir : str or None    dfl_train* run with the lowest Pseudo_DFL mean_cost,
                             searched in base_dir, else in its parent
    saa_dir : str or None    SAA/ or SAA/sc*/ with the largest sample (sample
                             paths in tree mode, scenarios otherwise)
    sddip_dir : str or None  SDDiP/ or SDDiP/iter*/ with the most iterations
    selection_labels : dict  short label per method key for the cost-bar ticks,
                             e.g. 'n=5,000', '16,384 paths', '782 iter'
    """
    base = Path(base_dir)
    selection_labels = {}

    # ── DFL: scan all dfl_train* dirs, pick lowest Pseudo_DFL mean_cost ──
    dfl_dir = None
    best_cost = float('inf')
    candidates = sorted(base.glob('dfl_train*/evaluation.json'))
    if not candidates:
        # Also look one level up, for an output dir inside the network dir
        # (e.g. a dfl_train<N>/ run passed as --output-dir).
        candidates = sorted(base.parent.glob('dfl_train*/evaluation.json'))
    for eval_path in candidates:
        try:
            data = _load_json(eval_path)
            cost = data.get('Pseudo_DFL', {}).get('mean_cost', float('inf'))
            if cost < best_cost:
                best_cost = cost
                dfl_dir = str(eval_path.parent)
        except Exception:
            continue
    if dfl_dir:
        # Label from the dir name: dfl_train8000 gives 'n=8,000'
        stem = Path(dfl_dir).name
        n = stem.replace('dfl_train', '')
        selection_labels['Pseudo_DFL'] = f'n={int(n):,}' if n.isdigit() else stem
        print(f"  Auto-selected DFL dir  : {dfl_dir}  (mean_cost=${best_cost:,.0f})")

    # ── SAA ──────────────────────────────────────────────────────────────
    # Check SAA/ itself and SAA/sc*/; keep the largest sample.
    saa_dir = None
    saa_root = base / 'SAA'
    _saa_best_n = -1

    def _try_saa_dir(candidate):
        nonlocal saa_dir, _saa_best_n
        sol_path = candidate / 'solution.json'
        if not sol_path.exists():
            return
        try:
            sol = _load_json(sol_path)
            meta = sol.get('solver_metadata') or {}
            if meta.get('mode') == 'tree':
                # Tree mode: the size is the tree's B^T sample paths. The stored
                # n_scenarios is the --n-scenarios argument, which does not set the
                # tree size.
                n_scen = meta.get('n_sample_paths') or 0
                label = f'{n_scen:,} paths' if n_scen else ''
            else:
                n_scen = (sol.get('n_scenarios')
                          or sol.get('num_scenarios')
                          or meta.get('n_scenarios')
                          or 0)
                label = f'{n_scen:,} sc' if n_scen else ''
            if n_scen > _saa_best_n:
                _saa_best_n = n_scen
                saa_dir = str(candidate)
                selection_labels['SAA'] = label
        except Exception:
            pass

    if saa_root.is_dir():
        _try_saa_dir(saa_root)
        for sc_sub in sorted(saa_root.glob('sc*')):
            if sc_sub.is_dir():
                _try_saa_dir(sc_sub)

    if saa_dir:
        print(f"  Auto-selected SAA dir  : {saa_dir}  [{selection_labels.get('SAA', '')}]")

    # ── SDDiP ─────────────────────────────────────────────────────────────
    # Check SDDiP/ itself and SDDiP/iter*/; keep the run with the most iterations.
    sddip_dir = None
    sddip_root = base / 'SDDiP'
    _sddip_best_iters = -1

    def _try_sddip_dir(candidate):
        nonlocal sddip_dir, _sddip_best_iters
        for fname in ('sddip_solution.json', 'sddip_solution_interrupted.json'):
            sol_path = candidate / fname
            if sol_path.exists():
                try:
                    sol = _load_json(sol_path)
                    iters = sol.get('iterations') or 0
                    if iters > _sddip_best_iters:
                        _sddip_best_iters = iters
                        sddip_dir = str(candidate)
                        selection_labels['SDDiP'] = f'{iters} iter' if iters else ''
                except Exception:
                    pass
                return  # only the first file found in a candidate counts

    if sddip_root.is_dir():
        _try_sddip_dir(sddip_root)
        for iter_sub in sorted(sddip_root.glob('iter*')):
            if iter_sub.is_dir():
                _try_sddip_dir(iter_sub)

    if sddip_dir:
        print(f"  Auto-selected SDDiP dir: {sddip_dir}  [{selection_labels.get('SDDiP', '')}]")

    return dfl_dir, saa_dir, sddip_dir, selection_labels


def _build_results_dict(dfl_dir, saa_dir, sddip_dir):
    """
    Assemble results_dict ({method: cost statistics}) from the files on disk.

    Sources, in order:
      1. dfl_dir/evaluation.json: Pseudo_DFL, AlwaysRepair, NoRepair and any
         other method stored there (NearSightedMPC from --with-mpc or
         inject_mpc_baseline.py; SAA and SDDiP if the DFL run loaded them
         from --saa-dir / --sddip-dir). These entries are kept as they are.
      2. The out-of-sample test_costs in saa_dir/solution.json (written by
         main.py --evaluate), if SAA is not yet present.
      3. Per-scenario SDDiP costs from the first of simulation_results.json,
         sddip_results.json and results.json found in sddip_dir, if SDDiP is
         not yet present. SDDiP_main.py does not write these files, so SDDiP
         usually enters the comparison only through sddip_solution.json
         (bounds, mean repair decisions and solve time).
    """
    results = {}

    # 1. DFL evaluation
    eval_json = Path(dfl_dir) / 'evaluation.json'
    if not eval_json.exists():
        raise FileNotFoundError(
            f"No evaluation.json found in {dfl_dir}. "
            "Run Pseudo_DFL_main.py --mode evaluate first."
        )
    raw = _load_json(eval_json)
    # Keep the method entries only (skips the top-level 'timing' block)
    for label, val in raw.items():
        if isinstance(val, dict) and 'mean_cost' in val:
            results[label] = val

    # 2. SAA out-of-sample costs (if not already loaded)
    if 'SAA' not in results and saa_dir:
        saa_json = Path(saa_dir) / 'solution.json'
        if saa_json.exists():
            saa_sol = _load_json(saa_json)
            oos = saa_sol.get('out_of_sample_evaluation', {})
            if 'test_costs' in oos:
                results['SAA'] = _cost_stats(
                    np.array(oos['test_costs']), 'SAA'
                )
                print(f"  Loaded SAA costs from {saa_json}")
            else:
                print(f"  Warning: no out-of-sample costs in {saa_json}. "
                      "Re-run SAA with --evaluate to enable SAA comparison.")
        else:
            print(f"  Warning: {saa_json} not found — SAA excluded.")

    # 3. SDDiP per-scenario costs (if not already loaded); only the first
    #    existing file is read, whether or not it has a cost key
    if 'SDDiP' not in results and sddip_dir:
        for fname in ('simulation_results.json', 'sddip_results.json', 'results.json'):
            fpath = Path(sddip_dir) / fname
            if fpath.exists():
                data = _load_json(fpath)
                for key in ('costs', 'simulation_costs', 'total_costs'):
                    if key in data:
                        results['SDDiP'] = _cost_stats(
                            np.array(data[key]), 'SDDiP'
                        )
                        print(f"  Loaded SDDiP simulation costs from {fpath}")
                        break
                break

    return results


def _load_sddip_solution(sddip_dir):
    """Load sddip_solution.json, else sddip_solution_interrupted.json; None if neither exists."""
    if not sddip_dir:
        return None
    for fname in ('sddip_solution.json', 'sddip_solution_interrupted.json'):
        p = Path(sddip_dir) / fname
        if p.exists():
            print(f"  Loaded SDDiP solution from {p}")
            return _load_json(p)
    print(f"  Warning: no sddip_solution.json found in {sddip_dir}")
    return None


def _load_saa_solution(saa_dir):
    """Load saa_dir/solution.json; None if it does not exist."""
    if not saa_dir:
        return None
    p = Path(saa_dir) / 'solution.json'
    if p.exists():
        print(f"  Loaded SAA solution from {p}")
        return _load_json(p)
    print(f"  Warning: solution.json not found in {saa_dir}")
    return None


def _load_dfl_rollout(dfl_dir, network, params, lp_method, hidden_dims):
    """Wrap dfl_dir/model.pt in a PseudoDFLRollout; None if it is missing or fails to load."""
    model_path = Path(dfl_dir) / 'model.pt'
    if not model_path.exists():
        print(f"  Warning: model.pt not found in {dfl_dir} — DFL heatmap skipped.")
        return None
    n_comp = network.n_generators + network.n_lines
    in_dim = n_comp + 1   # component availabilities + stage fraction t/T
    try:
        model, eps = load_repair_network(
            model_path, in_dim, n_comp,
            hidden_dims=tuple(hidden_dims)
        )
        print(f"  Loaded DFL model from {model_path}")
        return PseudoDFLRollout(network, params, model,
                                epsilon=eps, lp_method=lp_method)
    except Exception as exc:
        print(f"  Warning: could not load DFL model ({exc}) — DFL heatmap skipped.")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cross-method comparison for one experiment / network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--network',    type=str, default='toy',
                        help='Network name: toy, case14, case30, case57, case118')
    parser.add_argument('--dfl-dir',    type=str, default=None,
                        help='DFL run directory (evaluation.json + model.pt). '
                             'If omitted, the best dfl_train* run in --output-dir '
                             'is selected automatically.')
    parser.add_argument('--saa-dir',    type=str, default=None,
                        help='SAA results directory (solution.json expected). '
                             'Auto-discovered from --output-dir/SAA/ if omitted.')
    parser.add_argument('--sddip-dir',  type=str, default=None,
                        help='SDDiP results directory (sddip_solution.json expected). '
                             'Auto-discovered from --output-dir/SDDiP/ if omitted.')
    parser.add_argument('--eval-seed',  type=int, default=DEFAULT_EVAL_SEED,
                        help='Seed used to generate test scenarios (must match '
                             'the DFL evaluation seed)')
    parser.add_argument('--n-eval',     type=int, default=DEFAULT_N_EVAL,
                        help='Number of test scenarios to generate; the heatmaps use the first 50')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Training seed (used to create ModelParameters)')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                        default=list(NN_HIDDEN_DIMS),
                        help='RepairNetwork hidden layer widths (must match trained model)')
    parser.add_argument('--lp-method',  type=str, default=_config.LP_METHOD,
                        help="per-stage DC-OPF backend: scipy linprog method or "
                             "'gurobi' (default follows solver_config / ED_OPF_SOLVER)")
    parser.add_argument('--mpc-window', type=int, default=NEAR_SIGHTED_MPC_WINDOW,
                        help='Near-Sighted MPC window for the repair heatmap '
                             '(only used if NearSightedMPC is present in results)')
    parser.add_argument('--repair-budget-fraction', type=float, default=REPAIR_BUDGET_FRACTION,
                        help='Model-agnostic budget fraction phi (B = phi * total '
                             'repair cost) used when building rollout objects for the '
                             'repair heatmaps; match the runs being compared')
    parser.add_argument('--repair-base-scale', type=float, default=1.0,
                        help='Uniform multiplier for both repair cost bases')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Parent experiment directory; comparison/ is created inside')

    args = parser.parse_args()

    # ── Auto-discover dirs when not explicitly provided ───────────────────
    user_dfl_dir = args.dfl_dir   # remember whether the user passed it explicitly
    selection_labels = {}
    if args.dfl_dir is None or args.saa_dir is None or args.sddip_dir is None:
        print("\nAuto-discovering method directories from:", args.output_dir)
        auto_dfl, auto_saa, auto_sddip, selection_labels = _autodiscover_dirs(
            args.output_dir)
        if args.dfl_dir is None:
            args.dfl_dir = auto_dfl
        if args.saa_dir is None:
            args.saa_dir = auto_saa
        if args.sddip_dir is None:
            args.sddip_dir = auto_sddip

    # With an explicit --dfl-dir, the auto-discovered label may describe a
    # different run; rebuild it from the given directory name.
    if user_dfl_dir is not None and args.dfl_dir is not None:
        stem = Path(args.dfl_dir).name
        n = stem.replace('dfl_train', '')
        selection_labels['Pseudo_DFL'] = f'n={int(n):,}' if n.isdigit() else stem

    if args.dfl_dir is None:
        raise ValueError(
            "No DFL directory found. Either pass --dfl-dir explicitly or "
            "ensure dfl_train*/ subdirectories exist in --output-dir.")

    print()
    print("=" * 70)
    print("CROSS-METHOD COMPARISON")
    print("=" * 70)
    print(f"  Network    : {args.network}")
    print(f"  DFL dir    : {args.dfl_dir}"
          + (f"  [{selection_labels.get('Pseudo_DFL', '')}]"
             if 'Pseudo_DFL' in selection_labels else ''))
    print(f"  SAA dir    : {args.saa_dir}"
          + (f"  [{selection_labels.get('SAA', '')}]"
             if 'SAA' in selection_labels else ''))
    print(f"  SDDiP dir  : {args.sddip_dir}"
          + (f"  [{selection_labels.get('SDDiP', '')}]"
             if 'SDDiP' in selection_labels else ''))
    print(f"  eval seed  : {args.eval_seed}   n_eval: {args.n_eval}")
    print(f"  output dir : {args.output_dir}/comparison/")
    print()

    # ── 1. Network + params ───────────────────────────────────────────────
    print("Step 1: Initialising network and parameters...")
    # Mean repair costs at scale s; 550 and 350 are the config.py bases at s = 1.
    _config.REPAIR_BASE_SCALE = args.repair_base_scale
    _config.GEN_REPAIR_BASE   = 550.0 * args.repair_base_scale
    _config.LINE_REPAIR_BASE  = 350.0 * args.repair_base_scale
    network = create_network(args.network)
    params  = create_model_parameters(
        network, seed=args.seed,
        gen_repair_base=_config.GEN_REPAIR_BASE,
        line_repair_base=_config.LINE_REPAIR_BASE,
        repair_budget_fraction=args.repair_budget_fraction,
    )

    # ── 2. Test scenarios (same seed as DFL evaluation) ───────────────────
    print(f"Step 2: Generating {args.n_eval} test scenarios (seed={args.eval_seed})...")
    gen_eval   = create_scenario_generator(params, seed=args.eval_seed)
    test_scen  = gen_eval.generate_scenarios(args.n_eval)

    # ── 3. Load cost results ──────────────────────────────────────────────
    print("Step 3: Loading evaluation results...")
    results_dict = _build_results_dict(args.dfl_dir, args.saa_dir, args.sddip_dir)
    print(f"  Methods in results_dict: {list(results_dict.keys())}")

    # ── 4. Load solution objects (SDDiP LB/UB, repair policies, solve times) ──
    print("Step 4: Loading solution objects...")
    sddip_solution = _load_sddip_solution(args.sddip_dir)
    saa_solution   = _load_saa_solution(args.saa_dir)

    # ── 5. Build rollout objects (for heatmaps) ───────────────────────────
    print("Step 5: Building rollout objects for repair heatmaps...")
    dfl_rollout = _load_dfl_rollout(
        args.dfl_dir, network, params,
        lp_method=args.lp_method,
        hidden_dims=args.hidden_dims
    )
    ar_rollout = AlwaysRepairRollout(network, params, lp_method=args.lp_method)
    nr_rollout = NoRepairRollout(network, params, lp_method=args.lp_method)

    rollouts_dict = {'AlwaysRepair': ar_rollout, 'NoRepair': nr_rollout}
    # Build the MPC rollout (for the repair heatmap) only if MPC is in the loaded
    # results. evaluation.json does not record the window, so --mpc-window has to
    # match the evaluated run.
    if 'NearSightedMPC' in results_dict:
        rollouts_dict['NearSightedMPC'] = NearSightedMPCRollout(
            network, params, window=args.mpc_window, lp_method=args.lp_method)
    if dfl_rollout is not None:
        rollouts_dict['Pseudo_DFL'] = dfl_rollout

    # ── 6. Timing (seconds) for computation_time.png ──────────────────────
    timing_dict = {}
    # DFL: data generation + training time from evaluation.json (seed averages
    # in an aggregated multi-seed dir). data_generation_s is the wall time of
    # that run's data stage, so it leaves out the label MIPs of the scenarios
    # loaded from the cache.
    dfl_eval_json = Path(args.dfl_dir) / 'evaluation.json'
    if dfl_eval_json.exists():
        raw = _load_json(dfl_eval_json)
        timing = raw.get('timing', {})
        data_gen_s = timing.get('data_generation_s')
        train_s    = timing.get('training_s')
        if data_gen_s and data_gen_s > 0:
            timing_dict['DFL data generation (s)'] = data_gen_s
        if train_s and train_s > 0:
            timing_dict['DFL training (s)'] = train_s
        # DFL rollout time per test scenario (forward passes, repair decisions
        # and stage OPF LPs)
        dfl_res = raw.get('Pseudo_DFL', {})
        infer_s = dfl_res.get('mean_time_per_scenario_s')
        if infer_s and infer_s > 0:
            timing_dict['DFL inference / scenario (s)'] = infer_s

    # Near-sighted MPC: rollout time per test scenario, lookahead search included
    # (present only when MPC was evaluated)
    mpc_res = results_dict.get('NearSightedMPC')
    if mpc_res:
        mpc_s = mpc_res.get('mean_time_per_scenario_s')
        if mpc_s and mpc_s > 0:
            timing_dict['MPC / scenario (s)'] = mpc_s

    # SAA: solve time from solution.json (top-level or nested in solver_metadata)
    if saa_solution is not None:
        saa_solve_s = saa_solution.get('solve_time')
        if not saa_solve_s:
            saa_solve_s = (saa_solution.get('solver_metadata') or {}).get('solve_time')
        if saa_solve_s and saa_solve_s > 0:
            timing_dict['SAA solve (s)'] = saa_solve_s

    # SDDiP: solve_time of the loaded solution file (model build, training and
    # policy simulation for a finished run)
    if sddip_solution is not None:
        sddip_solve_s = sddip_solution.get('solve_time')
        if sddip_solve_s and sddip_solve_s > 0:
            timing_dict['SDDiP training (s)'] = sddip_solve_s

    # ── 7. Generate comparison folder ────────────────────────────────────
    # Destination of each method's own repair_heatmap_<method>.png
    method_save_dirs = {
        'Pseudo_DFL':  args.dfl_dir,
        'AlwaysRepair': args.dfl_dir,
        'NoRepair':    args.dfl_dir,
    }
    if 'NearSightedMPC' in results_dict:
        method_save_dirs['NearSightedMPC'] = args.dfl_dir
    if args.saa_dir:
        method_save_dirs['SAA'] = args.saa_dir
    if args.sddip_dir:
        method_save_dirs['SDDiP'] = args.sddip_dir

    print("Step 7: Generating comparison folder + individual heatmaps...")
    create_comparison_folder(
        results_dict     = results_dict,
        rollouts_dict    = rollouts_dict,
        scenarios        = test_scen,
        network          = network,
        params           = params,
        out_dir          = args.output_dir,
        sddip_solution   = sddip_solution,
        saa_solution     = saa_solution,
        timing_dict      = timing_dict,
        method_save_dirs = method_save_dirs,
        selection_labels = selection_labels,
    )

    # ── 8. DFL convergence curve (mean cost vs n_train) ──────────────────
    print("Step 8: Generating DFL convergence curve...")
    _plot_dfl_convergence(args.output_dir)

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"  Output: {args.output_dir}/comparison/")
    print()


def _plot_dfl_convergence(base_dir):
    """Plot DFL mean cost against n_train over the dfl_train*/ runs in base_dir.

    Each point is the Pseudo_DFL mean_cost in dfl_train<N>/evaluation.json; for
    a multi-seed run that file holds the mean of the per-seed means. When a run
    has two or more seed_*/evaluation.json files, the error bar is the
    population std (ddof=0) of their DFL mean costs; otherwise the point has no
    error bar. The AlwaysRepair mean cost of the first run that has one is drawn
    as a reference line. Needs at least two runs; writes
    base_dir/comparison/dfl_convergence.png.
    """
    import matplotlib.pyplot as plt
    import numpy as _np

    base = Path(base_dir)
    points  = []   # (n_train, mean_cost, std_or_None)
    ar_cost = None

    for eval_json in sorted(base.glob('dfl_train*/evaluation.json')):
        try:
            d = _load_json(eval_json)
            n_train = (d.get('timing') or {}).get('n_train')
            if n_train is None:
                try:
                    n_train = int(eval_json.parent.name.replace('dfl_train', ''))
                except ValueError:
                    continue
            dfl_cost = (d.get('Pseudo_DFL') or {}).get('mean_cost')
            if dfl_cost is None:
                continue

            # Check for per-seed subdirs to compute error bar
            seed_means = []
            for seed_eval in sorted(eval_json.parent.glob('seed_*/evaluation.json')):
                try:
                    sd = _load_json(seed_eval)
                    sc = (sd.get('Pseudo_DFL') or {}).get('mean_cost')
                    if sc is not None:
                        seed_means.append(float(sc))
                except Exception:
                    continue
            std = float(_np.std(seed_means)) if len(seed_means) > 1 else None

            points.append((int(n_train), float(dfl_cost), std))
            if ar_cost is None:
                ar_cost = (d.get('AlwaysRepair') or {}).get('mean_cost')
        except Exception:
            continue

    if len(points) < 2:
        print("  Skipping convergence curve — fewer than 2 DFL runs found.")
        return

    points.sort(key=lambda x: x[0])
    n_trains = [p[0] for p in points]
    costs    = [p[1] for p in points]
    stds     = [p[2] for p in points]
    has_eb   = any(s is not None for s in stds)

    fig, ax = plt.subplots(figsize=(8, 5))

    if has_eb:
        yerr = [s if s is not None else 0.0 for s in stds]
        ax.errorbar(n_trains, costs, yerr=yerr, fmt='-o', linewidth=2,
                    color='steelblue', capsize=4, elinewidth=1.2, ecolor='black',
                    label='DFL mean cost ± std (across seeds)')
    else:
        ax.plot(n_trains, costs, marker='o', linewidth=2, color='steelblue',
                label='DFL mean cost')
        ax.scatter(n_trains, costs, s=60, color='steelblue', zorder=5)

    if ar_cost is not None:
        ax.axhline(ar_cost, color='tomato', linewidth=1.5,
                   linestyle='--', label=f'AlwaysRepair ({ar_cost:,.0f})')

    ax.set_xlabel('DFL training scenarios (n_train)', fontsize=12)
    ax.set_ylabel('Mean cost', fontsize=12)
    ax.set_title('DFL convergence vs training size', fontsize=13)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = base / 'comparison' / 'dfl_convergence.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Convergence curve saved → {out_path}  ({len(points)} points)")


if __name__ == "__main__":
    main()
