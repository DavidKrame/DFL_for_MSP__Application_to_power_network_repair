"""
Pseudo_DFL Main Script
======================
Runs the DFL pipeline for the multistage DC-OPF repair problem with stochastic
component failures. RepairNetwork maps the state (component availabilities and
the stage fraction t/T) to one repair value per component. It is trained with a
Fenchel-Young (FY) loss on the repair decisions of single-scenario
(perfect-information) MIPs. The deployed policy repairs every failed component
with a positive value; under a per-stage repair budget it solves a 0/1 knapsack
over these values instead.

Modes
-----
  data      Solve the single-scenario MIPs and store the training data.
  train     Train RepairNetwork with the FY loss and save model.pt
            (or load the model given by --load-model).
  evaluate  Roll out DFL, AlwaysRepair, NoRepair and, with --with-mpc,
            Near-sighted MPC on fresh test scenarios; add SAA/SDDiP costs
            from --saa-dir/--sddip-dir; write evaluation.json.
  compare   Make the plots from the evaluation results (evaluation.json when
            run on its own) and the trained model.
  all       Run all stages in order (default).

Usage examples
--------------
  # Full pipeline (200 training scenarios, 100 test scenarios):
  python Pseudo_DFL_main.py --mode all --n-train 200 --n-eval 100

  # Generate data only (useful before training on a GPU machine):
  python Pseudo_DFL_main.py --mode data --n-train 300 --solver glpk

  # Train on the cached training data (missing scenarios are solved first):
  python Pseudo_DFL_main.py --mode train --output-dir <run_dir>

  # Evaluate the trained model with SAA costs included, then make the plots:
  python Pseudo_DFL_main.py --mode evaluate --output-dir <run_dir> --saa-dir <saa_dir>
  python Pseudo_DFL_main.py --mode compare  --output-dir <run_dir>

  # Quick smoke test (10 training scenarios, 5 test scenarios, 10 epochs):
  python Pseudo_DFL_main.py --test

Results are written to --output-dir (default: Experiment_<DD>_<Mon>_<YYYY>__<HH>:<MM>/):
  config_snapshot.json        config.py values, with this run's repair costs, budget and
                              train/eval seeds
  model.pt                    trained PyTorch model checkpoint
  evaluation.json             per-scenario costs and summary statistics per method, plus timing
  training_curves.png         FY loss (train/val) and validation repair ratio per epoch
  cost_distribution.png       violin/box comparison
  cost_comparison_bar.png     mean ± std bar chart
  repair_heatmap.png          DFL availability and repairs on the first test scenario
  availability_trajectory.png availability under DFL and NoRepair on the same scenario
  repair_decisions.png        DFL repair rate per component and stage (first 50 test scenarios)
  repair_heatmap_avg_50.png   DFL availability and repair rates (first 50 test scenarios)
  repair_heatmap_<method>.png the same averaged heatmap for each rolled-out method
  training_data.pkl           training data, written here only when no cache path is set

The training data normally lives in a cache shared across runs:
data_cache[_scale<s>]/<network>/seed<seed>/training_data[_budgetfrac<phi>].pkl.
With several --init-seeds, each seed runs in <output-dir>/seed_<s>/ and
<output-dir>/evaluation.json holds the mean and std of the DFL mean cost across
the seeds.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Project imports
import config as _config
from config import (
    NN_HIDDEN_DIMS, NN_DROPOUT, FY_EPSILON,
    TRAIN_N_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_VAL_SPLIT, TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_WEIGHT_DECAY, TRAIN_SCHED_PATIENCE, TRAIN_SCHED_FACTOR, TRAIN_GRAD_CLIP,
    T, TAU,
    LOAD_SHEDDING_COST, GEN_REPAIR_BASE, LINE_REPAIR_BASE, REPAIR_BASE_SCALE,
    FAILURE_K, FAILURE_T_STAR, FAILURE_CLASS_PROB,
    FAILURE_P_MAX_STD_LOW, FAILURE_P_MAX_STD_HIGH,
    FAILURE_P_MAX_ELV_LOW, FAILURE_P_MAX_ELV_HIGH,
    DATA_SOLVER_NAME, LP_METHOD, DATA_MIP_REL_GAP, DATA_THREADS_PER_WORKER,
    NEAR_SIGHTED_MPC_WINDOW, REPAIR_BUDGET_FRACTION,
    DEFAULT_TRAIN_SEED, DEFAULT_INIT_SEED, DEFAULT_EVAL_SEED, DEFAULT_N_EVAL,
    save_snapshot,
)
import solver_config
from network_data        import create_network
from model_parameters    import create_model_parameters
from scenario_generator  import create_scenario_generator

from Pseudo_DFL_data        import (generate_training_data, save_training_data,
                              load_training_data, _subsample_dataset)
from Pseudo_DFL_model       import RepairNetwork, load_repair_network
from Pseudo_DFL_training    import PseudoDFLTrainer
from Pseudo_DFL_solver      import (PseudoDFLRollout, AlwaysRepairRollout, NoRepairRollout,
                              NearSightedMPCRollout, evaluate_mpc_cached,
                              print_evaluation_summary, _cost_stats)
from Pseudo_DFL_visualization import (create_pseudo_dfl_report, load_sddip_costs,
                                plot_cost_distribution,
                                plot_cost_comparison_bar,
                                _save_or_show)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Pseudo_DFL for multistage DC-OPF repair',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode
    p.add_argument('--mode', choices=['data', 'train', 'evaluate',
                                      'compare', 'all'],
                   default='all',
                   help='Pipeline stage to run')
    p.add_argument('--test', action='store_true',
                   help='Quick smoke test (small n_train/n_eval, few epochs)')

    # Data generation
    p.add_argument('--n-train', type=int, default=200,
                   help='Number of training scenarios (single-scenario MIPs)')
    p.add_argument('--solver', type=str, default=DATA_SOLVER_NAME,
                   choices=sorted(solver_config.MIP_SOLVERS),
                   help='MIP solver for training data generation (Layer A). '
                        'Default follows solver_config / ED_OPF_SOLVER.')
    p.add_argument('--seed', type=int, default=DEFAULT_TRAIN_SEED)
    p.add_argument('--init-seeds', type=int, nargs='+', default=[DEFAULT_INIT_SEED],
                   help='Seed(s) for network weight initialisation. '
                        'One value → single run (default: DEFAULT_INIT_SEED from config). '
                        'Multiple values → one run per seed, results aggregated into output-dir.')

    # Training
    p.add_argument('--n-epochs', type=int, default=None,
                   help='Training epochs (default: scales linearly with n_train, '
                        f'anchored at {TRAIN_N_EPOCHS} epochs for n_train=2000)')
    p.add_argument('--batch-size', type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=TRAIN_LR)
    p.add_argument('--epsilon', type=float, default=FY_EPSILON,
                   help='FY perturbation parameter ε')
    p.add_argument('--hidden-dims', type=int, nargs='+', default=list(NN_HIDDEN_DIMS),
                   help='Hidden layer widths for RepairNetwork')
    p.add_argument('--early-stop-patience', type=int, default=TRAIN_EARLY_STOP_PATIENCE,
                   help='Stop training if val loss does not improve for this many epochs')
    p.add_argument('--weight-decay', type=float, default=TRAIN_WEIGHT_DECAY)
    p.add_argument('--sched-patience', type=int, default=TRAIN_SCHED_PATIENCE,
                   help='ReduceLROnPlateau patience (epochs)')
    p.add_argument('--sched-factor', type=float, default=TRAIN_SCHED_FACTOR,
                   help='ReduceLROnPlateau multiplicative factor')
    p.add_argument('--grad-clip', type=float, default=TRAIN_GRAD_CLIP,
                   help='Gradient clipping max norm')
    p.add_argument('--dropout', type=float, default=NN_DROPOUT,
                   help='Dropout rate for RepairNetwork')
    p.add_argument('--val-split', type=float, default=TRAIN_VAL_SPLIT,
                   help='Fraction of training data held out for validation')

    # Problem structure
    p.add_argument('--n-stages', type=int, default=T,
                   help='Total number of stages T')
    p.add_argument('--tau', type=int, default=TAU,
                   help='Repair pipeline delay (stages)')
    p.add_argument('--load-shedding-cost', type=float, default=LOAD_SHEDDING_COST,
                   help='Load shedding penalty ($/MWh)')
    p.add_argument('--repair-base-scale', type=float, default=1.0,
                   help='Uniform multiplier for both repair cost bases '
                        '(overrides --gen-repair-base / --line-repair-base when != 1)')
    p.add_argument('--gen-repair-base', type=float, default=GEN_REPAIR_BASE,
                   help='Mean generator repair cost ($); ignored when --repair-base-scale != 1')
    p.add_argument('--line-repair-base', type=float, default=LINE_REPAIR_BASE,
                   help='Mean line repair cost ($); ignored when --repair-base-scale != 1')
    p.add_argument('--failure-k', type=float, default=FAILURE_K,
                   help='Failure model global shift K')
    p.add_argument('--failure-t-star', type=int, default=FAILURE_T_STAR,
                   help='Stage at which failure probability peaks')
    p.add_argument('--failure-class-prob', type=float, default=FAILURE_CLASS_PROB,
                   help='P(Standard vulnerability class)')
    p.add_argument('--failure-p-max-std-low', type=float, default=FAILURE_P_MAX_STD_LOW)
    p.add_argument('--failure-p-max-std-high', type=float, default=FAILURE_P_MAX_STD_HIGH)
    p.add_argument('--failure-p-max-elv-low', type=float, default=FAILURE_P_MAX_ELV_LOW)
    p.add_argument('--failure-p-max-elv-high', type=float, default=FAILURE_P_MAX_ELV_HIGH)

    # Data generation: MIP solver settings
    p.add_argument('--mip-rel-gap', type=float, default=DATA_MIP_REL_GAP,
                   help='MIP optimality tolerance for training data generation')
    p.add_argument('--threads-per-worker', type=int, default=DATA_THREADS_PER_WORKER,
                   help='HiGHS threads per parallel worker')

    # Evaluation
    p.add_argument('--n-eval', type=int, default=None,
                   help='Number of test scenarios for evaluation '
                        '(default: max(5, n_train // 3), about a third of --n-train)')
    p.add_argument('--eval-seed', type=int, default=DEFAULT_EVAL_SEED,
                   help='Seed for test scenarios (different from training)')
    p.add_argument('--lp-method', type=str, default=LP_METHOD,
                   help='Per-stage DC-OPF backend (Layer C): a scipy linprog '
                        "method ('highs', 'highs-ds', ...) or 'gurobi' (gurobipy). "
                        'Default follows solver_config / ED_OPF_SOLVER.')
    p.add_argument('--with-mpc', action='store_true',
                   help='Also evaluate the Near-Sighted MPC baseline. Off by '
                        'default. When set it is computed once and cached, then '
                        'reused across every run with the same '
                        'network/scale/seed/eval/window/budget. The compare stage '
                        'still reruns it on up to 50 test scenarios for its heatmap.')
    p.add_argument('--mpc-window', type=int, default=NEAR_SIGHTED_MPC_WINDOW,
                   help='Near-Sighted MPC lookahead window W (requires '
                        'TAU+1 <= W <= T). Only used with --with-mpc.')
    p.add_argument('--repair-budget-fraction', type=float, default=REPAIR_BUDGET_FRACTION,
                   help='Per-stage repair budget as a model-agnostic fraction phi of '
                        'the network total repair cost: B = phi * sum(repair costs). '
                        'Same phi = same relative tightness on any network. '
                        'Omit the flag for no budget.')

    # I/O
    p.add_argument('--output-dir', type=str, default='auto',
                   help='Output directory (default: Experiment_<DD>_<Mon>_<YYYY>__<HH>:<MM> '
                        'in the working directory)')
    p.add_argument('--load-data',  type=str, default=None,
                   help='Path to pre-generated training_data.pkl (ignored when a cache path '
                        'is set, which happens by default)')
    p.add_argument('--cache-path', type=str, default=None,
                   help='Global cache pkl for incremental scenario reuse; '
                        'extends automatically when n_train exceeds cached count '
                        '(preferred over --load-data)')
    p.add_argument('--load-model', type=str, default=None,
                   help='Path to pre-trained model.pt')
    p.add_argument('--sddip-dir',  type=str, default=None,
                   help='SDDiP results directory for comparison')
    p.add_argument('--saa-dir',    type=str, default=None,
                   help='SAA results directory; loads out-of-sample evaluation costs '
                        'for comparison (requires SAA run with --evaluate flag)')
    p.add_argument('--network', type=str, default='toy',
                   help='Network to use: "toy" (built-in 5-bus) or an IEEE case '
                        'name such as case14, case30, case118 (requires pandapower).')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_data(args, network, params, out_dir):
    """Generate the training data (only scenarios missing from the cache) and save it."""
    print("\n" + "="*70)
    print("STAGE: DATA GENERATION")
    print("="*70)
    t0 = time.time()

    cache_path = getattr(args, 'cache_path', None)
    dataset, _ = generate_training_data(
        network, params,
        n_scenarios=args.n_train,
        solver_name=args.solver,
        seed=args.seed,
        verbose=True,
        cache_path=cache_path,
        mip_rel_gap=args.mip_rel_gap,
        threads_per_worker=args.threads_per_worker,
    )

    data_path = out_dir / 'training_data.pkl'
    # With a cache path, generate_training_data has already saved the merged
    # dataset to the cache, so no copy is written to out_dir.
    if not cache_path:
        save_training_data(dataset, data_path)
    elapsed = time.time() - t0
    print(f"  Data generation time: {elapsed:.1f}s")
    return dataset, data_path, elapsed


def stage_train(args, network, params, dataset, out_dir):
    """Train RepairNetwork and save model."""
    print("\n" + "="*70)
    print("STAGE: TRAINING")
    print("="*70)
    t0 = time.time()

    trainer = PseudoDFLTrainer(
        network, params,
        epsilon=args.epsilon,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        seed=args.seed,
        init_seed=args.init_seed,
    )

    trainer.train(
        dataset,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
        early_stop_patience=args.early_stop_patience,
        val_split=args.val_split,
        weight_decay=args.weight_decay,
        sched_patience=args.sched_patience,
        sched_factor=args.sched_factor,
        grad_clip=args.grad_clip,
    )

    model_path = out_dir / 'model.pt'
    trainer.save(model_path)
    trainer.print_summary()
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    return trainer, model_path, elapsed


def stage_evaluate(args, network, params, model, trainer, out_dir,
                   data_gen_time=0.0, train_time=0.0):
    """
    Evaluate DFL and the baselines on fresh test scenarios; write evaluation.json.

    Returns (results_dict, test_scen, dfl_rollout, ar_rollout, nr_rollout,
    mpc_rollout); mpc_rollout is None without --with-mpc.
    """
    print("\n" + "="*70)
    print("STAGE: EVALUATION")
    print("="*70)
    t0 = time.time()

    gen_eval  = create_scenario_generator(params, seed=args.eval_seed)
    test_scen = gen_eval.generate_scenarios(args.n_eval)

    print(f"  Generated {args.n_eval} test scenarios (seed={args.eval_seed})")

    # DFL policy
    dfl_rollout = PseudoDFLRollout(network, params, model,
                             epsilon=args.epsilon,
                             lp_method=args.lp_method)
    print("\n  Pseudo_DFL policy:")
    dfl_res = dfl_rollout.evaluate(test_scen, verbose=True)
    print_evaluation_summary(dfl_res)

    # AlwaysRepair baseline
    ar_rollout = AlwaysRepairRollout(network, params, lp_method=args.lp_method)
    print("\n  Always-repair baseline:")
    ar_res = ar_rollout.evaluate(test_scen, verbose=True)
    print_evaluation_summary(ar_res)

    # NoRepair baseline
    nr_rollout = NoRepairRollout(network, params, lp_method=args.lp_method)
    print("\n  No-repair baseline:")
    nr_res = nr_rollout.evaluate(test_scen, verbose=True)
    print_evaluation_summary(nr_res)

    results_dict = {
        'Pseudo_DFL':  dfl_res,
        'AlwaysRepair': ar_res,
        'NoRepair':     nr_res,
    }

    # Near-sighted MPC baseline (--with-mpc). Its result does not depend on
    # n_train, so it is cached on disk and reused by every run with the same
    # signature (network, repair costs and budget, seeds, n_eval, window, T, tau
    # and the schedule-cap settings; see evaluate_mpc_cached) instead of being
    # recomputed for each training size like AlwaysRepair and NoRepair.
    mpc_rollout = None
    if getattr(args, 'with_mpc', False):
        cache_root = Path('data_cache' + _config.experiment_suffix())
        _mpc_btag  = ('' if args.repair_budget_fraction is None
                      else f'_budgetfrac{args.repair_budget_fraction}')
        mpc_cache  = (cache_root / args.network / f'seed{args.seed}'
                      / f'mpc_W{args.mpc_window}{_mpc_btag}.json')
        print("\n  Near-sighted MPC baseline:")
        mpc_res = evaluate_mpc_cached(
            network, params, test_scen,
            window=args.mpc_window, cache_path=mpc_cache,
            eval_seed=args.eval_seed, network_name=args.network,
            lp_method=args.lp_method, verbose=True)
        print_evaluation_summary(mpc_res)
        results_dict['NearSightedMPC'] = mpc_res
        # Rollout object for the repair heatmap in stage_compare only, which
        # re-runs MPC on up to 50 test scenarios; the MPC costs above come from
        # evaluate_mpc_cached.
        mpc_rollout = NearSightedMPCRollout(network, params,
                                            window=args.mpc_window,
                                            lp_method=args.lp_method)

    # SDDiP per-scenario costs: load_sddip_costs reads simulation_results.json,
    # results.json or sddip_results.json in --sddip-dir.
    if args.sddip_dir:
        sddip_costs = load_sddip_costs(args.sddip_dir)
        if sddip_costs is not None:
            sddip_res = _cost_stats(sddip_costs, 'SDDiP')
            results_dict['SDDiP'] = sddip_res
            print("\n  SDDiP (loaded from simulation):")
            print_evaluation_summary(sddip_res)
        else:
            print(f"\n  Warning: no simulation costs found in {args.sddip_dir}")

    # SAA out-of-sample costs from <saa_dir>/solution.json (main.py --evaluate).
    if args.saa_dir:
        saa_json = Path(args.saa_dir) / 'solution.json'
        if saa_json.exists():
            with open(saa_json) as f:
                saa_sol = json.load(f)
            oos = saa_sol.get('out_of_sample_evaluation')
            if oos and 'test_costs' in oos:
                saa_res = _cost_stats(np.array(oos['test_costs']), 'SAA')
                results_dict['SAA'] = saa_res
                print("\n  SAA (loaded from out-of-sample evaluation):")
                print_evaluation_summary(saa_res)
            else:
                print(f"\n  Warning: no out-of-sample evaluation in {saa_json}")
                print("           Re-run SAA with --evaluate to enable SAA comparison.")
        else:
            print(f"\n  Warning: solution.json not found in {args.saa_dir}")

    eval_time = time.time() - t0
    timing = {
        'data_generation_s':        round(data_gen_time, 2),
        'training_s':               round(train_time, 2),
        'evaluation_s':             round(eval_time, 2),
        'n_train':                  args.n_train,
        'n_eval':                   args.n_eval,
    }

    # Save evaluation JSON
    eval_path = out_dir / 'evaluation.json'
    _save_eval_json(results_dict, eval_path, timing=timing)
    print(f"\n  Evaluation results saved → {eval_path}")
    print(f"  Evaluation time: {eval_time:.1f}s")

    return results_dict, test_scen, dfl_rollout, ar_rollout, nr_rollout, mpc_rollout


def stage_compare(args, results_dict, trainer, dfl_rollout, ar_rollout, nr_rollout,
                   test_scen, network, params, out_dir, mpc_rollout=None):
    """Write the DFL report plots and one averaged repair heatmap per method to out_dir."""
    from compare_visualization import plot_single_repair_heatmap, _avg_heatmap_data

    print("\n" + "="*70)
    print("STAGE: VISUALISATION")
    print("="*70)

    create_pseudo_dfl_report(
        trainer      = trainer,
        results_dict = results_dict,
        dfl_rollout  = dfl_rollout,
        one_scenario = test_scen[0],
        nr_rollout   = nr_rollout,
        scenarios    = test_scen,
        network      = network,
        params       = params,
        output_dir   = str(out_dir)
    )

    # repair_heatmap_<method>.png: availability and repair rates of each method,
    # averaged over the first n_sc test scenarios.
    n_sc = min(50, len(test_scen))
    for label, rollout_obj in [('Pseudo_DFL', dfl_rollout),
                                ('AlwaysRepair', ar_rollout),
                                ('NoRepair', nr_rollout),
                                ('NearSightedMPC', mpc_rollout)]:
        if rollout_obj is None:
            continue
        try:
            hm = _avg_heatmap_data(rollout_obj, test_scen, n_eval=n_sc)
            plot_single_repair_heatmap(
                label, hm, network, params,
                n_scenarios=n_sc,
                save_path=out_dir / f'repair_heatmap_{label.lower()}.png'
            )
        except Exception as exc:
            print(f"  Warning: could not save individual heatmap for {label}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Quick smoke test: 10 training scenarios, 10 epochs, all stages.
    if args.test:
        args.n_train  = 10
        args.n_epochs = 10
        args.mode     = 'all'
        print("=" * 70)
        print("Pseudo_DFL QUICK SMOKE TEST")
        print("=" * 70)
    else:
        # Epochs scale linearly with training set size, anchored at
        # TRAIN_N_EPOCHS for n_train=2000. Explicit --n-epochs takes priority.
        if args.n_epochs is None:
            args.n_epochs = max(1, int(TRAIN_N_EPOCHS * args.n_train / 2000))

    # The test set defaults to n_train // 3 scenarios (at least 5), so the
    # evaluation grows with the amount of training data.
    if args.n_eval is None:
        args.n_eval = max(5, args.n_train // 3)

    # A repair-cost scale s != 1 sets the mean repair costs to 550*s (generators)
    # and 350*s (lines), overriding --gen-repair-base / --line-repair-base.
    if args.repair_base_scale != 1.0:
        args.gen_repair_base  = 550.0 * args.repair_base_scale
        args.line_repair_base = 350.0 * args.repair_base_scale

    # Write these CLI values into config so that save_snapshot records them and
    # experiment_suffix() sees this run's scale. This must come before the
    # cache paths are built, so that each scale gets its own cache. Other CLI
    # overrides (e.g. --tau, --epsilon) are not written to config, so the
    # snapshot shows the config defaults for them.
    _config.REPAIR_BASE_SCALE  = args.repair_base_scale
    _config.GEN_REPAIR_BASE    = args.gen_repair_base
    _config.LINE_REPAIR_BASE   = args.line_repair_base
    _config.DEFAULT_TRAIN_SEED = args.seed
    _config.DEFAULT_EVAL_SEED  = args.eval_seed
    _config.REPAIR_BUDGET_FRACTION  = args.repair_budget_fraction # recorded in config_snapshot

    # Default training-data cache, one file per repair-cost scale s, network,
    # seed and budget:
    #   s = 1  → data_cache/<net>/seed<seed>/training_data<tag>.pkl
    #   s ≠ 1  → data_cache_scale<s>/<net>/seed<seed>/training_data<tag>.pkl
    # The file is created on first use. No default is set when --cache-path is
    # given, or when --load-shedding-cost differs from LOAD_SHEDDING_COST, since
    # the path does not encode the shedding cost. The path does not encode
    # --n-stages, --tau, the failure-model flags or, at s = 1, --gen-repair-base
    # and --line-repair-base either, and only the seed is checked on load, so a
    # cache built under other values of these flags is not detected.
    # --load-data is read only when no cache path is set.
    if not args.cache_path and args.load_shedding_cost == LOAD_SHEDDING_COST:
        cache_root = Path('data_cache' + _config.experiment_suffix())
        # A repair budget changes the optimal labels, so budgeted training data
        # goes to a separate file (<tag> = _budgetfrac<phi>; empty without a budget).
        tag = '' if args.repair_budget_fraction is None \
            else f'_budgetfrac{args.repair_budget_fraction}'
        auto_cache = cache_root / args.network / f'seed{args.seed}' / f'training_data{tag}.pkl'
        args.cache_path = str(auto_cache)
        print(f"  Auto cache       : {auto_cache}"
              + (" (will be created)" if not auto_cache.exists() else ""))

    if args.output_dir == 'auto':
        from datetime import datetime
        args.output_dir = datetime.now().strftime("Experiment_%d_%b_%Y__%H:%M")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_snapshot(out_dir)

    print()
    print("=" * 70)
    print("PSEUDO-DFL — DC-OPF REPAIR PROBLEM")
    print("=" * 70)
    print(f"  Mode        : {args.mode}")
    print(f"  Output dir  : {out_dir}")
    print(f"  n_train     : {args.n_train}")
    print(f"  n_eval      : {args.n_eval}")
    print(f"  n_epochs    : {args.n_epochs}")
    print(f"  epsilon     : {args.epsilon}")
    print(f"  hidden_dims : {args.hidden_dims}")
    print()

    # Common setup
    network = create_network(args.network)
    params  = create_model_parameters(
        network, seed=args.seed,
        t=args.n_stages, tau=args.tau,
        load_shedding_cost=args.load_shedding_cost,
        gen_repair_base=args.gen_repair_base,
        line_repair_base=args.line_repair_base,
        failure_k=args.failure_k,
        failure_t_star=args.failure_t_star,
        failure_class_prob=args.failure_class_prob,
        failure_p_max_std_low=args.failure_p_max_std_low,
        failure_p_max_std_high=args.failure_p_max_std_high,
        failure_p_max_elv_low=args.failure_p_max_elv_low,
        failure_p_max_elv_high=args.failure_p_max_elv_high,
        repair_budget_fraction=args.repair_budget_fraction,
    )
    if params.repair_budget_per_stage is not None:
        print(f"  Repair budget    : ${params.repair_budget_per_stage:,.0f} / stage"
              f"  (phi={args.repair_budget_fraction} of fleet repair cost)")

    dataset   = None
    trainer   = None
    model     = None
    test_scen = None
    results_dict  = None
    dfl_rollout   = None
    ar_rollout    = None
    nr_rollout    = None
    data_gen_time = 0.0
    train_time    = 0.0

    # Data stage
    if args.mode in ('data', 'all'):
        if args.cache_path:
            # Global cache: only missing scenarios are generated and merged back.
            dataset, _, data_gen_time = stage_data(args, network, params, out_dir)
        elif args.load_data and Path(args.load_data).exists():
            print(f"  Loading existing training data from {args.load_data}")
            dataset = load_training_data(args.load_data)
            if dataset['n_scenarios'] > args.n_train:
                print(f"  Subsampling dataset: {dataset['n_scenarios']} → {args.n_train} scenarios")
                dataset = _subsample_dataset(dataset, args.n_train)
            elif dataset['n_scenarios'] < args.n_train:
                print(f"  Warning: loaded dataset has {dataset['n_scenarios']} scenarios, "
                      f"fewer than --n-train={args.n_train}. Training on all available.")
        else:
            dataset, _, data_gen_time = stage_data(args, network, params, out_dir)

    # ----------------------------------------------------------------
    # Loop over the init seeds:
    #   single seed    → run_dir = out_dir
    #   multiple seeds → run_dir = out_dir/seed_{s}/, aggregated after the loop
    # ----------------------------------------------------------------
    seeds    = args.init_seeds
    multi    = len(seeds) > 1
    all_seed_results = {}   # seed → results_dict  (populated only for multi)

    for s in seeds:
        run_dir = out_dir / f'seed_{s}' if multi else out_dir
        if multi:
            run_dir.mkdir(parents=True, exist_ok=True)

        # Seed s initializes the network weights in stage_train. It is also
        # written to config, but only the per-seed snapshots of a multi-seed run
        # record it; the snapshot in out_dir is written before the loop and
        # keeps config's DEFAULT_INIT_SEED.
        args.init_seed = s
        _config.DEFAULT_INIT_SEED = s
        if multi:
            save_snapshot(run_dir)

        trainer       = None
        model         = None
        test_scen     = None
        results_dict  = None
        dfl_rollout   = None
        ar_rollout    = None
        nr_rollout    = None
        mpc_rollout   = None
        train_time    = 0.0

        # ---- Train stage ------------------------------------------------
        if args.mode in ('train', 'all'):
            if dataset is None:
                if args.cache_path:
                    dataset, _, data_gen_time = stage_data(args, network, params, run_dir)
                elif args.load_data and Path(args.load_data).exists():
                    dataset = load_training_data(args.load_data)
                    if dataset['n_scenarios'] > args.n_train:
                        print(f"  Subsampling dataset: {dataset['n_scenarios']} → {args.n_train} scenarios")
                        dataset = _subsample_dataset(dataset, args.n_train)
                    elif dataset['n_scenarios'] < args.n_train:
                        print(f"  Warning: loaded dataset has {dataset['n_scenarios']} scenarios, "
                              f"fewer than --n-train={args.n_train}. Training on all available.")
                else:
                    data_pkl = run_dir / 'training_data.pkl'
                    if data_pkl.exists():
                        dataset = load_training_data(data_pkl)
                    else:
                        raise RuntimeError(
                            "No training data found. Run with --mode data first "
                            "or supply --load-data / --cache-path."
                        )

            if args.load_model and Path(args.load_model).exists():
                print(f"\n  Loading pre-trained model from {args.load_model}")
                n_comp = network.n_generators + network.n_lines
                in_dim = n_comp + 1
                model, eps = load_repair_network(
                    args.load_model, in_dim, n_comp,
                    hidden_dims=tuple(args.hidden_dims)
                )
                trainer = PseudoDFLTrainer(network, params, epsilon=eps,
                                     hidden_dims=tuple(args.hidden_dims))
                trainer.load(args.load_model)
            else:
                trainer, model_path, train_time = stage_train(args, network, params,
                                                              dataset, run_dir)
                model = trainer.model

        # ---- Evaluate stage ---------------------------------------------
        if args.mode in ('evaluate', 'all'):
            if model is None:
                model_path = run_dir / 'model.pt' if args.load_model is None \
                    else Path(args.load_model)
                if not model_path.exists():
                    raise RuntimeError(
                        f"Model not found at {model_path}. "
                        "Run --mode train first."
                    )
                n_comp = network.n_generators + network.n_lines
                in_dim = n_comp + 1
                model, eps = load_repair_network(
                    model_path, in_dim, n_comp,
                    hidden_dims=tuple(args.hidden_dims)
                )
                if trainer is None:
                    trainer = PseudoDFLTrainer(network, params, epsilon=eps,
                                         hidden_dims=tuple(args.hidden_dims))
                    trainer.load(model_path)

            results_dict, test_scen, dfl_rollout, ar_rollout, nr_rollout, mpc_rollout = stage_evaluate(
                args, network, params, model, trainer, run_dir,
                data_gen_time=data_gen_time, train_time=train_time,
            )

        # ---- Compare stage ----------------------------------------------
        if args.mode in ('compare', 'all'):
            if results_dict is None:
                eval_json = run_dir / 'evaluation.json'
                if not eval_json.exists():
                    raise RuntimeError(
                        f"evaluation.json not found in {run_dir}. "
                        "Run --mode evaluate first."
                    )
                with open(eval_json) as f:
                    results_dict = json.load(f)
                print(f"  Loaded evaluation results from {eval_json}")

            if model is None or dfl_rollout is None:
                n_comp = network.n_generators + network.n_lines
                in_dim = n_comp + 1
                model_path = run_dir / 'model.pt' if args.load_model is None \
                    else Path(args.load_model)
                model, eps = load_repair_network(
                    model_path, in_dim, n_comp,
                    hidden_dims=tuple(args.hidden_dims)
                )
                if trainer is None:
                    trainer = PseudoDFLTrainer(network, params, epsilon=eps,
                                         hidden_dims=tuple(args.hidden_dims))
                    try:
                        trainer.load(model_path)
                    except Exception:
                        pass
                dfl_rollout = PseudoDFLRollout(network, params, model,
                                         lp_method=args.lp_method)
                ar_rollout  = AlwaysRepairRollout(network, params,
                                                  lp_method=args.lp_method)
                nr_rollout  = NoRepairRollout(network, params,
                                              lp_method=args.lp_method)
                mpc_rollout = None
                if isinstance(results_dict, dict) and 'NearSightedMPC' in results_dict:
                    mpc_rollout = NearSightedMPCRollout(network, params,
                                                        window=args.mpc_window,
                                                        lp_method=args.lp_method)

            if test_scen is None:
                gen_eval  = create_scenario_generator(params, seed=args.eval_seed)
                test_scen = gen_eval.generate_scenarios(args.n_eval)

            stage_compare(args, results_dict, trainer, dfl_rollout, ar_rollout, nr_rollout,
                          test_scen, network, params, run_dir, mpc_rollout=mpc_rollout)

        if multi:
            all_seed_results[s] = results_dict

    # ---- Multi-seed aggregation -----------------------------------------
    if multi:
        _aggregate_seed_runs(all_seed_results, seeds, out_dir)

    # ---- Final summary (uses last seed's results_dict) ------------------
    print()
    print("=" * 70)
    print("Pseudo_DFL COMPLETED SUCCESSFULLY")
    print("=" * 70)
    if results_dict:
        print()
        print("Cost summary:")
        print("-" * 50)
        for label, res in results_dict.items():
            if not isinstance(res, dict) or 'mean_cost' not in res:
                continue
            print(f"  {label:<15}: mean=${res['mean_cost']:>10,.2f}  "
                  f"std=${res['std_cost']:>9,.2f}")
        print()
        if 'NoRepair' in results_dict and 'Pseudo_DFL' in results_dict:
            base  = results_dict['NoRepair']['mean_cost']
            dfl_m = results_dict['Pseudo_DFL']['mean_cost']
            if base > 0:
                saving = (base - dfl_m) / base * 100
                print(f"  Pseudo_DFL cost saving vs no-repair: {saving:+.1f}%")
        if 'SAA' in results_dict and 'Pseudo_DFL' in results_dict:
            s_m   = results_dict['SAA']['mean_cost']
            dfl_m = results_dict['Pseudo_DFL']['mean_cost']
            gap   = (dfl_m - s_m) / s_m * 100
            print(f"  Pseudo_DFL vs SAA gap:                {gap:+.1f}%")
        if 'SDDiP' in results_dict and 'Pseudo_DFL' in results_dict:
            s_m   = results_dict['SDDiP']['mean_cost']
            dfl_m = results_dict['Pseudo_DFL']['mean_cost']
            gap   = (dfl_m - s_m) / s_m * 100
            print(f"  Pseudo_DFL vs SDDiP gap:              {gap:+.1f}%")
    print()
    print(f"  Results saved to: {out_dir}/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_seed_runs(all_seed_results, seeds, out_dir):
    """Aggregate per-seed evaluation results into out_dir.

    Each seed produces one value: its DFL mean_cost over the evaluation
    scenarios. The mean and std reported here (population std, ddof=0) are
    taken across these per-seed values. The seeds share the data and differ in
    init_seed (and, under a budget, in the unseeded Monte Carlo perturbations).

    Writes:
      out_dir/evaluation.json  mean/std across seeds and the per-seed means;
                               baselines from the first seed; training and
                               data-generation times averaged over the seeds
    """
    # One DFL mean cost per seed
    per_seed_means = {}
    for s, rd in all_seed_results.items():
        if rd is None:
            continue
        mc = (rd.get('Pseudo_DFL') or {}).get('mean_cost')
        if mc is not None:
            per_seed_means[s] = float(mc)

    if not per_seed_means:
        print("  Warning: no Pseudo_DFL costs found across seeds — skipping aggregation.")
        return

    means_arr = np.array(list(per_seed_means.values()))
    grand_mean = float(means_arr.mean())
    grand_std  = float(means_arr.std())

    # Timing is read back from each seed's evaluation.json: stage_evaluate passes
    # it straight to _save_eval_json, so the in-memory results_dict has no timing.
    train_times    = []
    datagen_times  = []
    ref_timing     = {}
    for s in seeds:
        seed_json = out_dir / f'seed_{s}' / 'evaluation.json'
        if not seed_json.exists():
            continue
        try:
            with open(seed_json) as f:
                sd = json.load(f)
            t = sd.get('timing', {})
            if t.get('training_s', 0) > 0:
                train_times.append(float(t['training_s']))
            if t.get('data_generation_s', 0) > 0:
                datagen_times.append(float(t['data_generation_s']))
            if not ref_timing:
                ref_timing = t.copy()
        except Exception:
            continue

    agg_timing = ref_timing.copy()
    if train_times:
        agg_timing['training_s'] = round(float(np.mean(train_times)), 2)
    if datagen_times:
        agg_timing['data_generation_s'] = round(float(np.mean(datagen_times)), 2)
    agg_timing['n_seeds'] = len(per_seed_means)

    # Build aggregated results_dict: baselines taken from first available seed
    first_rd = next(r for r in all_seed_results.values() if r is not None)
    agg = {k: v for k, v in first_rd.items() if k not in ('Pseudo_DFL', 'timing')}
    agg['timing'] = agg_timing
    agg['Pseudo_DFL'] = {
        'mean_cost':  grand_mean,
        'std_cost':   grand_std,
        'n_seeds':    len(per_seed_means),
        'seed_means': per_seed_means,
    }

    eval_path = out_dir / 'evaluation.json'
    _save_eval_json(agg, eval_path)
    print(f"\n  Aggregated evaluation ({len(per_seed_means)} seeds): "
          f"mean=${grand_mean:,.0f}  std=${grand_std:,.0f}  → {eval_path}")



def _save_eval_json(results_dict, path, timing=None):
    """Write evaluation results to JSON (NumPy arrays as lists, timing under 'timing')."""
    serial = {}
    if timing is not None:
        serial['timing'] = timing
    for label, res in results_dict.items():
        serial[label] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in res.items()}
    with open(path, 'w') as f:
        json.dump(serial, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
