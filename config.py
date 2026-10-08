"""
config.py: default parameters for the repair-scheduling experiments.

Pseudo_DFL_main.py (DFL) takes most of its command-line defaults from these
constants; the other entry points (main.py for SAA, SDDiP_main.py,
compare_methods.py) take some of theirs from here. The library modules use the
constants as keyword-argument defaults.

Sections
--------
  1.  Problem structure          (time horizon, repair delay)
  2.  Cost coefficients          (shedding penalty, repair costs)
  3.  Failure probability model  (log-linear logit)
  4.  Neural network             (architecture, regularization)
  5.  Fenchel-Young loss         (perturbation strength)
  6.  Training hyperparameters   (optimizer, scheduler, gradient clipping)
  7.  Data generation / MIP solver
  8.  SAA solver
  9.  Near-sighted MPC baseline (lookahead controller)
  10. Repair budget + Fenchel-Young sampling (budgeted model)
  11. Default seeds and evaluation
  Utilities: experiment_suffix(), get_snapshot(), save_snapshot()
"""

# =============================================================================
# 1. Problem structure
# =============================================================================

T   = 14    # total number of stages
TAU = 1     # repair delay (stages); a component whose repair is ordered at
            # stage t is back in service at stage t + TAU

# =============================================================================
# 2. Cost coefficients
# =============================================================================

LOAD_SHEDDING_COST = 1000.0   # $/MWh, penalty on unserved load (value of lost load)
REPAIR_BASE_SCALE  = 1        # repair-cost scale s, multiplies both repair cost bases;
                              #     1 = baseline costs, 50 = 50x more expensive
GEN_REPAIR_BASE    = 550.0 * REPAIR_BASE_SCALE   # $, mean generator repair cost
LINE_REPAIR_BASE   = 350.0 * REPAIR_BASE_SCALE   # $, mean line repair cost
# Runs set s with --repair-base-scale. The bases 550 and 350 are repeated as
# literals in main.py, SDDiP_main.py, compare_methods.py, Pseudo_DFL_main.py (used
# when s != 1), exact_dp/exact_dp_toy.py, paper/reproduce_paper_numbers.py,
# scripts/sensitivity_plot.py and scripts/slurm/sensitivity/sensitivity_sweep.sh;
# a change to them has to be made there as well.

# =============================================================================
# 3. Failure probability model  (log-linear logit)
# =============================================================================
# lambda^t[k] = 1 / (1 + (t + K)^{-A[k]} * exp(B[k] * t))
# where A[k], B[k] are set so that lambda^t[k] peaks at t = t_star with value
# p_max[k] in (1/2, 1) (see ModelParameters._compute_lambda).
# Validity: delta = (t_star + K)*ln(t_star + K) - t_star > 0
#           For t_star = 1 this requires K > ~0.76. K = 1.5 → delta ≈ 1.29.

FAILURE_K      = 1.5   # global shift; smaller K → faster decay after the peak
FAILURE_T_STAR = 1     # stage at which failure probability peaks (all components)

# Vulnerability classes: each component is drawn into the Standard or Elevated
# class with the model seed (the seed passed to ModelParameters).
FAILURE_CLASS_PROB     = 0.5   # P(Standard class); 1 - p → Elevated class
FAILURE_P_MAX_STD_LOW  = 0.55  # Standard: p_max ~ Uniform[STD_LOW, STD_HIGH]
FAILURE_P_MAX_STD_HIGH = 0.70
FAILURE_P_MAX_ELV_LOW  = 0.80  # Elevated:  p_max ~ Uniform[ELV_LOW, ELV_HIGH]
FAILURE_P_MAX_ELV_HIGH = 0.90

# =============================================================================
# 4. Neural network architecture
# =============================================================================

NN_HIDDEN_DIMS = (128, 256, 128)   # hidden layer widths for RepairNetwork
NN_DROPOUT     = 0.3                    # dropout probability after each hidden layer

# =============================================================================
# 5. Fenchel-Young loss
# =============================================================================

FY_EPSILON          = 0.1   # perturbation strength ε; larger ε → softer decisions

# =============================================================================
# 6. Training hyperparameters
# =============================================================================

TRAIN_N_EPOCHS   = 200   # epochs at n_train = 2000; Pseudo_DFL_main.py uses
                         # max(1, int(TRAIN_N_EPOCHS * n_train / 2000)) unless --n-epochs is set
TRAIN_BATCH_SIZE = 64     # mini-batch size
TRAIN_LR             = 1e-3   # initial Adam learning rate
TRAIN_WEIGHT_DECAY   = 1e-5   # Adam weight decay (L2 penalty coefficient)
TRAIN_VAL_SPLIT      = 0.2    # fraction of (scenario, stage) samples held out for validation
TRAIN_GRAD_CLIP      = 1.0    # gradient clipping max-norm (nn.utils.clip_grad_norm_)
TRAIN_SCHED_PATIENCE = 200     # ReduceLROnPlateau: epochs without improvement before LR drop
TRAIN_SCHED_FACTOR   = 0.5    # ReduceLROnPlateau LR multiplier; halving keeps the LR usable
                              # after several drops
TRAIN_EARLY_STOP_PATIENCE = 330  # stop if val loss has not improved for this many epochs

# =============================================================================
# 7. Data generation / MIP solver
# =============================================================================

# Solver names come from solver_config.py, which reads the ED_OPF_SOLVER,
# ED_OPF_MIP_SOLVER, ED_OPF_LP_SOLVER and ED_OPF_SDDIP_SOLVER environment
# variables. The two names below alias its Layer A and Layer C defaults; main.py,
# saa_solver.py, Pseudo_DFL_main.py and compare_methods.py import them from here,
# the other modules read solver_config directly.
import solver_config as _solver_config

DATA_SOLVER_NAME        = _solver_config.MIP_SOLVER  # default Pyomo MIP solver (Layer A)
LP_METHOD               = _solver_config.LP_SOLVER   # default per-stage DC-OPF backend (Layer C)
DATA_MIP_REL_GAP        = 1e-6            # relative MIP gap for the training-label MIPs
DATA_THREADS_PER_WORKER = 1               # solver threads per data-generation worker (HiGHS only).
                                          # One worker runs per CPU, so >1 over-subscribes the CPUs
DATA_MAXTASKS_PER_CHILD = 64              # recycle each data-gen worker after this many solves.
                                          # Each worker reuses one persistent solver across all
                                          # its scenarios; that solver's memory grows with the
                                          # number of solves, so on large networks / large n_train
                                          # long-lived workers can exhaust the node's memory and
                                          # the job is killed. Recycling releases the solver
                                          # periodically and bounds peak RAM. None keeps workers
                                          # for the whole run.

# =============================================================================
# 8. SAA solver
# =============================================================================

SAA_TIME_LIMIT_S = 172800  # wall-clock time limit for the SAA MIP solve (seconds, 48 h)
SAA_MIP_GAP      = 0.05   # relative MIP gap at which the SAA solve stops (5 %)
# Both are SAASolver.solve() defaults. main.py (--time-limit, --mip-gap) and
# EDOPFProblem.solve() repeat the same values as literals, so a change here has
# to be made there as well.

# Branching factor B of the sampled scenario tree used by the multistage SAA
# (mode='tree'). The tree has B + B^2 + ... + B^T decision nodes and B^T sample
# paths; each node carries one decision shared by every path through it, which
# enforces non-anticipativity. Node count grows as B^T, so at T=14 only B=2 is
# tractable (B=3 already implies millions of nodes). With B sampled outcomes per
# node, the tree is a coarse approximation of the per-stage failure distribution
# and its objective is a biased estimate; use it as a cross-check against SDDiP.
# mode='ws' ignores B and solves one MIP over independent scenarios (no coupling
# between them), whose objective estimates the Wait-and-See lower bound E[WS].
SAA_TREE_BRANCHING = 2

# =============================================================================
# 9. Near-sighted MPC baseline (lookahead controller)
# =============================================================================
# Certainty-equivalent MPC with zero terminal cost. At each stage it enumerates
# repair schedules of the currently failed components over a W-stage lookahead
# (each failed component -> a repair step or "never"), executes the step-0
# repairs of the cheapest schedule and replans at the next stage (rolling
# horizon). Step 0 is priced with the exact StagewiseOPF on the true state. The
# later steps roll the state forward with expected failures (lambda), which makes
# availability fractional, and are priced with StagewiseOPF.solve_derated
# (generator capacity, line susceptance and line capacity scaled by availability).
# Under a repair budget, schedules that overspend at any step are skipped.
# Any W with TAU + 1 <= W <= T is allowed; W = TAU + 1 (= 2 here) reduces to a
# plain subset enumeration and is the minimum meaningful window. Larger W captures
# more stages of repair benefit but enumerates (L - TAU + 1)^|F_t| schedules per
# stage, with L = min(W, T - t) for the 0-indexed stage t.

NEAR_SIGHTED_MPC_WINDOW = 3   # lookahead window W (default of --mpc-window); TAU + 1 <= W <= T

# ---------------------------------------------------------------------------
# Schedule-search cap (keeps MPC tractable on larger networks)
# ---------------------------------------------------------------------------
# Exact MPC enumerates (L - TAU + 1)^|F_t| repair schedules per stage. That is
# cheap on the toy (5-bus) network with 7 components but explodes on case14 (25
# components). The cap bounds the per-stage search:
#
#     reference = (W - TAU + 1) ** MPC_CAP_REFERENCE_COMPONENTS
#     cap       = ceil(ratio * reference)   # when ratio (MPC_SCHEDULE_CAP_RATIO) is not None
#
# Per stage: if the exact schedule count <= cap, all schedules are enumerated;
# otherwise MPC evaluates the no-repair schedule plus cap - 1 schedules drawn
# uniformly at random (with replacement, generated on the fly so the full space is
# never built) and keeps the cheapest. MPC is therefore exact wherever the count
# is within the cap and an approximate randomized lookahead elsewhere.
#
# MPC_CAP_REFERENCE_COMPONENTS is a plain reference fleet size. Its default is the
# toy network's component count (7), so ratio=1 means "as many schedules as a
# 7-component fleet at this window". The MPC code does not build or need the toy
# network.
#
# MPC_SCHEDULE_CAP_RATIO: None = exact everywhere, no cap (intractable beyond toy).
# 1 (default) = cap at the reference count; keeps toy exact (its largest count,
# 3^7 = 2187 at W = 3, equals the reference, so toy never samples) while making
# case14 tractable. 2 = double the cap, etc. Changing the cap affects only
# MPC results; the cap settings are part of the MPC cache signature
# (evaluate_mpc_cached in Pseudo_DFL_solver.py).

MPC_CAP_REFERENCE_COMPONENTS = 7      # reference fleet size (toy: 2 generators + 5 lines)
MPC_SCHEDULE_CAP_RATIO       = 1      # None = exact (no cap); r > 0 -> cap = ceil(r * reference)
MPC_SAMPLE_SEED              = 0      # seed of the schedule-sampling RNG (one RNG per MPC object)

# =============================================================================
# 10. Repair budget + Fenchel-Young sampling (budgeted model)
# =============================================================================
# Per-stage monetary repair budget B^t (exogenous; unspent budget does not carry
# over): at each stage the repair spend sum_k beta_k * r^t[k] must not exceed B^t.
# Every method enforces it: the Pyomo MIPs (training labels, SAA) and SDDiP as a
# linear constraint per stage, DFL deployment as a 0/1 knapsack (knapsack.py),
# Near-sighted MPC by skipping schedules that overspend, AlwaysRepair by repairing
# the cheapest failed components that fit, and exact DP by restricting its actions.
#
# The budget is a fraction phi of the network's total fleet repair cost:
#     B^t = phi * (sum gen_repair_cost + sum line_repair_cost).
# Because repair costs already adapt to each network's components, the same phi
# yields the same relative tightness on any network (phi >= 1 never binds, since
# repairing every component at once costs the fleet total). None disables the
# budget. ModelParameters turns phi into the absolute repair_budget_per_stage
# (or None) that the methods read.

REPAIR_BUDGET_FRACTION = None   # phi in (0, 1]; None = no budget (default). Set per run with
                                # --repair-budget-fraction (the budgeted scripts use 0.3)

# Monte-Carlo perturbation samples M for the budgeted Fenchel-Young surrogate
# (perturb-and-MAP with Gaussian noise over the 0/1 knapsack). Only used when the
# budget is active; kept small (< 10) because each sample costs one knapsack solve
# per row of every training batch and of the validation set. With the budget
# disabled, training uses the closed-form sigmoid surrogate and ignores this value.

FY_MC_SAMPLES = 5                # perturbation samples M (< 10)

# =============================================================================
# 11. Default seeds and evaluation
# =============================================================================

DEFAULT_TRAIN_SEED = 42     # seed for the failure-model draws (ModelParameters), the training
                            # scenarios, the train/val split, the batch order and dropout
                            # (default --seed of Pseudo_DFL_main.py)
DEFAULT_INIT_SEED  = 42     # seed for the network weight initialization (--init-seeds)
DEFAULT_EVAL_SEED  = 1000   # seed for the out-of-sample test scenarios; must differ from the
                            # training seed, or the test set repeats the first training scenarios
DEFAULT_N_EVAL     = 500    # test-set size: default --n-eval of compare_methods.py. The experiment
                            # scripts pass 500; Pseudo_DFL_main.py defaults to max(5, n_train // 3)

# =============================================================================
# Utilities
# =============================================================================

def experiment_suffix():
    """Return the folder suffix encoding the current repair-cost scale s.

    Returns '_scale<s>' when REPAIR_BASE_SCALE != 1 (a whole-number scale is
    printed without decimals), empty string otherwise. Pseudo_DFL_main.py uses it
    for the root of its training-data and MPC caches, e.g.:
        'data_cache' + experiment_suffix()
        → 'data_cache'          (scale 1)
        → 'data_cache_scale55'  (scale 55)
    Shell usage:
        python3 -c "import config; config.REPAIR_BASE_SCALE=55; print(config.experiment_suffix())"
    """
    s = REPAIR_BASE_SCALE
    if s == 1:
        return ""
    n = int(s) if s == int(s) else s
    return f"_scale{n}"


def get_snapshot():
    """Return all config constants as a plain dict (for reproducibility logs).

    Includes every uppercase module-level name that does not start with an
    underscore; functions and the solver_config module are left out. Tuples
    are converted to lists so the result is directly JSON-serializable.
    """
    import sys
    module = sys.modules[__name__]
    result = {}
    for k, v in vars(module).items():
        if k.isupper() and not k.startswith('_'):
            result[k] = list(v) if isinstance(v, tuple) else v
    return result


def save_snapshot(output_dir):
    """Write get_snapshot() to output_dir/config_snapshot.json.

    main.py, Pseudo_DFL_main.py and SDDiP_main.py call it once the output
    directory exists, after copying the repair-cost scale and bases, the budget
    fraction and (Pseudo_DFL_main.py only) the seeds from their command line into
    this module. Other command-line overrides (e.g. --tau, --failure-k, --mip-gap)
    are not reflected in the file.
    """
    import json
    from pathlib import Path
    path = Path(output_dir) / 'config_snapshot.json'
    with open(path, 'w') as f:
        json.dump(get_snapshot(), f, indent=2)
    print(f"  Config snapshot   → {path}")
