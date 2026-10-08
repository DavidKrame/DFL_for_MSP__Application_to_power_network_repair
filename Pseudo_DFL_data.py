"""
Pseudo_DFL Data Module
======================
Generates, caches and loads the DFL training data.

Training samples are (state, optimal_repair) pairs extracted from
single-scenario T-stage MIPs, solved to a relative gap of mip_rel_gap
(DATA_MIP_REL_GAP = 1e-6 by default).

For each training scenario s:
  1. Solve the T-stage deterministic MIP for scenario s (EDOPFProblem with this
     one scenario: the failure path is known in advance and there is no
     branching).
  2. For each stage t = 1..T extract:
       features[t]  = [a_gen^t, a_line^t, t/T]   shape (input_dim,)
       r_target[t]  = [r_gen^t, r_line^t]         shape (n_components,)
       mask[t]      = [1-a_gen^t, 1-a_line^t]     shape (n_components,)
                      (1 = failed = may be repaired)

with input_dim = n_gen + n_lines + 1 and n_components = n_gen + n_lines.
A scenario whose MIP fails keeps zero arrays and solved_flag = False, so the
dataset holds n_solved × T usable training samples.

Data arrays have shape (n_scenarios, T, ...) so both the per-stage and
per-scenario structure are preserved. Call flatten_dataset() to get flat
arrays (N, ...) for PyTorch DataLoader usage.
"""

import logging
import multiprocessing as mp
import os
import platform
import numpy as np
import pickle
import warnings
from pathlib import Path

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from network_data import create_network
from model_parameters import create_model_parameters
from scenario_generator import create_scenario_generator
from problem_formulation import EDOPFProblem
from config import DATA_MIP_REL_GAP, DATA_THREADS_PER_WORKER, DATA_MAXTASKS_PER_CHILD
import solver_config


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

# Per-worker-process state, filled by _worker_init (the Pool initializer).
# Each worker process has its own copy.
_worker_state: dict = {}


def _worker_init(network, params, solver_name,
                 mip_rel_gap=DATA_MIP_REL_GAP,
                 threads_per_worker=DATA_THREADS_PER_WORKER):
    """
    Pool initializer: called once in each worker process, including the
    workers that replace retired ones under maxtasksperchild.
    Sets up the solver; the Pyomo model is built fresh for each scenario so
    that xi failure values are baked as Python constants into the A-matrix,
    keeping the problem a clean MILP with no mutable-coefficient overhead.
    """
    # One OpenMP / MKL / OpenBLAS / numexpr thread per worker. The limit takes
    # effect through the environment that generate_training_data sets before
    # the pool starts (NumPy is already imported when this initializer runs);
    # setdefault keeps any value already set. Without the limit, W workers on a
    # W-CPU node each running multithreaded BLAS would oversubscribe the CPUs.
    os.environ.setdefault('OMP_NUM_THREADS',     '1')
    os.environ.setdefault('MKL_NUM_THREADS',     '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    # appsi_highs (the default) runs HiGHS in-process through highspy, so there
    # is no temp-file round trip to a solver executable, and it is safe in
    # spawned/forkserver workers (fresh process, no inherited file descriptors).
    # make_mip_solver sets each solver's own option keys from the registry in
    # solver_config: HiGHS-family solvers get threads, presolve, mip_rel_gap
    # and a silent log; other solvers (glpk, gurobi, ...) get only their gap
    # option and skip the HiGHS-only settings.
    solver = solver_config.make_mip_solver(
        solver_name, threads=threads_per_worker, mip_gap=mip_rel_gap, quiet=True)

    _worker_state['network']  = network
    _worker_state['params']   = params
    _worker_state['solver']   = solver
    _worker_state['n_gen']    = network.n_generators
    _worker_state['n_lines']  = network.n_lines
    _worker_state['T']        = params.T


def _solve_single_scenario(args):
    """
    Solve one scenario by building a fresh Pyomo model and solving it.
    xi failure values are baked as Python constants at model build time,
    so the solver always sees a clean MILP with a fixed A-matrix.

    args is (scenario, s_idx). Returns a dict with 's_idx' and 'solved' and,
    for a solved scenario, the arrays 'features' (T, input_dim), 'r_target'
    and 'mask' (T, n_components), 'a_gen_all' (T, n_gen) and 'a_line_all'
    (T, n_lines).
    """
    scenario, s_idx = args

    network = _worker_state['network']
    params  = _worker_state['params']
    solver  = _worker_state['solver']
    n_gen   = _worker_state['n_gen']
    n_lines = _worker_state['n_lines']
    T       = _worker_state['T']

    problem = EDOPFProblem(network, params, [scenario])
    problem.build_model()
    model = problem.model

    pyomo_logger = logging.getLogger('pyomo')
    prev_level = pyomo_logger.level
    pyomo_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results = solver.solve(model, tee=False)
    except RuntimeError:
        # appsi_highs raises RuntimeError when no feasible solution is found
        # (e.g. the solver ran out of memory on a busy node; the data MIPs have
        # no time limit). Skip the scenario rather than crashing the whole
        # worker pool.
        pyomo_logger.setLevel(prev_level)
        return {'s_idx': s_idx, 'solved': False}
    pyomo_logger.setLevel(prev_level)

    tc = results.solver.termination_condition
    if tc == pyo.TerminationCondition.infeasible:
        return {'s_idx': s_idx, 'solved': False}
    # Any other non-optimal status counts as solved if a solution was loaded,
    # i.e. if the objective can be evaluated.
    if tc != pyo.TerminationCondition.optimal:
        try:
            pyo.value(model.objective)
        except Exception:
            return {'s_idx': s_idx, 'solved': False}

    # Read the solution into (T, n_gen) and (T, n_lines) arrays in one pass.
    # The .value attribute avoids the per-call overhead of pyo.value();
    # rounding removes the solver's integrality tolerance from the binaries.
    a_gen_out  = np.round(np.array(
        [[model.a_gen [0, t+1, g].value for g in range(n_gen)]   for t in range(T)],
        dtype=float))
    a_line_out = np.round(np.array(
        [[model.a_line[0, t+1, e].value for e in range(n_lines)] for t in range(T)],
        dtype=float))
    r_gen_all  = np.round(np.array(
        [[model.r_gen [0, t+1, g].value for g in range(n_gen)]   for t in range(T)],
        dtype=float))
    r_line_all = np.round(np.array(
        [[model.r_line[0, t+1, e].value for e in range(n_lines)] for t in range(T)],
        dtype=float))

    stage_fracs = (np.arange(1, T + 1, dtype=float) / T).reshape(T, 1)
    features    = np.concatenate([a_gen_out, a_line_out, stage_fracs], axis=1)
    r_target    = np.concatenate([r_gen_all, r_line_all], axis=1)
    mask        = np.concatenate([1 - a_gen_out, 1 - a_line_out], axis=1)

    return {
        's_idx':      s_idx,
        'solved':     True,
        'features':   features,
        'r_target':   r_target,
        'mask':       mask,
        'a_gen_all':  a_gen_out,
        'a_line_all': a_line_out,
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _subsample_dataset(dataset, n_scenarios):
    """Return dataset containing only the first n_scenarios scenarios."""
    if dataset['n_scenarios'] <= n_scenarios:
        return dataset
    result = dict(dataset)
    for k in ('features', 'r_target', 'mask', 'a_gen_all', 'a_line_all', 'solved_flag'):
        result[k] = dataset[k][:n_scenarios].copy()
    if 'scenario_ids' in dataset:
        result['scenario_ids'] = dataset['scenario_ids'][:n_scenarios].copy()
    result['n_scenarios'] = n_scenarios
    result['n_solved']    = int(result['solved_flag'].sum())
    return result


def _merge_datasets(existing, new_data):
    """Concatenate two datasets along the scenario axis (new_data if existing is None)."""
    if existing is None:
        return new_data
    merged = {}
    for k in ('features', 'r_target', 'mask', 'a_gen_all', 'a_line_all'):
        merged[k] = np.concatenate([existing[k], new_data[k]], axis=0)
    merged['solved_flag'] = np.concatenate(
        [existing['solved_flag'], new_data['solved_flag']])
    # A dataset without 'scenario_ids' (cache files written without the field)
    # gets IDs from its position: 0..n-1 for existing, then n, n+1, ... for new_data.
    ex_ids  = existing.get('scenario_ids',
                            np.arange(existing['n_scenarios']))
    new_ids = new_data.get('scenario_ids',
                            np.arange(existing['n_scenarios'],
                                      existing['n_scenarios'] + new_data['n_scenarios']))
    merged['scenario_ids'] = np.concatenate([ex_ids, new_ids])
    merged['n_scenarios']  = existing['n_scenarios'] + new_data['n_scenarios']
    merged['n_solved']     = int(merged['solved_flag'].sum())
    for k in ('T', 'n_gen', 'n_lines', 'input_dim', 'n_components'):
        merged[k] = existing[k]
    merged['seed'] = existing.get('seed') or new_data.get('seed')
    return merged


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _checkpoint_cache(cached, features_all, r_target_all, mask_all,
                      a_gen_all, a_line_all, solved_flag,
                      n_new, n_solved, T, n_gen, n_lines, in_dim, n_comp,
                      seed, n_cached, n_scenarios, cache_path):
    """Atomically save in-progress generation results to the cache.

    Writes all n_new scenario slots; slots not solved yet (pending or failed)
    have solved_flag=False and zero arrays. On restart the cache is fully sized,
    so generation is skipped and training proceeds on whatever was solved: the
    pending scenarios are not solved again. The temp-file + rename pattern
    prevents a corrupt cache if the process is killed mid-write.
    """
    partial_new = {
        'features':     features_all,
        'r_target':     r_target_all,
        'mask':         mask_all,
        'a_gen_all':    a_gen_all,
        'a_line_all':   a_line_all,
        'solved_flag':  solved_flag.copy(),
        'n_scenarios':  n_new,
        'n_solved':     int(n_solved),
        'T':            T,
        'n_gen':        n_gen,
        'n_lines':      n_lines,
        'input_dim':    in_dim,
        'n_components': n_comp,
        'seed':         seed,
        'scenario_ids': np.arange(n_cached, n_scenarios),
    }
    merged = _merge_datasets(cached, partial_new)
    cache_path = Path(cache_path)
    tmp_path   = cache_path.with_suffix('.tmp')
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, 'wb') as f:
        pickle.dump(merged, f)
    tmp_path.replace(cache_path)
    print(f"  Checkpoint saved: {n_solved}/{n_new} new scenarios → {cache_path}",
          flush=True)


def generate_training_data(network, params, n_scenarios,
                            solver_name=None, seed=42, verbose=True,
                            cache_path=None,
                            mip_rel_gap=DATA_MIP_REL_GAP,
                            threads_per_worker=DATA_THREADS_PER_WORKER,
                            maxtasksperchild=DATA_MAXTASKS_PER_CHILD):
    """
    Generate DFL training data by solving single-scenario MIPs in parallel.

    Scenario i is drawn with the RNG seed [seed, i], so it does not depend on
    n_scenarios. If cache_path is provided the function checks for an existing
    cache on disk. Scenarios already stored there are reused; only the missing
    IDs n_cached … n_scenarios-1 are generated and solved. The merged dataset
    is saved back to cache_path, so a later call with a larger n_scenarios
    solves only the new MIPs. With verbose=True the partial results are also
    checkpointed to the cache about every 5% of the new scenarios (see
    _checkpoint_cache).

    The cache is checked only for its seed; a cache with another seed is
    ignored and then overwritten. Keeping caches for different networks,
    repair-cost scales and budgets apart is up to the caller
    (Pseudo_DFL_main.py encodes them in the cache path).

    Parameters
    ----------
    network      : NetworkData
    params       : ModelParameters
    n_scenarios  : int        total number of training scenarios needed
                              (IDs 0 … n_scenarios-1)
    solver_name  : str or None  MIP solver ('appsi_highs', 'gurobi', 'glpk', …).
                              None resolves to solver_config.MIP_SOLVER.
    seed         : int        scenario seed
    verbose      : bool       progress output; also enables the checkpoints
    cache_path   : str or Path or None
                   Path to the reusable cache pkl.  If it exists and its
                   stored seed matches, its scenarios are reused.
    mip_rel_gap  : float      relative MIP gap
    threads_per_worker : int  solver threads per worker (HiGHS-family only)
    maxtasksperchild   : int or None
                   scenarios a worker solves before it is replaced;
                   None keeps each worker for the whole run

    Returns
    -------
    dataset  : dict   shape (n_scenarios, T, …); see the module docstring
    scenarios: list of newly generated Scenario objects (empty on cache hit)
    """
    solver_name = solver_config.resolve_mip_solver(solver_name)
    n_gen   = network.n_generators
    n_lines = network.n_lines
    T       = params.T
    n_comp  = n_gen + n_lines
    in_dim  = n_gen + n_lines + 1  # a_gen, a_line, stage_frac

    # ── Load existing cache ──────────────────────────────────────────────────
    cached   = None
    n_cached = 0
    if cache_path is not None:
        _cpath = Path(cache_path)
        if _cpath.exists():
            try:
                cached = load_training_data(_cpath)
                if cached.get('seed') != seed:
                    print(f"  Warning: cache seed ({cached.get('seed')}) != "
                          f"requested seed ({seed}). Ignoring cache.")
                    cached = None
                else:
                    n_cached = cached['n_scenarios']
            except Exception as exc:
                print(f"  Warning: cache load failed ({exc}). Regenerating.")

    # ── Pure cache hit: no MIPs to solve ────────────────────────────────────
    if n_cached >= n_scenarios:
        if verbose:
            print(f"  Cache hit: {n_cached} scenarios available, "
                  f"{n_scenarios} requested — 0 MIPs to solve.")
        return _subsample_dataset(cached, n_scenarios), []

    # ── Generate and solve only the missing scenarios ────────────────────────
    n_new = n_scenarios - n_cached

    gen_obj       = create_scenario_generator(params, seed=seed)
    new_scenarios = gen_obj.generate_scenarios(n_new, start_id=n_cached)

    # forkserver on Linux; spawn on other platforms such as macOS (Tk/GUI safe).
    _mp_ctx = 'forkserver' if platform.system() == 'Linux' else 'spawn'

    # Single-threaded BLAS/OpenMP in the workers, which inherit these variables
    # (see _worker_init).
    for _env_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(_env_var, '1')

    _slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    _avail_cpus = int(_slurm_cpus) if _slurm_cpus else (os.cpu_count() or 1)
    n_workers = min(_avail_cpus, n_new)

    if verbose:
        extra = f", extending cache from {n_cached}" if n_cached > 0 else ""
        print(f"Generating training data: {n_new} new scenarios "
              f"(IDs {n_cached}..{n_scenarios - 1}{extra}, "
              f"solver={solver_name}, seed={seed}, workers={n_workers}, "
              f"maxtasksperchild={maxtasksperchild}, ctx={_mp_ctx})",
              flush=True)

    features_all = np.zeros((n_new, T, in_dim))
    r_target_all = np.zeros((n_new, T, n_comp))
    mask_all     = np.zeros((n_new, T, n_comp))
    a_gen_all    = np.zeros((n_new, T, n_gen))
    a_line_all   = np.zeros((n_new, T, n_lines))
    solved_flag  = np.zeros(n_new, dtype=bool)
    n_solved     = 0

    ctx   = mp.get_context(_mp_ctx)
    tasks = [(new_scenarios[i], i) for i in range(n_new)]
    progress_interval = max(1, n_new // 20)

    with ctx.Pool(
            processes=n_workers,
            maxtasksperchild=maxtasksperchild,
            initializer=_worker_init,
            initargs=(network, params, solver_name,
                      mip_rel_gap, threads_per_worker)) as pool:

        for completed, res in enumerate(
                pool.imap_unordered(_solve_single_scenario, tasks), start=1):
            s_idx = res['s_idx']
            if res['solved']:
                features_all[s_idx] = res['features']
                r_target_all[s_idx] = res['r_target']
                mask_all    [s_idx] = res['mask']
                a_gen_all   [s_idx] = res['a_gen_all']
                a_line_all  [s_idx] = res['a_line_all']
                solved_flag [s_idx] = True
                n_solved += 1
            if verbose and completed % progress_interval == 0:
                print(f"  [{completed / n_new * 100:5.1f}%] "
                      f"{completed}/{n_new} new scenarios done "
                      f"({n_solved} solved)", flush=True)
                if cache_path is not None:
                    _checkpoint_cache(
                        cached, features_all, r_target_all, mask_all,
                        a_gen_all, a_line_all, solved_flag,
                        n_new, n_solved, T, n_gen, n_lines, in_dim, n_comp,
                        seed, n_cached, n_scenarios, cache_path,
                    )

    if verbose:
        if n_new % progress_interval != 0:
            print(f"  [100.0%] {n_new}/{n_new} new scenarios done "
                  f"({n_solved} solved)", flush=True)
        n_valid   = mask_all.sum()
        n_repairs = r_target_all.sum()
        print(f"  Solved {n_solved}/{n_new} new scenarios successfully.")
        print(f"  New training samples : {n_solved * T}")
        print(f"  Total repairs in new data : {int(n_repairs)}")
        if n_valid > 0:
            print(f"  Repair rate (of possible repairable): "
                  f"{n_repairs / n_valid * 100:.1f}%")

    new_dataset = {
        'features':     features_all,
        'r_target':     r_target_all,
        'mask':         mask_all,
        'a_gen_all':    a_gen_all,
        'a_line_all':   a_line_all,
        'solved_flag':  solved_flag,
        'n_scenarios':  n_new,
        'n_solved':     int(n_solved),
        'T':            T,
        'n_gen':        n_gen,
        'n_lines':      n_lines,
        'input_dim':    in_dim,
        'n_components': n_comp,
        'seed':         seed,
        'scenario_ids': np.arange(n_cached, n_scenarios),
    }

    # ── Merge with existing cache and persist ────────────────────────────────
    merged = _merge_datasets(cached, new_dataset)
    if cache_path is not None:
        save_training_data(merged, Path(cache_path))

    return merged, new_scenarios


# ---------------------------------------------------------------------------
# Flat view for training
# ---------------------------------------------------------------------------

def flatten_dataset(dataset, only_solved=True, only_repairable=False):
    """
    Flatten (n_scenarios, T, ...) → (N, ...) for PyTorch DataLoader.

    Parameters
    ----------
    dataset         : dict from generate_training_data
    only_solved     : bool  keep only rows from successfully solved scenarios
    only_repairable : bool  keep only samples where at least one component
                            could be repaired (mask.sum > 0)

    Returns
    -------
    features : ndarray (N, input_dim)
    r_target : ndarray (N, n_components)
    mask     : ndarray (N, n_components)
    """
    N_s = dataset['n_scenarios']
    T   = dataset['T']

    features = dataset['features'].reshape(N_s * T, -1)
    r_target = dataset['r_target'].reshape(N_s * T, -1)
    mask     = dataset['mask'].reshape(N_s * T, -1)

    # Row indices: row s*T + t belongs to scenario s
    keep = np.ones(N_s * T, dtype=bool)

    if only_solved:
        solved = dataset['solved_flag']                  # (N_s,)
        solved_expanded = np.repeat(solved, T)           # (N_s*T,)
        keep &= solved_expanded

    if only_repairable:
        keep &= (mask.sum(axis=1) > 0)

    return features[keep], r_target[keep], mask[keep]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_training_data(dataset, path):
    """Save dataset to disk (pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    if True:   # always print
        n = dataset['n_solved'] * dataset['T']
        print(f"  Training data saved: {n} samples → {path}")


def load_training_data(path):
    """Load dataset from disk."""
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    n = dataset.get('n_solved', dataset['n_scenarios']) * dataset['T']
    print(f"  Training data loaded: {n} samples ← {path}")
    return dataset


# ---------------------------------------------------------------------------
# Self-test: 10 scenarios on the default toy (5-bus) network
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from network_data import create_network
    from model_parameters import create_model_parameters

    net    = create_network()
    params = create_model_parameters(net, seed=42)

    dataset, scens = generate_training_data(
        net, params,
        n_scenarios=10,
        solver_name=None,   # -> solver_config.MIP_SOLVER
        seed=42,
        verbose=True
    )

    feats, r_tgt, msk = flatten_dataset(dataset, only_solved=True,
                                        only_repairable=False)
    print(f"\nFlattened shapes: features={feats.shape}, "
          f"r_target={r_tgt.shape}, mask={msk.shape}")
    print(f"Feature sample (scenario 0, stage 1): {feats[0]}")
    print(f"Repair target  (scenario 0, stage 1): {r_tgt[0]}")
