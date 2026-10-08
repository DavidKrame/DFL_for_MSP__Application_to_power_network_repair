"""
Solver configuration
====================
Chooses the optimization solver for each of the three layers of the pipeline:

    Layer A  Pyomo MIPs        (DFL training labels, SAA)       -> SolverFactory
    Layer B  SDDiP / SDDP.jl   (Julia, through juliacall)       -> HiGHS.Optimizer
    Layer C  per-stage DC-OPF  (StagewiseOPF, scipy linprog)    -> method='highs'

The defaults shown (HiGHS in every layer) apply when no environment variable is
set. The --solver and --lp-method flags of the entry points take their defaults
from the values resolved here.

Selecting a solver
------------------
Selection is via environment variables (read once, at import time):

    ED_OPF_SOLVER          master setting; e.g. "gurobi" switches all three layers
    ED_OPF_MIP_SOLVER      overrides Layer A only
    ED_OPF_LP_SOLVER       overrides Layer C only  (scipy method name, or "gurobi")
    ED_OPF_SDDIP_SOLVER    overrides Layer B only  ("highs" or "gurobi")

The experiment scripts in scripts/slurm/unbudgeted/ and scripts/slurm/budgeted/
export ED_OPF_SOLVER from their SOLVER variable (default gurobi).

A Gurobi-family master (any name containing "gurobi", e.g. "gurobi",
"gurobi_direct", "appsi_gurobi") maps Layer C to the gurobipy LP path and Layer B
to ``Gurobi.Optimizer``. Layer C only accepts a scipy linprog method or 'gurobi',
and Layer B only 'highs' or 'gurobi', so any other master keeps both on HiGHS.
Gurobi in Layer B needs Gurobi.jl in the Julia environment, which juliapkg.json
does not include (see README.md).

Editing ``_DEFAULT_MIP_SOLVER`` below changes the Layer A default only; Layers B
and C follow ED_OPF_SOLVER, not that default.

To add a Pyomo MIP solver, add an entry to ``MIP_SOLVERS`` below; the --solver
flags of main.py and Pseudo_DFL_main.py take their choices from it.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Defaults (edit here for an environment-independent change of Layer A)
# ---------------------------------------------------------------------------
_DEFAULT_MIP_SOLVER = 'appsi_highs'   # Pyomo MIP solver used when no variable is set


# ---------------------------------------------------------------------------
# Per-solver capability registry (Layer A)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MIPSolverSpec:
    """Option names and capabilities of one Pyomo MIP solver.

    ``*_option`` are the Pyomo ``solver.options[...]`` keys for that solver
    (``None`` means the option is never set and the solver keeps its own default).
    """
    mip_gap_option:     Optional[str]
    time_limit_option:  Optional[str]
    time_limit_cast:    Callable = float
    threads_option:     Optional[str] = None      # None -> threads not set; solver picks its own
    supports_chunking:  bool = False              # may run as checkpointed time chunks
    supports_warmstart: bool = False              # warm start from the values on the model
    is_highs_family:    bool = False              # quiet=True applies the HiGHS options


# Gurobi variants: gurobipy parameter names are case-insensitive, so the same
# option keys ('MIPGap', 'timelimit') work for the shell ('gurobi') and the
# in-memory ('gurobi_direct', 'appsi_gurobi') interfaces. Threads are left to
# Gurobi's own core auto-detection (respects SLURM cgroup allocation); only the
# HiGHS entries set a thread count.
MIP_SOLVERS = {
    'appsi_highs': MIPSolverSpec(
        mip_gap_option='mip_rel_gap', time_limit_option='time_limit',
        time_limit_cast=float, threads_option='threads',
        supports_chunking=True, supports_warmstart=False, is_highs_family=True),
    'highs': MIPSolverSpec(
        mip_gap_option='mip_rel_gap', time_limit_option='time_limit',
        time_limit_cast=float, threads_option='threads',
        supports_chunking=True, supports_warmstart=False, is_highs_family=True),
    'glpk': MIPSolverSpec(
        mip_gap_option='mipgap', time_limit_option='tmlim',
        time_limit_cast=int, threads_option=None,
        supports_chunking=False, supports_warmstart=False),
    'cbc': MIPSolverSpec(
        mip_gap_option='ratioGap', time_limit_option='seconds',
        time_limit_cast=float, threads_option=None,
        supports_chunking=True, supports_warmstart=True),
    'cplex': MIPSolverSpec(
        mip_gap_option='mip_tolerances_mipgap', time_limit_option='timelimit',
        time_limit_cast=float, threads_option=None,
        supports_chunking=True, supports_warmstart=True),
    'gurobi': MIPSolverSpec(
        mip_gap_option='MIPGap', time_limit_option='timelimit',
        time_limit_cast=float, threads_option=None,
        supports_chunking=True, supports_warmstart=True),
    'gurobi_direct': MIPSolverSpec(
        mip_gap_option='MIPGap', time_limit_option='timelimit',
        time_limit_cast=float, threads_option=None,
        supports_chunking=True, supports_warmstart=True),
    'appsi_gurobi': MIPSolverSpec(
        mip_gap_option='MIPGap', time_limit_option='timelimit',
        time_limit_cast=float, threads_option=None,
        supports_chunking=True, supports_warmstart=True),
}

# Spec for an unregistered solver name: the solver keeps its own gap and time
# defaults, with no threads, chunking or warmstart. get_spec() builds the same
# spec for unknown names, with is_highs_family inferred from the name.
_GENERIC_SPEC = MIPSolverSpec(mip_gap_option=None, time_limit_option=None)


def _is_gurobi_family(name: Optional[str]) -> bool:
    return name is not None and 'gurobi' in name.lower()


# ---------------------------------------------------------------------------
# Resolved defaults for the three layers (computed once at import)
# ---------------------------------------------------------------------------
_master = os.environ.get('ED_OPF_SOLVER') or None

#: Layer A: default Pyomo MIP solver name (ED_OPF_MIP_SOLVER, else the master,
#: else _DEFAULT_MIP_SOLVER).
MIP_SOLVER = os.environ.get('ED_OPF_MIP_SOLVER') or _master or _DEFAULT_MIP_SOLVER

#: Layer C: default scipy linprog method, or the literal 'gurobi' (gurobipy path).
LP_SOLVER = (os.environ.get('ED_OPF_LP_SOLVER')
             or ('gurobi' if _is_gurobi_family(_master) else 'highs'))

#: Layer B: default SDDP.jl optimizer, 'highs' or 'gurobi'.
SDDIP_SOLVER = (os.environ.get('ED_OPF_SDDIP_SOLVER')
                or ('gurobi' if _is_gurobi_family(_master) else 'highs'))


# ---------------------------------------------------------------------------
# Helpers (Layer A): apply options through the registry, so callers need no
# per-solver logic
# ---------------------------------------------------------------------------
def get_spec(solver_name: str) -> MIPSolverSpec:
    """Return the registry spec for ``solver_name`` (generic fallback if unknown)."""
    spec = MIP_SOLVERS.get(solver_name)
    if spec is not None:
        return spec
    # Unknown solver: infer highs-family from the name, otherwise use solver defaults.
    return MIPSolverSpec(mip_gap_option=None, time_limit_option=None,
                         is_highs_family='highs' in solver_name.lower())


def resolve_mip_solver(solver_name: Optional[str]) -> str:
    """``solver_name`` if given, else the configured default :data:`MIP_SOLVER`."""
    return solver_name or MIP_SOLVER


def supports_chunking(solver_name: str) -> bool:
    """Whether EDOPFProblem.solve may split the solve into checkpointed time chunks."""
    return get_spec(solver_name).supports_chunking


def supports_warmstart(solver_name: str) -> bool:
    return get_spec(solver_name).supports_warmstart


def set_mip_gap(solver, solver_name: str, mip_gap) -> None:
    """Set the MIP optimality-gap tolerance, using the right option key."""
    if mip_gap is None:
        return
    opt = get_spec(solver_name).mip_gap_option
    if opt is not None:
        solver.options[opt] = mip_gap


def set_time_limit(solver, solver_name: str, time_limit) -> None:
    """Set the wall-clock time limit, using the right option key and cast."""
    if time_limit is None:
        return
    spec = get_spec(solver_name)
    if spec.time_limit_option is not None:
        solver.options[spec.time_limit_option] = spec.time_limit_cast(time_limit)


def make_mip_solver(solver_name: Optional[str] = None, *,
                    mip_gap=None, time_limit=None, threads=None,
                    quiet: bool = False):
    """Build a configured Pyomo ``SolverFactory`` for ``solver_name``.

    Applies threads / gap / time-limit through the registry, so each solver gets
    its own option names; ``threads`` is ignored for solvers without a threads
    option. ``quiet`` silences console logging and turns presolve on for
    HiGHS-family solvers; it is a no-op for other solvers.
    """
    # Local imports keep this module cheap to import (config.py imports it).
    # Importing pyomo.environ registers the named solver plugins ('appsi_highs',
    # 'gurobi', ...), so the factory works whatever the caller has imported.
    import pyomo.environ  # noqa: F401  (side effect: registers solver plugins)
    from pyomo.opt import SolverFactory

    solver_name = resolve_mip_solver(solver_name)
    spec = get_spec(solver_name)
    solver = SolverFactory(solver_name)

    if threads is not None and spec.threads_option is not None:
        solver.options[spec.threads_option] = threads
    set_mip_gap(solver, solver_name, mip_gap)
    set_time_limit(solver, solver_name, time_limit)

    if quiet and spec.is_highs_family:
        solver.options['presolve'] = 'on'
        solver.options['log_to_console'] = 0

    return solver


def describe() -> str:
    """One-line summary of the active selection (for logs / verify_installation)."""
    src = f"master={_master!r}" if _master else "master=<unset>"
    return (f"solver_config: MIP={MIP_SOLVER}  LP={LP_SOLVER}  "
            f"SDDiP={SDDIP_SOLVER}  ({src})")
