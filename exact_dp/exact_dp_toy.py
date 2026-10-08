#!/usr/bin/env python3
"""
Exact finite-horizon dynamic program for the toy (5-bus) network.

The file lives in exact_dp/ and imports the network, parameter and stage-OPF
code from the repository root, so it solves the same toy model, with the same
failure probabilities and repair costs, as the other methods.

Exactness scope
---------------
The dynamic program solves the toy instance of the multistage model in
SDDiP_julia_bridge.py exactly. It relies on:

* the built-in toy network (2 generators and 5 lines);
* repair delay tau = 1, so a repair ordered at stage t is in service at t + 1;
* independent Bernoulli failures with the stage probabilities in
  ModelParameters;
* repair decisions taken after the current failure realization is observed;
* the DC-OPF of Pseudo_DFL_stage_opf.StagewiseOPF as the stage dispatch problem.

It is not an AC-power-flow benchmark. Numerical LP tolerances still apply.

State encoding
--------------
Each component has one of three reachable pre-shock states:

    0 = operational
    1 = failed, no repair pending
    2 = failed, repair pending and completing this stage

For seven components this gives at most 3^7 = 2,187 states per stage. The
dynamic program recombines identical states instead of enumerating complete
histories. After the failure shock the state is a binary availability mask
(2^7 = 128 values), and the repair policy is defined on these masks.

Recursion
---------
With V_T = 0 and 0-indexed stages t = T-1, ..., 0:

    V_t(s) = sum_a P_t(a | s) * [ OPF(a) + min_r ( c(r) + V_{t+1}(next(a, r)) ) ]

where a is the post-shock availability mask, OPF(a) the stage DC-OPF cost
(generation plus load shedding), r a subset of the components failed in a
whose repair cost c(r) fits the per-stage budget (when there is one), and
next(a, r) the next pre-shock state: available components operational,
repaired ones pending, the rest failed.

Usage
-----
From the repository root:

    python exact_dp/exact_dp_toy.py [--variant {unbudgeted,budgeted,both}] [...]

The defaults solve both variants with seed 42, T = 14, repair-cost scale
s = 100, budget fraction phi = 0.3 for the budgeted variant, and scipy's
HiGHS for the stage LPs. Each variant writes
<output-dir>/<variant>/exact_dp_solution.json (default output dir:
exact_dp/results). Unless --no-sddip-comparison is given, the file also holds
a comparison with the saved toy SDDiP solution (see default_sddip_path).
exact_dp/run_local.sh and scripts/slurm/exact_dp.sh run these defaults, each
with its own output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_parameters import create_model_parameters
from network_data import create_network
from Pseudo_DFL_stage_opf import create_stagewise_opf


STATUS_UP = 0
STATUS_DOWN = 1
STATUS_PENDING = 2
STATUS_NAMES = {
    STATUS_UP: "operational",
    STATUS_DOWN: "failed_no_pending_repair",
    STATUS_PENDING: "failed_pending_repair",
}


def mask_to_bits(mask: int, n_components: int) -> np.ndarray:
    """Return little-endian component bits for an integer mask."""
    return np.fromiter(
        ((mask >> k) & 1 for k in range(n_components)),
        dtype=np.int8,
        count=n_components,
    )


def bits_to_mask(bits: Sequence[int]) -> int:
    """Encode little-endian component bits as an integer mask."""
    mask = 0
    for k, bit in enumerate(bits):
        if int(bit):
            mask |= 1 << k
    return mask


def decode_state(index: int, n_components: int) -> np.ndarray:
    """Decode a base-3 state index into component statuses."""
    statuses = np.empty(n_components, dtype=np.int8)
    value = int(index)
    for k in range(n_components):
        statuses[k] = value % 3
        value //= 3
    return statuses


def encode_state(statuses: Sequence[int]) -> int:
    """Encode component statuses as a base-3 state index."""
    value = 0
    multiplier = 1
    for status in statuses:
        status_int = int(status)
        if status_int not in STATUS_NAMES:
            raise ValueError(f"Invalid component status {status_int}")
        value += status_int * multiplier
        multiplier *= 3
    return value


def next_state_index(
    availability_mask: int,
    repair_mask: int,
    n_components: int,
) -> int:
    """Return the next pre-shock state after choosing repairs."""
    if repair_mask & availability_mask:
        raise ValueError("A repair was assigned to an operational component")
    statuses = np.empty(n_components, dtype=np.int8)
    for k in range(n_components):
        if availability_mask & (1 << k):
            statuses[k] = STATUS_UP
        elif repair_mask & (1 << k):
            statuses[k] = STATUS_PENDING
        else:
            statuses[k] = STATUS_DOWN
    return encode_state(statuses)


def effective_availability_distribution(
    statuses: np.ndarray,
    failure_probabilities: np.ndarray,
) -> List[Tuple[int, float]]:
    """
    Return the post-shock availability distribution of a pre-shock state.

    Only operational components are exposed to the failure shock. The failure
    bits of the other components do not change the outcome, so they are summed
    out analytically: a failed component stays failed, and a pending repair
    completes at this stage without being exposed to the shock, as in the
    availability transition of SDDiP_julia_bridge.py.

    Returns (availability_mask, probability) pairs, one per failure pattern of
    the operational components.
    """
    n_components = len(statuses)
    stochastic_up = [
        k for k in range(n_components) if statuses[k] == STATUS_UP
    ]
    repaired_mask = bits_to_mask(statuses == STATUS_PENDING)

    distribution: List[Tuple[int, float]] = []
    for failure_pattern in range(1 << len(stochastic_up)):
        availability_mask = repaired_mask
        probability = 1.0
        for local_bit, component in enumerate(stochastic_up):
            fails = (failure_pattern >> local_bit) & 1
            lam = float(failure_probabilities[component])
            if fails:
                probability *= lam
            else:
                probability *= 1.0 - lam
                availability_mask |= 1 << component
        distribution.append((availability_mask, probability))

    total_probability = math.fsum(prob for _, prob in distribution)
    if not math.isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=2e-12):
        raise RuntimeError(
            f"Transition probabilities sum to {total_probability:.16f}, not 1"
        )
    return distribution


class ExactToyDP:
    """Exact dynamic program for the seven-component toy system.

    Arrays use 0-based stages: values[t, s] is the optimal expected cost of
    stages t, ..., T-1 from pre-shock state s (values[T] = 0), and policy[t, a]
    is the optimal repair mask for post-shock availability mask a.
    """

    def __init__(
        self,
        *,
        seed: int,
        repair_base_scale: float,
        budget_fraction: Optional[float],
        lp_solver: str,
        horizon: int = 14,
    ) -> None:
        self.seed = int(seed)
        self.repair_base_scale = float(repair_base_scale)
        self.budget_fraction = (
            None if budget_fraction is None else float(budget_fraction)
        )
        self.lp_solver = str(lp_solver)
        self.horizon = int(horizon)

        self.network = create_network("toy")
        # 550 and 350 are the scale-1 GEN_REPAIR_BASE and LINE_REPAIR_BASE of config.py.
        self.params = create_model_parameters(
            self.network,
            seed=self.seed,
            t=self.horizon,
            tau=1,
            gen_repair_base=550.0 * self.repair_base_scale,
            line_repair_base=350.0 * self.repair_base_scale,
            repair_budget_fraction=self.budget_fraction,
        )

        if self.params.tau != 1:
            raise ValueError("This exact DP currently supports repair delay tau=1 only")
        self.n_generators = int(self.network.n_generators)
        self.n_lines = int(self.network.n_lines)
        self.n_components = self.n_generators + self.n_lines
        if self.n_components != 7:
            raise ValueError(
                "This implementation is intentionally restricted to the "
                f"seven-component toy network, got {self.n_components} components"
            )

        self.n_masks = 1 << self.n_components
        self.n_states = 3 ** self.n_components
        self.all_mask = self.n_masks - 1
        self.component_names = (
            [f"G{g + 1}" for g in range(self.n_generators)]
            + [f"L{e + 1}" for e in range(self.n_lines)]
        )
        self.repair_costs = np.concatenate(
            [
                np.asarray(self.params.gen_repair_cost, dtype=float),
                np.asarray(self.params.line_repair_cost, dtype=float),
            ]
        )
        self.failure_probabilities = np.vstack(
            [
                np.asarray(self.params.lambda_g, dtype=float),
                np.asarray(self.params.lambda_e, dtype=float),
            ]
        )

        self.statuses = np.vstack(
            [decode_state(state, self.n_components) for state in range(self.n_states)]
        )
        self.mask_bits = np.vstack(
            [mask_to_bits(mask, self.n_components) for mask in range(self.n_masks)]
        )
        self.repair_cost_by_mask = np.array(
            [
                float(np.dot(self.mask_bits[mask], self.repair_costs))
                for mask in range(self.n_masks)
            ],
            dtype=float,
        )
        self.next_state = np.full(
            (self.n_masks, self.n_masks), -1, dtype=np.int32
        )
        for availability_mask in range(self.n_masks):
            failed_mask = self.all_mask ^ availability_mask
            for repair_mask in range(self.n_masks):
                if repair_mask & ~failed_mask:
                    continue
                self.next_state[availability_mask, repair_mask] = next_state_index(
                    availability_mask, repair_mask, self.n_components
                )

        self.opf_cost = np.zeros(self.n_masks, dtype=float)
        self.opf_generation = np.zeros(
            (self.n_masks, self.n_generators), dtype=float
        )
        self.opf_load_served = np.zeros(
            (self.n_masks, self.network.n_loads), dtype=float
        )
        self.opf_runtime_seconds = 0.0
        self.feasible_repairs: List[List[int]] = []
        self.transition_cache: List[List[List[Tuple[int, float]]]] = []
        self.values = np.zeros((self.horizon + 1, self.n_states), dtype=float)
        self.postshock_values = np.zeros(
            (self.horizon, self.n_masks), dtype=float
        )
        self.policy = np.zeros((self.horizon, self.n_masks), dtype=np.int16)
        self.stage_backward_seconds = np.zeros(self.horizon, dtype=float)
        self.dp_runtime_seconds = 0.0
        self.total_runtime_seconds = 0.0
        self.action_evaluations = 0
        self.transition_contributions = 0

    @property
    def variant_name(self) -> str:
        return "unbudgeted" if self.budget_fraction is None else "budgeted"

    def _precompute_opf(self) -> None:
        """Solve the stage DC-OPF once per availability mask (2^7 = 128 LPs).

        The stage OPF cost depends only on the post-shock availability: loads
        and generator costs are the same at every stage, and repairs act from
        the next stage on. One table therefore serves all stages.
        """
        started = time.perf_counter()
        stage_opf = create_stagewise_opf(
            self.network, self.params, lp_method=self.lp_solver
        )
        for availability_mask in range(self.n_masks):
            bits = self.mask_bits[availability_mask]
            a_gen = bits[: self.n_generators]
            a_line = bits[self.n_generators :]
            cost, generation, served, feasible = stage_opf.solve(a_gen, a_line)
            if not feasible:
                raise RuntimeError(
                    "Stage OPF failed for availability mask "
                    f"{availability_mask:0{self.n_components}b}"
                )
            self.opf_cost[availability_mask] = float(cost)
            self.opf_generation[availability_mask, :] = generation
            self.opf_load_served[availability_mask, :] = served
        self.opf_runtime_seconds = time.perf_counter() - started

    def _precompute_feasible_repairs(self) -> None:
        budget = self.params.repair_budget_per_stage
        tolerance = 1e-9
        self.feasible_repairs = []
        for availability_mask in range(self.n_masks):
            failed_mask = self.all_mask ^ availability_mask
            masks: List[int] = []
            for repair_mask in range(self.n_masks):
                if repair_mask & ~failed_mask:
                    continue
                if (
                    budget is not None
                    and self.repair_cost_by_mask[repair_mask] > budget + tolerance
                ):
                    continue
                masks.append(repair_mask)
            if not masks:
                raise RuntimeError(
                    f"No feasible repair action for availability mask {availability_mask}"
                )
            self.feasible_repairs.append(masks)

    def _precompute_transitions(self) -> None:
        self.transition_cache = []
        for stage in range(self.horizon):
            lam = self.failure_probabilities[:, stage]
            stage_transitions = [
                effective_availability_distribution(statuses, lam)
                for statuses in self.statuses
            ]
            self.transition_cache.append(stage_transitions)

    def _run_backward_recursion(self) -> None:
        started = time.perf_counter()
        tolerance = 1e-10

        for stage in range(self.horizon - 1, -1, -1):
            stage_started = time.perf_counter()
            next_values = self.values[stage + 1]

            for availability_mask in range(self.n_masks):
                best_continuation = math.inf
                best_repair = 0
                best_repair_cost = math.inf
                for repair_mask in self.feasible_repairs[availability_mask]:
                    state_next = int(
                        self.next_state[availability_mask, repair_mask]
                    )
                    candidate = (
                        self.repair_cost_by_mask[repair_mask]
                        + next_values[state_next]
                    )
                    repair_cost = self.repair_cost_by_mask[repair_mask]
                    # Near-ties (within the tolerance) go to the cheaper repair
                    # set, then to the smaller mask index.
                    if (
                        candidate < best_continuation - tolerance
                        or (
                            abs(candidate - best_continuation) <= tolerance
                            and (
                                repair_cost < best_repair_cost - tolerance
                                or (
                                    abs(repair_cost - best_repair_cost) <= tolerance
                                    and repair_mask < best_repair
                                )
                            )
                        )
                    ):
                        best_continuation = candidate
                        best_repair = repair_mask
                        best_repair_cost = repair_cost
                    self.action_evaluations += 1

                self.policy[stage, availability_mask] = best_repair
                self.postshock_values[stage, availability_mask] = (
                    self.opf_cost[availability_mask] + best_continuation
                )

            for state in range(self.n_states):
                value = 0.0
                for availability_mask, probability in self.transition_cache[stage][state]:
                    value += (
                        probability
                        * self.postshock_values[stage, availability_mask]
                    )
                    self.transition_contributions += 1
                self.values[stage, state] = value

            self.stage_backward_seconds[stage] = (
                time.perf_counter() - stage_started
            )

        self.dp_runtime_seconds = time.perf_counter() - started

    def solve(self) -> Dict:
        """Solve the DP, evaluate its policy and return a JSON-serializable result.

        Raises RuntimeError if the backward and forward values differ by more
        than 1e-7 * max(1, |optimum|), or if the policy repairs an operational
        component or exceeds the repair budget.
        """
        total_started = time.perf_counter()
        self._precompute_opf()
        self._precompute_feasible_repairs()
        self._precompute_transitions()
        self._run_backward_recursion()
        forward = self.evaluate_optimal_policy_exactly()
        self.total_runtime_seconds = time.perf_counter() - total_started

        initial_state = encode_state([STATUS_UP] * self.n_components)
        optimum = float(self.values[0, initial_state])
        forward_total = float(forward["expected_total_cost"])
        agreement = abs(optimum - forward_total)
        agreement_tolerance = 1e-7 * max(1.0, abs(optimum))
        if agreement > agreement_tolerance:
            raise RuntimeError(
                "Backward and forward values disagree: "
                f"{optimum:.12f} versus {forward_total:.12f}"
            )

        for stage in range(self.horizon):
            for availability_mask in range(self.n_masks):
                repair_mask = int(self.policy[stage, availability_mask])
                if repair_mask & availability_mask:
                    raise RuntimeError("Policy repairs an operational component")
                budget = self.params.repair_budget_per_stage
                if (
                    budget is not None
                    and self.repair_cost_by_mask[repair_mask] > budget + 1e-7
                ):
                    raise RuntimeError("Policy violates the repair budget")

        return self._build_result(forward)

    def evaluate_optimal_policy_exactly(self) -> Dict:
        """Propagate the state distribution forward under the optimal policy.

        Every reachable state and failure outcome is enumerated with its exact
        probability (no sampling). Returns the expected OPF, repair and total
        costs and per-stage metrics; the total must equal the backward value.
        """
        initial_state = encode_state([STATUS_UP] * self.n_components)
        state_probabilities: Dict[int, float] = {initial_state: 1.0}
        stage_rows: List[Dict] = []
        expected_total = 0.0
        expected_opf_total = 0.0
        expected_repair_total = 0.0

        for stage in range(self.horizon):
            next_probabilities: Dict[int, float] = {}
            expected_opf = 0.0
            expected_repair = 0.0
            expected_generation = np.zeros(self.n_generators)
            expected_load_served = np.zeros(self.network.n_loads)
            expected_availability = np.zeros(self.n_components)
            expected_repairs = np.zeros(self.n_components)
            postshock_support = set()

            for state, state_probability in state_probabilities.items():
                for availability_mask, shock_probability in self.transition_cache[stage][state]:
                    mass = state_probability * shock_probability
                    if mass == 0.0:
                        continue
                    postshock_support.add(availability_mask)
                    repair_mask = int(self.policy[stage, availability_mask])
                    next_state = int(
                        self.next_state[availability_mask, repair_mask]
                    )
                    next_probabilities[next_state] = (
                        next_probabilities.get(next_state, 0.0) + mass
                    )
                    expected_opf += mass * self.opf_cost[availability_mask]
                    expected_repair += (
                        mass * self.repair_cost_by_mask[repair_mask]
                    )
                    expected_generation += (
                        mass * self.opf_generation[availability_mask]
                    )
                    expected_load_served += (
                        mass * self.opf_load_served[availability_mask]
                    )
                    expected_availability += (
                        mass * self.mask_bits[availability_mask]
                    )
                    expected_repairs += mass * self.mask_bits[repair_mask]

            probability_mass = math.fsum(next_probabilities.values())
            if not math.isclose(
                probability_mass, 1.0, rel_tol=0.0, abs_tol=5e-11
            ):
                raise RuntimeError(
                    f"Stage {stage + 1} probability mass is {probability_mass}"
                )

            expected_stage_total = expected_opf + expected_repair
            expected_total += expected_stage_total
            expected_opf_total += expected_opf
            expected_repair_total += expected_repair
            stage_rows.append(
                {
                    "stage": stage + 1,
                    "reachable_pre_shock_states": len(state_probabilities),
                    "reachable_post_shock_availability_masks": len(postshock_support),
                    "reachable_next_states": len(next_probabilities),
                    "probability_mass": probability_mass,
                    "expected_opf_cost": expected_opf,
                    "expected_repair_cost": expected_repair,
                    "expected_total_cost": expected_stage_total,
                    "expected_generation": expected_generation.tolist(),
                    "expected_load_served": expected_load_served.tolist(),
                    "expected_availability": expected_availability.tolist(),
                    "expected_repairs": expected_repairs.tolist(),
                    "expected_number_available": float(
                        np.sum(expected_availability)
                    ),
                    "expected_number_repairs": float(np.sum(expected_repairs)),
                }
            )
            state_probabilities = next_probabilities

        return {
            "expected_total_cost": expected_total,
            "expected_opf_cost": expected_opf_total,
            "expected_repair_cost": expected_repair_total,
            "terminal_reachable_states": len(state_probabilities),
            "terminal_probability_mass": math.fsum(state_probabilities.values()),
            "stage_metrics": stage_rows,
        }

    def _build_result(self, forward: Dict) -> Dict:
        initial_state = encode_state([STATUS_UP] * self.n_components)
        optimum = float(self.values[0, initial_state])
        budget = self.params.repair_budget_per_stage
        policy_rows = []
        for stage in range(self.horizon):
            stage_actions = []
            for availability_mask in range(self.n_masks):
                repair_mask = int(self.policy[stage, availability_mask])
                stage_actions.append(
                    {
                        "availability_mask": availability_mask,
                        "availability_bits": self.mask_bits[
                            availability_mask
                        ].astype(int).tolist(),
                        "repair_mask": repair_mask,
                        "repair_bits": self.mask_bits[repair_mask]
                        .astype(int)
                        .tolist(),
                        "repair_cost": float(
                            self.repair_cost_by_mask[repair_mask]
                        ),
                    }
                )
            policy_rows.append({"stage": stage + 1, "actions": stage_actions})

        opf_rows = []
        for availability_mask in range(self.n_masks):
            opf_rows.append(
                {
                    "availability_mask": availability_mask,
                    "availability_bits": self.mask_bits[
                        availability_mask
                    ].astype(int).tolist(),
                    "cost": float(self.opf_cost[availability_mask]),
                    "generation": self.opf_generation[
                        availability_mask
                    ].tolist(),
                    "load_served": self.opf_load_served[
                        availability_mask
                    ].tolist(),
                }
            )

        return {
            "schema": "exact_dp_toy_v1",
            "exactness_scope": (
                "Exact finite-horizon DP for the current toy binary-failure, "
                "tau=1 repair, and DC-OPF model; numerical LP tolerances apply."
            ),
            "model": {
                "network": "toy",
                "variant": self.variant_name,
                "seed": self.seed,
                "horizon": self.horizon,
                "repair_delay": self.params.tau,
                "repair_base_scale": self.repair_base_scale,
                "budget_fraction": self.budget_fraction,
                "budget_per_stage": budget,
                "n_generators": self.n_generators,
                "n_lines": self.n_lines,
                "n_components": self.n_components,
                "component_order": self.component_names,
                "generator_repair_costs": np.asarray(
                    self.params.gen_repair_cost
                ).tolist(),
                "line_repair_costs": np.asarray(
                    self.params.line_repair_cost
                ).tolist(),
                "generator_failure_probabilities": np.asarray(
                    self.params.lambda_g
                ).tolist(),
                "line_failure_probabilities": np.asarray(
                    self.params.lambda_e
                ).tolist(),
                "lp_solver": self.lp_solver,
            },
            "state_space": {
                "status_encoding": {
                    str(code): name for code, name in STATUS_NAMES.items()
                },
                "state_index_encoding": (
                    "base-3, component 0 is the least-significant digit"
                ),
                "mask_encoding": (
                    "binary, component 0 is the least-significant bit"
                ),
                "maximum_states_per_stage": self.n_states,
                "failure_outcomes_per_stage": self.n_masks,
            },
            "solution": {
                "exact_optimal_value": optimum,
                "initial_state_index": initial_state,
                "expected_opf_cost": forward["expected_opf_cost"],
                "expected_repair_cost": forward["expected_repair_cost"],
                "forward_recomputed_value": forward["expected_total_cost"],
                "backward_forward_absolute_difference": abs(
                    optimum - forward["expected_total_cost"]
                ),
            },
            "work": {
                "opf_problems_solved": self.n_masks,
                "action_evaluations": self.action_evaluations,
                "transition_probability_contributions": (
                    self.transition_contributions
                ),
                "opf_runtime_seconds": self.opf_runtime_seconds,
                "dp_runtime_seconds": self.dp_runtime_seconds,
                "total_runtime_seconds": self.total_runtime_seconds,
                "backward_seconds_by_stage": (
                    self.stage_backward_seconds.tolist()
                ),
            },
            # Constant flags: effective_availability_distribution() and solve()
            # raise RuntimeError before this point if any of these checks fails.
            "validation": {
                "all_transition_probabilities_sum_to_one": True,
                "all_repairs_apply_only_to_failed_components": True,
                "repair_budget_respected": True,
                "backward_forward_values_agree": True,
            },
            "forward_evaluation": forward,
            "optimal_policy": policy_rows,
            "opf_by_availability": opf_rows,
            "value_function_by_stage": self.values.tolist(),
        }


def _display_path(path: Path) -> str:
    """Return ``path`` relative to the repository root when it lies inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def reconstruct_saved_sddip_mean(saved: Dict, network, params) -> float:
    """
    Return the sample mean cost of the simulated SDDiP policy.

    sddip_solution.json stores per-stage means over the policy simulations
    (generation, load served and repair indicators). The stage cost is affine
    in these quantities, so evaluating it at the means gives the exact sample
    mean of the simulated total costs. The individual costs (500 simulations
    with the default --n-simulations of SDDiP_main.py) and their variance
    cannot be recovered. paper/reproduce_paper_numbers.py imports this function.
    """
    expected_generation = saved["expected_generation"]
    expected_load_served = saved["expected_load_served"]
    repairs = saved["repairs"]
    total = 0.0
    for stage in range(1, params.T + 1):
        stage_key = str(stage)
        for generator in range(network.n_generators):
            total += (
                float(network.gen_cost[generator])
                * float(expected_generation[stage_key][str(generator)])
            )
            total += (
                float(params.gen_repair_cost[generator])
                * float(repairs["gen"][stage_key][str(generator)])
            )
        served = math.fsum(
            float(expected_load_served[stage_key][str(load)])
            for load in range(network.n_loads)
        )
        total += float(params.load_shedding_cost) * (
            float(np.sum(network.load_nominal)) - served
        )
        for line in range(network.n_lines):
            total += (
                float(params.line_repair_cost[line])
                * float(repairs["line"][stage_key][str(line)])
            )
    return total


def compare_with_saved_sddip(
    result: Dict,
    solver: ExactToyDP,
    sddip_path: Optional[Path],
) -> Optional[Dict]:
    """Compare the exact DP optimum with a saved SDDiP solution (sddip_solution.json).

    Returns None when sddip_path is None, and a record with available=False when
    the file does not exist. The SDDiP file stores no seed, repair-cost scale or
    budget, so it is assumed to come from a run with the same settings.
    """
    if sddip_path is None:
        return None
    if not sddip_path.exists():
        return {
            "path": _display_path(sddip_path),
            "available": False,
            "message": "Saved SDDiP artifact not found",
        }

    with sddip_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    exact_value = float(result["solution"]["exact_optimal_value"])
    lower_bound = float(saved["lower_bound"])
    reconstructed_mean = reconstruct_saved_sddip_mean(
        saved, solver.network, solver.params
    )
    legacy_value = float(saved.get("upper_bound", math.nan))
    iterations = int(saved.get("iterations", 0))
    n_cuts = int(saved.get("n_cuts", 0))
    convergence = saved.get("convergence", {})
    training_forward_costs = convergence.get("upper_bounds", [])

    return {
        "path": _display_path(sddip_path),
        "available": True,
        "parameter_match_note": (
            "The saved SDDiP JSON does not contain complete parameter metadata. "
            "This comparison assumes its path, seed, repair scale, and budget "
            "correspond to the exact-DP run."
        ),
        "exact_dp_optimum": exact_value,
        "sddip_training_lower_bound": lower_bound,
        "lower_bound_is_below_exact_optimum": (
            lower_bound <= exact_value + 1e-6 * max(1.0, abs(exact_value))
        ),
        "exact_optimum_minus_lower_bound": exact_value - lower_bound,
        "relative_lower_bound_shortfall_vs_exact_optimum": (
            (exact_value - lower_bound) / exact_value
            if exact_value != 0.0
            else None
        ),
        "reconstructed_500_simulation_in_sample_mean": reconstructed_mean,
        "mean_minus_exact_optimum": reconstructed_mean - exact_value,
        "relative_mean_excess_vs_exact_optimum": (
            (reconstructed_mean - exact_value) / exact_value
            if exact_value != 0.0
            else None
        ),
        "estimated_mean_minus_lb_ratio": (
            (reconstructed_mean - lower_bound) / reconstructed_mean
            if reconstructed_mean != 0.0
            else None
        ),
        "legacy_reported_upper_bound_field": legacy_value,
        "legacy_field_interpretation": (
            "One training forward-pass cost; not a policy upper bound."
        ),
        "iterations": iterations,
        "cuts": n_cuts,
        "expected_cuts_for_one_per_nonterminal_stage": (
            (solver.horizon - 1) * iterations
        ),
        "training_forward_cost_count": len(training_forward_costs),
        "policy_mean_note": (
            "This is the exact sample mean of the 500 in-sample policy "
            "simulations reconstructed from saved mean quantities. Individual "
            "costs and their variance were not saved."
        ),
    }


def print_solution_summary(
    result: Dict,
    comparison: Optional[Dict],
    output_path: Path,
) -> None:
    model = result["model"]
    solution = result["solution"]
    work = result["work"]
    stages = result["forward_evaluation"]["stage_metrics"]

    print()
    print("=" * 100)
    print("EXACT DYNAMIC PROGRAM — TOY NETWORK")
    print("=" * 100)
    print(f"Variant                    : {model['variant']}")
    print(f"Seed                       : {model['seed']}")
    print(f"Horizon / repair delay     : {model['horizon']} / {model['repair_delay']}")
    print(f"Repair-base scale          : {model['repair_base_scale']:g}")
    budget = model["budget_per_stage"]
    budget_text = "none" if budget is None else "$" + f"{budget:,.2f}"
    print(f"Repair budget per stage    : {budget_text}")
    print(f"Maximum DP states/stage    : {result['state_space']['maximum_states_per_stage']:,}")
    print(f"Failure outcomes/stage     : {result['state_space']['failure_outcomes_per_stage']:,}")
    print("Exact optimal value        : $" + f"{solution['exact_optimal_value']:,.6f}")
    print("  Expected OPF cost        : $" + f"{solution['expected_opf_cost']:,.6f}")
    print("  Expected repair cost     : $" + f"{solution['expected_repair_cost']:,.6f}")
    print(
        "Backward/forward check     : "
        f"{solution['backward_forward_absolute_difference']:.3e}"
    )
    print(f"OPFs solved                : {work['opf_problems_solved']:,}")
    print(f"Repair actions evaluated   : {work['action_evaluations']:,}")
    print(
        "Transition contributions   : "
        f"{work['transition_probability_contributions']:,}"
    )
    print(f"OPF precomputation time    : {work['opf_runtime_seconds']:.3f}s")
    print(f"DP recursion time          : {work['dp_runtime_seconds']:.3f}s")
    print(f"Total time                 : {work['total_runtime_seconds']:.3f}s")

    print()
    print("Exact optimal-policy expectations by stage")
    print("-" * 100)
    print(
        " stage | pre-states | post-masks | next-states |"
        "      OPF cost | repair cost |   total cost | served MW | E[# repairs]"
    )
    print("-" * 100)
    for row in stages:
        print(
            f" {row['stage']:>5d} |"
            f" {row['reachable_pre_shock_states']:>10d} |"
            f" {row['reachable_post_shock_availability_masks']:>10d} |"
            f" {row['reachable_next_states']:>11d} |"
            f" {row['expected_opf_cost']:>13,.2f} |"
            f" {row['expected_repair_cost']:>11,.2f} |"
            f" {row['expected_total_cost']:>12,.2f} |"
            f" {sum(row['expected_load_served']):>9.2f} |"
            f" {row['expected_number_repairs']:>12.4f}"
        )

    if comparison is not None:
        print()
        print("Comparison with saved SDDiP artifact")
        print("-" * 100)
        if not comparison.get("available", False):
            print(f"Saved artifact             : {comparison['path']}")
            print(f"Status                     : {comparison['message']}")
        else:
            print(f"Saved artifact             : {comparison['path']}")
            print(
                "SDDiP training LB           : $"
                + f"{comparison['sddip_training_lower_bound']:,.6f}"
            )
            print(
                "Exact optimum - LB          : $"
                + f"{comparison['exact_optimum_minus_lower_bound']:,.6f}"
            )
            print(
                "Relative LB shortfall       : "
                f"{100.0 * comparison['relative_lower_bound_shortfall_vs_exact_optimum']:.4f}%"
            )
            print(
                "LB validity check           : "
                + (
                    "PASS (LB <= exact optimum)"
                    if comparison["lower_bound_is_below_exact_optimum"]
                    else "FAIL (LB exceeds exact optimum)"
                )
            )
            print(
                "Reconstructed 500-sim mean  : $"
                + f"{comparison['reconstructed_500_simulation_in_sample_mean']:,.6f}"
            )
            print(
                "Mean - exact optimum        : $"
                + f"{comparison['mean_minus_exact_optimum']:,.6f}"
            )
            print(
                "Relative mean excess        : "
                f"{100.0 * comparison['relative_mean_excess_vs_exact_optimum']:.4f}%"
            )
            print(
                "Legacy 'upper_bound' field  : $"
                + f"{comparison['legacy_reported_upper_bound_field']:,.6f} "
                + "(single training forward pass)"
            )
            print(
                "Iterations / cuts           : "
                f"{comparison['iterations']:,} / {comparison['cuts']:,}"
            )
            print(
                "Training forward costs read : "
                f"{comparison['training_forward_cost_count']:,}"
            )
            print("Caution                    : no variance/CI for the reconstructed mean")

    print()
    print(f"Full exact values and policy: {output_path}")
    print("=" * 100)


def default_sddip_path(variant: str) -> Path:
    """Return the saved toy SDDiP solution for a variant.

    The paths are fixed and match the defaults of this script (seed 42, T = 14,
    s = 100, phi = 0.3), which are the toy settings of
    scripts/slurm/{unbudgeted,budgeted}/sddip.sh. They do not follow --seed,
    --horizon, --repair-base-scale or --budget-fraction; --sddip-unbudgeted and
    --sddip-budgeted select other files.
    """
    if variant == "unbudgeted":
        return (
            REPO_ROOT
            / "experiments"
            / "paper_unbudgeted"
            / "toy"
            / "SDDiP"
            / "sddip_solution.json"
        )
    return (
        REPO_ROOT
        / "experiments"
        / "paper_budget0.3"
        / "toy"
        / "SDDiP"
        / "sddip_solution.json"
    )


def solve_variant(
    *,
    variant: str,
    args: argparse.Namespace,
    output_base: Path,
) -> Dict:
    """Solve one variant, write its JSON result and print a summary.

    The result goes to <output_base>/<variant>/exact_dp_solution.json and holds
    the SDDiP comparison unless --no-sddip-comparison is given.
    """
    budget_fraction = None if variant == "unbudgeted" else args.budget_fraction
    solver = ExactToyDP(
        seed=args.seed,
        repair_base_scale=args.repair_base_scale,
        budget_fraction=budget_fraction,
        lp_solver=args.lp_solver,
        horizon=args.horizon,
    )
    result = solver.solve()

    if args.no_sddip_comparison:
        comparison = None
    else:
        explicit = (
            args.sddip_unbudgeted
            if variant == "unbudgeted"
            else args.sddip_budgeted
        )
        sddip_path = Path(explicit).resolve() if explicit else default_sddip_path(variant)
        comparison = compare_with_saved_sddip(result, solver, sddip_path)
        result["saved_sddip_comparison"] = comparison

    variant_dir = output_base / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    output_path = variant_dir / "exact_dp_solution.json"
    result["run_environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "output_file": _display_path(output_path),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print_solution_summary(result, comparison, output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the seven-component toy stochastic repair model "
            "exactly by finite-horizon dynamic programming."
        )
    )
    parser.add_argument(
        "--variant",
        choices=("unbudgeted", "budgeted", "both"),
        default="both",
        help="Variant(s) to solve (default: both).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument(
        "--repair-base-scale",
        type=float,
        default=100.0,
        help="Repair-cost scale s of the toy network (default: 100).",
    )
    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.3,
        help="Per-stage repair-budget fraction for the budgeted variant.",
    )
    parser.add_argument(
        "--lp-solver",
        choices=("highs", "gurobi"),
        default="highs",
        help="Backend for the 128 tiny DC-OPFs (default: scipy HiGHS).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "results"),
    )
    parser.add_argument(
        "--sddip-unbudgeted",
        type=str,
        default=None,
        help="Optional unbudgeted SDDiP JSON to compare against.",
    )
    parser.add_argument(
        "--sddip-budgeted",
        type=str,
        default=None,
        help="Optional budgeted SDDiP JSON to compare against.",
    )
    parser.add_argument(
        "--no-sddip-comparison",
        action="store_true",
        help="Skip reading existing SDDiP result files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.repair_base_scale <= 0:
        raise ValueError("--repair-base-scale must be positive")
    if not (0.0 < args.budget_fraction <= 1.0):
        raise ValueError("--budget-fraction must lie in (0, 1]")

    output_base = Path(args.output_dir).expanduser().resolve()
    variants = (
        ["unbudgeted", "budgeted"]
        if args.variant == "both"
        else [args.variant]
    )

    print("=" * 100)
    print("Exact toy DP validation run")
    print(f"Repository root : {REPO_ROOT}")
    print(f"Output base     : {output_base}")
    print(f"Variants        : {', '.join(variants)}")
    print(f"LP solver       : {args.lp_solver}")
    print("=" * 100)

    results = {}
    for variant in variants:
        results[variant] = solve_variant(
            variant=variant,
            args=args,
            output_base=output_base,
        )

    if len(results) == 2:
        unbudgeted = results["unbudgeted"]["solution"]["exact_optimal_value"]
        budgeted = results["budgeted"]["solution"]["exact_optimal_value"]
        print()
        print("=" * 100)
        print("CROSS-VARIANT EXACT-DP COMPARISON")
        print("=" * 100)
        print("Unbudgeted exact optimum : $" + f"{unbudgeted:,.6f}")
        print("Budgeted exact optimum   : $" + f"{budgeted:,.6f}")
        print("Budget effect            : $" + f"{budgeted - unbudgeted:,.6f}")
        print(
            "Relative budget effect   : "
            f"{100.0 * (budgeted - unbudgeted) / unbudgeted:.4f}%"
        )
        print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
