"""
Group 3: State Dynamics
=======================
Checks that component availability evolves correctly under the repair
pipeline model with delay tau (config.TAU = 1).

Tests
-----
8.  Initial availability: a^1[k] = 1 - xi^1[k] for every component k
9.  Repair delay: after a repair decision at stage t_repair the component is
    still failed at t_repair and available at t_repair + tau
10. Persistence: an available component with no new failure stays available
    throughout all subsequent stages
11. Availability is always in {0, 1}, never fractional or negative
"""

import numpy as np
import pytest

from scenario_generator import Scenario, create_scenario_generator


# ---------------------------------------------------------------------------
# Local scenario-builder helpers (plain functions, not pytest fixtures)
# ---------------------------------------------------------------------------

def zero_scenario(params):
    """A scenario with no failures at any stage."""
    T = params.T
    return Scenario(
        scenario_id=0,
        gen_failures=np.zeros((T, params.n_generators), dtype=int),
        line_failures=np.zeros((T, params.n_lines),      dtype=int),
    )


def single_failure_scenario(params, component, idx, stage_0indexed):
    """
    A scenario where exactly one component fails at one stage; no other failures.

    Parameters
    ----------
    component : 'gen' or 'line'
    idx : int   component index (0-based)
    stage_0indexed : int   stage index (0-based; stage 1 == 0)
    """
    T      = params.T
    gen_f  = np.zeros((T, params.n_generators), dtype=int)
    line_f = np.zeros((T, params.n_lines),      dtype=int)
    if component == "gen":
        gen_f[stage_0indexed, idx] = 1
    else:
        line_f[stage_0indexed, idx] = 1
    return Scenario(0, gen_f, line_f)


# ===========================================================================
# Test 8: Initial availability  a^1[k] = 1 - xi^1[k]
# ===========================================================================

class TestInitialAvailability:
    """Stage-1 availability is determined solely by stage-1 failures."""

    def test_no_failure_at_stage_1_all_components_up(self, dynamics, params):
        scenario = zero_scenario(params)
        avail    = dynamics.compute_availability_trajectories(scenario)
        assert np.all(avail["gen"][0, :]  == 1), (
            "All generators should be available at stage 1 when there are no failures"
        )
        assert np.all(avail["line"][0, :] == 1), (
            "All lines should be available at stage 1 when there are no failures"
        )

    def test_gen_failure_at_stage_1_makes_it_unavailable(self, dynamics, params):
        for g in range(params.n_generators):
            scenario = single_failure_scenario(params, "gen", g, stage_0indexed=0)
            avail    = dynamics.compute_availability_trajectories(scenario)
            assert avail["gen"][0, g] == 0, (
                f"Generator {g} should be unavailable at stage 1 after failure"
            )
            # All others must still be up
            for other in range(params.n_generators):
                if other != g:
                    assert avail["gen"][0, other] == 1, (
                        f"Generator {other} should be unaffected at stage 1"
                    )

    def test_line_failure_at_stage_1_makes_it_unavailable(self, dynamics, params):
        for e in range(params.n_lines):
            scenario = single_failure_scenario(params, "line", e, stage_0indexed=0)
            avail    = dynamics.compute_availability_trajectories(scenario)
            assert avail["line"][0, e] == 0, (
                f"Line {e} should be unavailable at stage 1 after failure"
            )

    def test_all_components_fail_at_stage_1(self, dynamics, params):
        T     = params.T
        gen_f = np.zeros((T, params.n_generators), dtype=int)
        lin_f = np.zeros((T, params.n_lines),      dtype=int)
        gen_f[0, :] = 1   # every generator fails at stage 1
        lin_f[0, :] = 1   # every line fails at stage 1
        scenario = Scenario(0, gen_f, lin_f)
        avail    = dynamics.compute_availability_trajectories(scenario)
        assert np.all(avail["gen"][0, :]  == 0), "All generators must be down at stage 1"
        assert np.all(avail["line"][0, :] == 0), "All lines must be down at stage 1"


# ===========================================================================
# Test 9: Repair delay
# ===========================================================================

class TestRepairDelay:
    """
    After a repair decision at stage t_repair (0-indexed):
      - the component is still failed at t_repair itself (the repair has only
        entered the pipeline; it has no instantaneous effect);
      - the component is available at t_repair + tau.

    With tau = 1, the repair completes one stage after it is ordered.
    """

    def _repair_gen(self, params, g):
        """Return a repair_decisions dict that repairs generator g at t = 0 (stage 1)."""
        rd = np.zeros(params.n_generators, dtype=int)
        rd[g] = 1
        return {0: {"gen": rd, "line": np.zeros(params.n_lines, dtype=int)}}

    def _repair_line(self, params, e, t_repair):
        """Return a repair_decisions dict that repairs line e at t_repair."""
        rd = np.zeros(params.n_lines, dtype=int)
        rd[e] = 1
        return {t_repair: {"gen": np.zeros(params.n_generators, dtype=int), "line": rd}}

    # --- Generator repair ---

    def test_gen_still_failed_at_repair_stage(self, dynamics, params):
        """Component remains failed at the stage repair is initiated (no instant fix)."""
        tau      = params.tau
        scenario = single_failure_scenario(params, "gen", 0, stage_0indexed=0)
        avail    = dynamics.compute_availability_trajectories(
            scenario, self._repair_gen(params, 0)
        )
        # t = 0 (stage 1): the generator has failed and its repair is only
        # ordered at this stage, so it is still down.
        assert avail["gen"][0, 0] == 0, (
            "Generator 0 should still be failed at the stage repair is initiated"
        )

    def test_gen_available_exactly_at_t_plus_tau(self, dynamics, params):
        """Component is available tau stages after repair is initiated."""
        tau      = params.tau
        scenario = single_failure_scenario(params, "gen", 0, stage_0indexed=0)
        avail    = dynamics.compute_availability_trajectories(
            scenario, self._repair_gen(params, 0)
        )
        assert avail["gen"][tau, 0] == 1, (
            f"Generator 0 should be available at t={tau} (= t_repair + tau = 0 + {tau})"
        )

    def test_gen_repair_at_later_stage_completes_correctly(self, dynamics, params):
        """
        Generator 1 fails at stage 1. Its repair is ordered at stage 3 (t=2, 0-indexed).
        It should be failed at t=2 and available at t = 2 + tau.
        """
        tau      = params.tau
        t_repair = 2
        scenario = single_failure_scenario(params, "gen", 1, stage_0indexed=0)
        rd = np.zeros(params.n_generators, dtype=int)
        rd[1] = 1
        repair_decisions = {
            t_repair: {"gen": rd, "line": np.zeros(params.n_lines, dtype=int)}
        }
        avail = dynamics.compute_availability_trajectories(scenario, repair_decisions)

        # Still failed at the repair stage itself
        assert avail["gen"][t_repair, 1] == 0, (
            f"Generator 1 should still be failed at t={t_repair} (repair just started)"
        )
        # Available tau stages later
        assert avail["gen"][t_repair + tau, 1] == 1, (
            f"Generator 1 should be available at t={t_repair + tau}"
        )

    # --- Line repair ---

    def test_line_repair_delay_respected(self, dynamics, params):
        """Line 3 fails at stage 1; repair ordered at stage 2 (t=1). Available at t = 1 + tau."""
        tau      = params.tau
        t_repair = 1
        scenario = single_failure_scenario(params, "line", 3, stage_0indexed=0)
        avail    = dynamics.compute_availability_trajectories(
            scenario, self._repair_line(params, 3, t_repair)
        )
        assert avail["line"][t_repair, 3] == 0, (
            f"Line 3 should still be failed at t={t_repair} (repair just started)"
        )
        assert avail["line"][t_repair + tau, 3] == 1, (
            f"Line 3 should be available at t={t_repair + tau}"
        )

    def test_no_repair_component_stays_failed(self, dynamics, params):
        """Without a repair decision, a failed component remains failed indefinitely."""
        scenario = single_failure_scenario(params, "gen", 0, stage_0indexed=0)
        avail    = dynamics.compute_availability_trajectories(scenario)   # no repairs
        # Failed at every stage: no repair is ordered
        for t in range(params.T):
            assert avail["gen"][t, 0] == 0, (
                f"Generator 0 should remain failed at t={t} with no repair decision"
            )


# ===========================================================================
# Test 10: Persistence
# ===========================================================================

class TestPersistence:
    """An available component that does not fail stays available."""

    def test_no_failures_all_stages_all_up(self, dynamics, params):
        """Zero-failure scenario: every component stays available for all T stages."""
        scenario = zero_scenario(params)
        avail    = dynamics.compute_availability_trajectories(scenario)
        assert np.all(avail["gen"]  == 1), "All generators should stay up with no failures"
        assert np.all(avail["line"] == 1), "All lines should stay up with no failures"

    def test_repaired_component_stays_up(self, dynamics, params):
        """
        Generator 0 fails at stage 1 and its repair is ordered at t_repair = 0;
        no further failures occur. It should be available at every stage from
        t_repair + tau onward.
        """
        tau      = params.tau
        scenario = single_failure_scenario(params, "gen", 0, stage_0indexed=0)
        rd = np.zeros(params.n_generators, dtype=int)
        rd[0] = 1
        repair_decisions = {0: {"gen": rd, "line": np.zeros(params.n_lines, dtype=int)}}
        avail = dynamics.compute_availability_trajectories(scenario, repair_decisions)

        for t in range(tau, params.T):
            assert avail["gen"][t, 0] == 1, (
                f"Generator 0 should remain available at t={t} after repair (no new failures)"
            )


# ===========================================================================
# Test 11: Availability always binary
# ===========================================================================

class TestAvailabilityBinary:
    """Availability values are always exactly 0 or 1, never fractional."""

    N_SCENARIOS = 30

    def test_availability_is_binary_across_scenarios(self, dynamics, params):
        gen       = create_scenario_generator(params, seed=17)
        scenarios = gen.generate_scenarios(self.N_SCENARIOS)
        for i, s in enumerate(scenarios):
            avail = dynamics.compute_availability_trajectories(s)
            gen_unique  = np.unique(avail["gen"])
            line_unique = np.unique(avail["line"])
            assert set(gen_unique).issubset({0, 1}), (
                f"Scenario {i}: generator availability has non-binary values: {gen_unique}"
            )
            assert set(line_unique).issubset({0, 1}), (
                f"Scenario {i}: line availability has non-binary values: {line_unique}"
            )
