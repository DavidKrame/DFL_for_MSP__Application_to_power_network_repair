"""
Model Parameters Module
=======================
Builds the model parameters for a given network:
- Time horizon T and repair delay tau
- Load-shedding cost and per-component repair costs
- Per-stage repair budget (when a budget fraction is set)
- Stage-dependent failure probabilities (log-linear logit model)

Defaults come from config.py; ModelParameters keyword arguments override them.
"""

import numpy as np

from config import (
    T, TAU, LOAD_SHEDDING_COST, GEN_REPAIR_BASE, LINE_REPAIR_BASE,
    FAILURE_K, FAILURE_T_STAR,
    FAILURE_CLASS_PROB,
    FAILURE_P_MAX_STD_LOW, FAILURE_P_MAX_STD_HIGH,
    FAILURE_P_MAX_ELV_LOW, FAILURE_P_MAX_ELV_HIGH,
    REPAIR_BUDGET_FRACTION,
)


class ModelParameters:
    """Container for all model parameters."""

    def __init__(self, network_data, seed=42,
                 t=T, tau=TAU,
                 load_shedding_cost=LOAD_SHEDDING_COST,
                 gen_repair_base=GEN_REPAIR_BASE,
                 line_repair_base=LINE_REPAIR_BASE,
                 failure_k=FAILURE_K, failure_t_star=FAILURE_T_STAR,
                 failure_class_prob=FAILURE_CLASS_PROB,
                 failure_p_max_std_low=FAILURE_P_MAX_STD_LOW,
                 failure_p_max_std_high=FAILURE_P_MAX_STD_HIGH,
                 failure_p_max_elv_low=FAILURE_P_MAX_ELV_LOW,
                 failure_p_max_elv_high=FAILURE_P_MAX_ELV_HIGH,
                 repair_budget_fraction=REPAIR_BUDGET_FRACTION):
        """
        Initialize model parameters.

        Parameters:
        -----------
        network_data : NetworkData
            Network topology and parameters
        seed : int
            Seed for the vulnerability-class and p_max draws (generators use
            seed, lines seed + 1)
        All other kwargs override the corresponding config.py constant.
        """
        self.seed = seed

        # Time parameters
        self.T   = t
        self.tau = tau

        # Cost parameters
        self.load_shedding_cost = load_shedding_cost

        # Repair costs: proportional to component capacity, scaled to a
        # network-independent reference level.
        self.gen_repair_cost  = self._repair_costs(
            network_data.gen_pmax, base=gen_repair_base)
        self.line_repair_cost = self._repair_costs(
            np.abs(network_data.line_flow_max), base=line_repair_base)

        # Per-stage monetary repair budget B^t ($), the same at every stage and
        # resolved to a single absolute value (or None) that every method reads.
        # It is a fraction phi of the network's total fleet repair cost, so the
        # same phi gives the same relative tightness on any network.
        # None ⇒ no budget.
        self.repair_budget_fraction = repair_budget_fraction
        if repair_budget_fraction is not None:
            total_repair_cost = float(np.sum(self.gen_repair_cost)
                                      + np.sum(self.line_repair_cost))
            self.repair_budget_per_stage = float(repair_budget_fraction) * total_repair_cost
        else:
            self.repair_budget_per_stage = None

        # Failure probability model parameters (log-linear logit).
        # delta = (t_star + K)*ln(t_star + K) - t_star must be > 0.
        # For t_star = 1 this requires (1+K)*ln(1+K) > 1, satisfied for K > ~0.76.
        # K = 1.5 gives delta ≈ 1.29 > 0.
        self.K      = failure_k
        self.t_star = failure_t_star

        # Store failure class params for use in _assign_p_max
        self._failure_class_prob      = failure_class_prob
        self._failure_p_max_std_low   = failure_p_max_std_low
        self._failure_p_max_std_high  = failure_p_max_std_high
        self._failure_p_max_elv_low   = failure_p_max_elv_low
        self._failure_p_max_elv_high  = failure_p_max_elv_high

        # Peak failure probabilities p_max[k] in (1/2, 1), one per component.
        # Components are randomly assigned to a vulnerability class using the
        # model seed, so the assignment is deterministic and ignores the
        # component's location and capacity (ranges are the config.py defaults):
        #   Standard : p_max ~ Uniform(0.55, 0.70)
        #   Elevated : p_max ~ Uniform(0.80, 0.90)
        self.p_max_gen  = self._assign_p_max(network_data.n_generators, offset=0)
        self.p_max_line = self._assign_p_max(network_data.n_lines,      offset=1)

        # Store network dimensions
        self.n_generators = network_data.n_generators
        self.n_lines = network_data.n_lines

        # Compute failure probabilities for all stages
        self._compute_failure_probabilities()

    # ------------------------------------------------------------------
    # Parameter generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_costs(capacities, base):
        """Return repair costs proportional to component capacity.

        Each cost is ``base * cap / mean(cap)``, so the mean cost equals
        ``base`` regardless of the number or size of components.
        """
        caps = np.asarray(capacities, dtype=float)
        mean = caps.mean()
        return (base * caps / mean) if mean > 0 else np.full(len(caps), base)

    def _assign_p_max(self, n_comp, offset):
        """Assign peak failure probabilities using a seeded RNG.

        Each component is independently assigned to the Standard class
        (p_max ~ Uniform[0.55, 0.70] by default) with probability
        failure_class_prob (default 0.5), otherwise to the Elevated class
        (p_max ~ Uniform[0.80, 0.90] by default).  The RNG is seeded with
        ``self.seed + offset`` so generators and lines get different but
        reproducible assignments.
        """
        rng   = np.random.default_rng(self.seed + offset)
        p_max = np.empty(n_comp)
        for k in range(n_comp):
            if rng.random() < self._failure_class_prob:   # Standard class
                p_max[k] = rng.uniform(self._failure_p_max_std_low, self._failure_p_max_std_high)
            else:                                          # Elevated class
                p_max[k] = rng.uniform(self._failure_p_max_elv_low, self._failure_p_max_elv_high)
        return p_max

    def _compute_lambda(self, p_max_arr):
        """
        Compute failure probability curves for a set of components.

        For component k and stage t in {1,...,T}:

            lambda^t[k] = 1 / (1 + (t + K)^{-A[k]} * exp(B[k] * t))

        where A[k] and B[k] are derived from the peak condition at t_star and
        the peak-value condition at p_max[k]:

            delta  = (t_star + K) * ln(t_star + K) - t_star
            B[k]   = ln(p_max[k] / (1 - p_max[k])) / delta
            A[k]   = (t_star + K) * B[k]

        Parameters:
        -----------
        p_max_arr : np.ndarray, shape (n_comp,)
            Peak failure probability for each component; each entry in (1/2, 1).

        Returns:
        --------
        lam : np.ndarray, shape (n_comp, T)
            lam[k, t-1] = lambda^t[k]
        """
        n_comp = len(p_max_arr)
        stages = np.arange(1, self.T + 1, dtype=float)  # t = 1, ..., T

        delta = (self.t_star + self.K) * np.log(self.t_star + self.K) - self.t_star
        if delta <= 0:
            raise ValueError(
                f"Model parameter K={self.K} with t_star={self.t_star} yields "
                f"delta={delta:.6f} <= 0. The failure probability model requires "
                f"(t_star + K)*ln(t_star + K) > t_star. Increase K."
            )
        logit_pmax = np.log(p_max_arr / (1.0 - p_max_arr))   # shape (n_comp,)
        B = logit_pmax / delta                                 # shape (n_comp,)
        A = (self.t_star + self.K) * B                        # shape (n_comp,)

        # Broadcast: A and B are (n_comp,), stages is (T,)
        # lam[k, t-1] = 1 / (1 + (t + K)^{-A[k]} * exp(B[k] * t))
        t = stages[None, :]                      # (1, T)
        exponent = -A[:, None] * np.log(t + self.K) + B[:, None] * t  # (n_comp, T)
        lam = 1.0 / (1.0 + np.exp(exponent))    # numerically stable via log form

        return lam

    def _compute_failure_probabilities(self):
        """Compute failure probabilities for all components and all stages."""
        self.lambda_g = self._compute_lambda(self.p_max_gen)   # shape (n_generators, T)
        self.lambda_e = self._compute_lambda(self.p_max_line)  # shape (n_lines, T)

    def get_failure_probability(self, component_type, component_idx, stage):
        """
        Get failure probability for a specific component at a specific stage.

        Parameters:
        -----------
        component_type : str
            'generator' or 'line'
        component_idx : int
            Component index (0-indexed)
        stage : int
            Stage number (1-indexed)

        Returns:
        --------
        prob : float
            Failure probability
        """
        if component_type == 'generator':
            return self.lambda_g[component_idx, stage - 1]
        elif component_type == 'line':
            return self.lambda_e[component_idx, stage - 1]
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def print_parameters(self):
        """Print formatted model parameters."""
        print("=" * 60)
        print("MODEL PARAMETERS")
        print("=" * 60)
        print(f"Time horizon:           T = {self.T} stages")
        print(f"Repair delay:           τ = {self.tau} stages")
        print(f"Load shedding cost:     α = ${self.load_shedding_cost:.2f}/MWh")
        print(f"Random seed:            {self.seed}")
        print()

        print("Repair Costs:")
        print("-" * 60)
        for g in range(self.n_generators):
            print(f"  Generator {g+1}: ${self.gen_repair_cost[g]:.2f}")
        for e in range(self.n_lines):
            print(f"  Line {e+1}:      ${self.line_repair_cost[e]:.2f}")
        print()

        print("Failure Probability Model (log-linear logit):")
        print("-" * 60)
        print(f"  Global shift:         K      = {self.K}  (δ > 0 needs K > ~0.763 when t* = 1)")
        print(f"  Peak stage:           t*     = {self.t_star}  (same for all components)")
        delta = (self.t_star + self.K) * np.log(self.t_star + self.K) - self.t_star
        print(f"  Derived delta:        δ      = {delta:.4f}")
        print()

        print("Component Peak Probabilities p_max[k]:")
        print("-" * 60)
        for g in range(self.n_generators):
            print(f"  Generator {g+1}: p_max = {self.p_max_gen[g]:.2f}  "
                  f"(λ at t=1: {self.lambda_g[g, 0]:.4f})")
        for e in range(self.n_lines):
            print(f"  Line {e+1}:      p_max = {self.p_max_line[e]:.2f}  "
                  f"(λ at t=1: {self.lambda_e[e, 0]:.4f})")
        print()

        print("Failure Probabilities by Stage:")
        print("-" * 60)
        for t in range(1, self.T + 1):
            print(f"  Stage {t}:")
            print(f"    Generators: ", end="")
            for g in range(self.n_generators):
                prob = self.lambda_g[g, t-1]
                print(f"G{g+1}={prob:.4f}  ", end="")
            print()
            print(f"    Lines:      ", end="")
            for e in range(self.n_lines):
                prob = self.lambda_e[e, t-1]
                print(f"L{e+1}={prob:.4f}  ", end="")
            print()
        print("=" * 60)


def create_model_parameters(network_data, seed=42, **kwargs):
    """Create a ModelParameters; kwargs are forwarded and override config defaults."""
    return ModelParameters(network_data, seed=seed, **kwargs)


if __name__ == "__main__":
    # Print the parameters of the toy (5-bus) network.
    from network_data import create_network

    network = create_network()
    params = create_model_parameters(network)
    params.print_parameters()
