"""
State Dynamics Module
=====================
Handles the evolution of component availability states based on:
- Failure events (stochastic)
- Repair decisions (decision variables)
- Repair delays: a repair ordered at stage t completes at stage t + tau

State transitions follow the model, starting from a^0[k] = 1 for all k:
    a^t[k] = 0 if a^{t-1}[k] = 1 and xi^t[k] = 1  (new failure)
    a^t[k] = 1 if a^{t-1}[k] = 0 and r^{t-tau}[k] = 1  (repair completed)
    a^t[k] = a^{t-1}[k] otherwise  (no change)

The MIPs in problem_formulation.py encode the same transition as linear
constraints; this module evaluates it for given failures and repair orders.
"""

import numpy as np
from typing import Dict, List


class ComponentState:
    """Track the state of a single component across stages."""
    
    def __init__(self, component_id, component_type):
        """
        Initialize component state.
        
        Parameters:
        -----------
        component_id : int
            Component identifier
        component_type : str
            'generator' or 'line'
        """
        self.component_id = component_id
        self.component_type = component_type
        self.availability_history = []  # a^t values
        self.repair_history = []  # r^t values
        self.failure_history = []  # xi^t values


class StateDynamics:
    """Manage state dynamics for all components across all scenarios."""
    
    def __init__(self, network_data, model_params):
        """
        Initialize state dynamics.
        
        Parameters:
        -----------
        network_data : NetworkData
            Network topology and parameters
        model_params : ModelParameters
            Model parameters including repair delay tau
        """
        self.network = network_data
        self.params = model_params
        self.T = model_params.T
        self.tau = model_params.tau
        
    def compute_availability_trajectories(self, scenario, repair_decisions=None):
        """
        Compute component availability trajectories for a scenario.
        
        Parameters:
        -----------
        scenario : Scenario
            Failure scenario with xi_g^t and xi_e^t
        repair_decisions : dict, optional
            Repair decisions {stage: {'gen': array, 'line': array}} with
            0-indexed stage keys and 0/1 arrays of length n_gen / n_lines.
            A repair ordered at stage t completes at stage t + tau and restores
            the component if it was down at stage t + tau - 1.
            If None, no repairs are initiated
        
        Returns:
        --------
        availability : dict
            Dictionary with availability trajectories:
            {
                'gen': np.ndarray of shape (T, n_gen),
                'line': np.ndarray of shape (T, n_lines)
            }
        """
        n_gen = self.network.n_generators
        n_lines = self.network.n_lines
        
        # Initialize repair tracking
        # repair_initiated[t, k] = 1 if repair was initiated at stage t for component k
        repair_initiated_gen = np.zeros((self.T, n_gen), dtype=int)
        repair_initiated_line = np.zeros((self.T, n_lines), dtype=int)

        if repair_decisions is not None:
            for t in range(self.T):
                if t in repair_decisions:
                    repair_initiated_gen[t, :] = repair_decisions[t]['gen']
                    repair_initiated_line[t, :] = repair_decisions[t]['line']

        # Stage 1 availability: a^1[k] = 1 - xi^1[k].
        # The pre-horizon state is all-operational (a^0[k] = 1 for all k).
        # Applying the state transition at t=1 with empty repair pipeline gives
        # a^1[k] = (1 - xi^1[k]) * 1 + 0 = 1 - xi^1[k].
        a_gen = np.zeros((self.T, n_gen), dtype=int)
        a_line = np.zeros((self.T, n_lines), dtype=int)
        for g in range(n_gen):
            a_gen[0, g] = 1 - scenario.gen_failures[0, g]
        for e in range(n_lines):
            a_line[0, e] = 1 - scenario.line_failures[0, e]

        # Compute availability for each stage t = 2, ..., T
        for t in range(1, self.T):  # t is 0-indexed (corresponds to stages 2, ..., T)
            # Update generator availability
            for g in range(n_gen):
                a_prev = a_gen[t-1, g]
                xi_t = scenario.gen_failures[t, g]
                
                # Check if repair is completed
                repair_completed = False
                if t >= self.tau:
                    # Check if repair was initiated tau stages ago
                    repair_completed = (repair_initiated_gen[t - self.tau, g] == 1)
                
                # Apply state transition rules
                if a_prev == 1 and xi_t == 1:
                    # New failure occurs
                    a_gen[t, g] = 0
                elif a_prev == 0 and repair_completed:
                    # Repair is completed
                    a_gen[t, g] = 1
                else:
                    # No change
                    a_gen[t, g] = a_prev
            
            # Update line availability
            for e in range(n_lines):
                a_prev = a_line[t-1, e]
                xi_t = scenario.line_failures[t, e]
                
                # Check if repair is completed
                repair_completed = False
                if t >= self.tau:
                    # Check if repair was initiated tau stages ago
                    repair_completed = (repair_initiated_line[t - self.tau, e] == 1)
                
                # Apply state transition rules
                if a_prev == 1 and xi_t == 1:
                    # New failure occurs
                    a_line[t, e] = 0
                elif a_prev == 0 and repair_completed:
                    # Repair is completed
                    a_line[t, e] = 1
                else:
                    # No change
                    a_line[t, e] = a_prev
        
        return {
            'gen': a_gen,
            'line': a_line
        }
    
    def is_repair_feasible(self, availability, component_type, component_idx, stage):
        """
        Check if repair is feasible for a component at a stage.
        
        Repair is only feasible if the component is currently failed.
        
        Parameters:
        -----------
        availability : dict
            Availability trajectories
        component_type : str
            'generator' or 'line'
        component_idx : int
            Component index (0-indexed)
        stage : int
            Stage number (0-indexed)
        
        Returns:
        --------
        feasible : bool
            True if repair is feasible
        """
        if component_type == 'generator':
            return availability['gen'][stage, component_idx] == 0
        elif component_type == 'line':
            return availability['line'][stage, component_idx] == 0
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def print_availability_trajectory(self, availability, scenario_id=0):
        """Print formatted availability trajectory."""
        print("=" * 80)
        print(f"AVAILABILITY TRAJECTORY - Scenario {scenario_id}")
        print("=" * 80)
        
        n_gen = self.network.n_generators
        n_lines = self.network.n_lines
        
        print(f"{'Stage':<8}", end="")
        for g in range(n_gen):
            print(f"Gen{g+1:<4}", end="")
        print("  |  ", end="")
        for e in range(n_lines):
            print(f"Line{e+1:<4}", end="")
        print()
        print("-" * 80)
        
        for t in range(self.T):
            print(f"{t+1:<8}", end="")
            for g in range(n_gen):
                status = "UP" if availability['gen'][t, g] == 1 else "DN"
                print(f"{status:<6}", end="")
            print("  |  ", end="")
            for e in range(n_lines):
                status = "UP" if availability['line'][t, e] == 1 else "DN"
                print(f"{status:<6}", end="")
            print()
        
        print("=" * 80)
        print("UP = Available, DN = Failed")
        print()
    
    def compute_availability_statistics(self, scenarios):
        """
        Compute availability statistics across scenarios, without repairs.
        
        Parameters:
        -----------
        scenarios : List[Scenario]
            List of scenarios
        
        Returns:
        --------
        stats : dict
            Mean availability per stage and component, shapes (T, n_gen) and
            (T, n_lines), plus the per-scenario trajectories, shapes
            (n_scenarios, T, n_gen) and (n_scenarios, T, n_lines)
        """
        n_scenarios = len(scenarios)
        n_gen = self.network.n_generators
        n_lines = self.network.n_lines
        
        # Compute availability for each scenario (no repairs)
        gen_availability = np.zeros((n_scenarios, self.T, n_gen))
        line_availability = np.zeros((n_scenarios, self.T, n_lines))
        
        for s, scenario in enumerate(scenarios):
            avail = self.compute_availability_trajectories(scenario)
            gen_availability[s, :, :] = avail['gen']
            line_availability[s, :, :] = avail['line']
        
        # Compute mean availability
        mean_gen_avail = np.mean(gen_availability, axis=0)
        mean_line_avail = np.mean(line_availability, axis=0)
        
        return {
            'mean_gen_availability': mean_gen_avail,
            'mean_line_availability': mean_line_avail,
            'gen_availability_by_scenario': gen_availability,
            'line_availability_by_scenario': line_availability,
        }


def create_state_dynamics(network_data, model_params):
    """Factory function to create state dynamics."""
    return StateDynamics(network_data, model_params)


if __name__ == "__main__":
    # Quick check on the toy network
    from network_data import create_network
    from model_parameters import create_model_parameters
    from scenario_generator import create_scenario_generator
    
    print("Testing State Dynamics")
    print()
    
    network = create_network()
    params = create_model_parameters(network, seed=42)
    state_dynamics = create_state_dynamics(network, params)
    
    # Five scenarios; the trajectories below use the first one
    gen = create_scenario_generator(params, seed=42)
    scenarios = gen.generate_scenarios(n_scenarios=5)
    
    # Compute availability for first scenario (no repairs)
    print("Scenario 0: No Repairs")
    availability = state_dynamics.compute_availability_trajectories(scenarios[0])
    state_dynamics.print_availability_trajectory(availability, scenario_id=0)
    
    # Compute availability with some repairs
    print("\nScenario 0: With Repairs at Stage 1")
    repair_decisions = {
        0: {  # Stage 1 (0-indexed)
            'gen': np.array([1, 0]),  # Repair generator 1
            'line': np.array([0, 1, 0, 0, 0])  # Repair line 2
        }
    }
    availability_repaired = state_dynamics.compute_availability_trajectories(
        scenarios[0], repair_decisions
    )
    state_dynamics.print_availability_trajectory(availability_repaired, scenario_id=0)
    
    # Compute statistics
    print("\nAvailability Statistics Across Scenarios:")
    stats = state_dynamics.compute_availability_statistics(scenarios)
    print("-" * 80)
    print("Mean Generator Availability by Stage:")
    for t in range(params.T):
        print(f"  Stage {t+1}: ", end="")
        for g in range(network.n_generators):
            print(f"Gen{g+1}={stats['mean_gen_availability'][t, g]:.3f}  ", end="")
        print()
    print()
    print("Mean Line Availability by Stage:")
    for t in range(params.T):
        print(f"  Stage {t+1}: ", end="")
        for e in range(network.n_lines):
            print(f"Line{e+1}={stats['mean_line_availability'][t, e]:.3f}  ", end="")
        print()
