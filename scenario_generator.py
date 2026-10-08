"""
Scenario Generator Module
=========================
Generates failure scenarios for generators and lines from the stage-dependent
failure probabilities of model_parameters.py: component k is hit by a failure
event at stage t with probability lambda^t[k] (lambda_g, lambda_e),
independently across components and stages.

Each scenario is a realization of failure events across all stages. The module
also builds sampled scenario trees for the tree-based SAA.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Scenario:
    """
    Container for a single failure scenario.
    
    Attributes:
    -----------
    scenario_id : int
        Identifier for this scenario (part of the RNG seed in
        generate_single_scenario)
    gen_failures : np.ndarray
        Binary failure events for generators, shape (T, n_generators); row t-1
        holds stage t. xi_g^t[g] = 1 if a failure event hits generator g at
        stage t; it takes the generator down only if it was available.
    line_failures : np.ndarray
        Binary failure events for lines, shape (T, n_lines), same convention:
        xi_e^t[e] = 1 if a failure event hits line e at stage t
    probability : float
        Weight of this scenario in the scenario-flat MIP of
        problem_formulation.py; generate_scenarios sets it to 1/n_scenarios
    """
    scenario_id: int
    gen_failures: np.ndarray
    line_failures: np.ndarray
    probability: float = 1.0


@dataclass
class TreeNode:
    """
    A single node of a sampled scenario tree.

    Attributes
    ----------
    node_id : int
        Unique id; 0 is always the root.
    stage : int
        0 for the root (pre-observation), 1..T for decision nodes.
    parent : int or None
        Parent node id (None only for the root).
    xi_gen, xi_line : np.ndarray
        Failure realization observed on the edge into this node, shapes
        (n_generators,) / (n_lines,). All zeros for the root.
    cond_prob : float
        Probability of this node conditional on its parent.
    path_prob : float
        Probability of the root->node path (product of cond_prob along it).
    children : list[int]
        Child node ids.
    """
    node_id: int
    stage: int
    parent: 'int | None'
    xi_gen: np.ndarray
    xi_line: np.ndarray
    cond_prob: float
    path_prob: float
    children: list


@dataclass
class ScenarioTree:
    """
    A sampled scenario tree that enforces non-anticipativity by construction:
    each decision node carries one decision, shared by every sample path
    through that node.

    The tree branches at each stage by `branching` sampled failure outcomes, so
    two leaves that share an ancestor at stage t share the same history
    xi^1..xi^t and therefore the same stage-t decision. Decision nodes are those
    with stage >= 1; the root (stage 0) only holds the all-available prior state.
    """
    nodes: list
    branching: int
    T: int

    @property
    def decision_nodes(self):
        """Node ids with a decision attached (stages 1..T)."""
        return [n.node_id for n in self.nodes if n.stage >= 1]

    @property
    def leaves(self):
        """Leaf node ids (stage == T)."""
        return [n.node_id for n in self.nodes if n.stage == self.T]

    def node(self, node_id):
        return self.nodes[node_id]


class ScenarioGenerator:
    """Sample failure scenarios and scenario trees from params.lambda_g / lambda_e."""
    
    def __init__(self, model_params, seed=None):
        """
        Initialize the scenario generator.
        
        Parameters:
        -----------
        model_params : ModelParameters
            Model parameters including failure probabilities
        seed : int, optional
            Random seed for reproducibility
        """
        self.params = model_params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate_single_scenario(self, scenario_id=0):
        """
        Generate a single failure scenario with per-scenario seeding.

        Each scenario has its own RNG seeded with [seed, scenario_id], so it is
        reproducible and independent of how many other scenarios are generated:
        scenario i is the same whatever total count is requested. This lets
        Pseudo_DFL_data.py extend a cached training set by generating only the
        new ids. With seed=None the RNG draws fresh OS entropy on every call.

        Parameters:
        -----------
        scenario_id : int
            Identifier for this scenario (used as part of the RNG seed).

        Returns:
        --------
        scenario : Scenario
            Generated failure scenario
        """
        seed = [self.seed, scenario_id] if self.seed is not None else None
        rng  = np.random.default_rng(seed)

        # lambda_g has shape (n_gen, T); .T gives (T, n_gen).
        gen_failures  = rng.binomial(1, self.params.lambda_g.T).astype(int)
        line_failures = rng.binomial(1, self.params.lambda_e.T).astype(int)

        return Scenario(
            scenario_id=scenario_id,
            gen_failures=gen_failures,
            line_failures=line_failures,
            probability=1.0  # generate_scenarios overwrites this with 1/N
        )

    def generate_scenarios(self, n_scenarios, start_id=0):
        """
        Generate multiple failure scenarios.

        Parameters:
        -----------
        n_scenarios : int
            Number of scenarios to generate.
        start_id : int
            ID assigned to the first scenario; subsequent scenarios receive
            start_id+1, start_id+2, ... Default 0.

        Returns:
        --------
        scenarios : List[Scenario]
            List of generated scenarios with probability = 1/n_scenarios.
        """
        prob = 1.0 / n_scenarios
        scenarios = []
        for s in range(n_scenarios):
            sc = self.generate_single_scenario(scenario_id=start_id + s)
            sc.probability = prob
            scenarios.append(sc)
        return scenarios

    def generate_scenario_tree(self, branching, T=None):
        """
        Build a sampled scenario tree by nested conditional sampling.

        Starting from an all-available root, each node spawns `branching`
        children whose incoming edge carries an independently sampled per-stage
        failure outcome (Bernoulli with the stage's lambda). Children are given
        equal conditional weight 1/branching, so the path probability of a
        stage-t node is 1/branching**t and the leaf probabilities form a uniform
        sample average over branching**T paths.

        All sample paths through a node share its history, so the decision
        attached to the node is common to all of them. A flat set of independent
        scenarios cannot express this non-anticipativity.

        Parameters
        ----------
        branching : int
            Number of children per node (branching factor B). Must be >= 1.
        T : int, optional
            Horizon; defaults to params.T.

        Returns
        -------
        ScenarioTree
            Tree with (branching**1 + ... + branching**T) decision nodes plus
            the root.
        """
        if branching < 1:
            raise ValueError(f"branching must be >= 1, got {branching}")
        if T is None:
            T = self.params.T

        n_gen  = self.params.lambda_g.shape[0]
        n_line = self.params.lambda_e.shape[0]

        # One RNG keyed on (seed, branching, T). With a fixed seed the tree is
        # reproducible, since the loop below samples in a fixed breadth-first order.
        tree_seed = [self.seed, branching, T] if self.seed is not None else None
        rng = np.random.default_rng(tree_seed)

        root = TreeNode(
            node_id=0, stage=0, parent=None,
            xi_gen=np.zeros(n_gen, dtype=int),
            xi_line=np.zeros(n_line, dtype=int),
            cond_prob=1.0, path_prob=1.0, children=[],
        )
        nodes = [root]
        frontier = [0]

        for t in range(1, T + 1):
            lam_g = self.params.lambda_g[:, t - 1]
            lam_e = self.params.lambda_e[:, t - 1]
            next_frontier = []
            for parent_id in frontier:
                parent = nodes[parent_id]
                for _ in range(branching):
                    xi_g = rng.binomial(1, lam_g).astype(int)
                    xi_e = rng.binomial(1, lam_e).astype(int)
                    child = TreeNode(
                        node_id=len(nodes), stage=t, parent=parent_id,
                        xi_gen=xi_g, xi_line=xi_e,
                        cond_prob=1.0 / branching,
                        path_prob=parent.path_prob / branching,
                        children=[],
                    )
                    parent.children.append(child.node_id)
                    nodes.append(child)
                    next_frontier.append(child.node_id)
            frontier = next_frontier

        return ScenarioTree(nodes=nodes, branching=branching, T=T)

    def print_scenario_summary(self, scenarios):
        """Print summary statistics for generated scenarios."""
        if not scenarios:
            print("No scenarios generated.")
            return
        
        n_scenarios = len(scenarios)
        T = self.params.T
        n_gen = self.params.n_generators
        n_lines = self.params.n_lines
        
        print("=" * 60)
        print("SCENARIO SUMMARY")
        print("=" * 60)
        print(f"Number of scenarios: {n_scenarios}")
        print()
        
        # Compute empirical failure frequencies
        print("Empirical Failure Frequencies (across scenarios):")
        print("-" * 60)
        
        for t in range(T):
            gen_freq = np.mean([s.gen_failures[t, :] for s in scenarios], axis=0)
            line_freq = np.mean([s.line_failures[t, :] for s in scenarios], axis=0)
            
            print(f"  Stage {t+1}:")
            print(f"    Generators: ", end="")
            for g in range(n_gen):
                print(f"G{g+1}={gen_freq[g]:.3f}  ", end="")
            print()
            print(f"    Lines:      ", end="")
            for e in range(n_lines):
                print(f"L{e+1}={line_freq[e]:.3f}  ", end="")
            print()
        
        print()
        
        # Compare with theoretical probabilities
        print("Comparison: Empirical vs Theoretical Probabilities:")
        print("-" * 60)
        for t in range(T):
            gen_freq = np.mean([s.gen_failures[t, :] for s in scenarios], axis=0)
            line_freq = np.mean([s.line_failures[t, :] for s in scenarios], axis=0)
            
            print(f"  Stage {t+1}:")
            for g in range(n_gen):
                emp = gen_freq[g]
                theo = self.params.lambda_g[g, t]
                print(f"    Gen {g+1}:  Empirical={emp:.4f}, Theoretical={theo:.4f}, "
                      f"Error={abs(emp-theo):.4f}")
            for e in range(n_lines):
                emp = line_freq[e]
                theo = self.params.lambda_e[e, t]
                print(f"    Line {e+1}: Empirical={emp:.4f}, Theoretical={theo:.4f}, "
                      f"Error={abs(emp-theo):.4f}")
        
        print("=" * 60)
    
    def get_scenario_statistics(self, scenarios):
        """
        Compute statistics about generated scenarios.
        
        Returns:
        --------
        stats : dict
            n_scenarios and the mean, max and min number of failure events per
            scenario for generators, lines and both together
        """
        if not scenarios:
            return {}
        
        n_scenarios = len(scenarios)
        T = self.params.T
        
        # Failure events (xi = 1) per scenario, including events that hit a
        # component that is already down
        total_gen_failures = [np.sum(s.gen_failures) for s in scenarios]
        total_line_failures = [np.sum(s.line_failures) for s in scenarios]
        total_failures = [g + l for g, l in zip(total_gen_failures, total_line_failures)]
        
        return {
            'n_scenarios': n_scenarios,
            'mean_gen_failures': np.mean(total_gen_failures),
            'mean_line_failures': np.mean(total_line_failures),
            'mean_total_failures': np.mean(total_failures),
            'max_gen_failures': np.max(total_gen_failures),
            'max_line_failures': np.max(total_line_failures),
            'max_total_failures': np.max(total_failures),
            'min_gen_failures': np.min(total_gen_failures),
            'min_line_failures': np.min(total_line_failures),
            'min_total_failures': np.min(total_failures),
        }


def create_scenario_generator(model_params, seed=None):
    """Factory function to create scenario generator."""
    return ScenarioGenerator(model_params, seed=seed)


if __name__ == "__main__":
    # Quick check on the toy network: empirical vs. model failure frequencies
    from network_data import create_network
    from model_parameters import create_model_parameters
    
    print("Testing Scenario Generator")
    print()
    
    network = create_network()
    params = create_model_parameters(network, seed=42)
    
    gen = create_scenario_generator(params, seed=42)
    scenarios = gen.generate_scenarios(n_scenarios=100)
    
    gen.print_scenario_summary(scenarios)
    
    stats = gen.get_scenario_statistics(scenarios)
    print()
    print("Overall Statistics:")
    print("-" * 60)
    for key, value in stats.items():
        print(f"{key:30s}: {value:.2f}")
