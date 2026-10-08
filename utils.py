"""
Utility Functions Module
========================
Helpers for SAA solution dicts (used by main.py and visualization.py): save and
load solution JSON, line flows per stage, feasibility checks, per-stage
generation and load-shedding statistics, and expected repair counts.
"""

import numpy as np
import json
from pathlib import Path


def save_solution(solution, filename):
    """
    Save solution to JSON file.
    
    Parameters:
    -----------
    solution : dict
        Solution dictionary
    filename : str or Path
        Output filename
    """
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    solution_serializable = convert_types(solution)
    
    with open(filename, 'w') as f:
        json.dump(solution_serializable, f, indent=2)
    
    print(f"Solution saved to {filename}")


def load_solution(filename):
    """
    Load solution from JSON file.

    JSON object keys are strings, so all integer keys (scenario, stage and
    component indices) come back as str. The statistics and validation helpers
    below expect integer keys.

    Parameters:
    -----------
    filename : str or Path
        Input filename
    
    Returns:
    --------
    solution : dict
        Solution dictionary
    """
    with open(filename, 'r') as f:
        solution = json.load(f)
    
    print(f"Solution loaded from {filename}")
    return solution


def compute_power_flows(solution, network_data, scenario_id=0):
    """
    Line flows on each line, by stage: {stage: {line: flow}}.

    Prefers the explicit flow variables recorded in the solution. If those are
    absent, flows are reconstructed from voltage angles, in which case a failed
    line is reported with zero flow: an outaged line carries no power regardless
    of the angle difference across its decoupled endpoints. Returns an empty
    dict if the solution has neither flows nor angles for this scenario.

    Parameters:
    -----------
    solution : dict
        Solution dictionary
    network_data : NetworkData
        Network data
    scenario_id : int
        Scenario ID to compute flows for

    Returns:
    --------
    flows : dict
        Line flows by stage {stage: {line: flow}}
    """
    flows = {}

    # Preferred: explicit line-flow variables from the solved model.
    line_flows = solution.get('line_flows')
    if line_flows and scenario_id in line_flows:
        for t, per_line in line_flows[scenario_id].items():
            flows[t] = {e: per_line[e] for e in per_line}
        return flows

    if scenario_id not in solution.get('angles', {}):
        return flows

    # Fallback: reconstruct from angles, forcing failed lines to zero flow.
    avail = solution.get('availability', {}).get(scenario_id, {})
    a_line = avail.get('line') if isinstance(avail, dict) else None

    for t in solution['angles'][scenario_id].keys():
        flows[t] = {}
        angles = solution['angles'][scenario_id][t]

        for e in range(network_data.n_lines):
            available = True
            if a_line is not None:
                try:
                    available = bool(a_line[int(t) - 1][e])  # a_line: (T, n_lines)
                except (IndexError, TypeError, ValueError):
                    available = True

            if not available:
                flows[t][e] = 0.0
                continue

            from_bus, to_bus = network_data.line_connections[e]
            susceptance = network_data.line_susceptances[e]
            flows[t][e] = susceptance * (angles[from_bus] - angles[to_bus])

    return flows


def validate_solution(solution, network_data, model_params, scenarios, tolerance=1e-4):
    """
    Check a solution against generation, load-served and line-flow limits.

    Bounds are the nameplate ones (generation in [pmin, pmax], load served in
    [0, demand], flows within the line limits), not the availability-scaled
    bounds of the model, so dispatch on a failed generator is not flagged.
    Power balance is not checked; violations['power_balance'] stays empty.

    Parameters:
    -----------
    solution : dict
        Solution to validate (integer keys, as returned by SAASolver.solve)
    network_data : NetworkData
        Network data
    model_params : ModelParameters
        Not used
    scenarios : list
        Not used
    tolerance : float
        Numerical tolerance for constraint violations

    Returns:
    --------
    validation : dict
        'feasible' (bool), 'n_violations' (int) and 'violations' (lists of
        violation records keyed by constraint type)
    """
    violations = {
        'power_balance': [],
        'generation_limits': [],
        'load_limits': [],
        'line_limits': []
    }
    
    for s in solution['generation'].keys():
        for t in solution['generation'][s].keys():
            # Check generation limits
            for g, pg in solution['generation'][s][t].items():
                pmin = network_data.gen_pmin[g]
                pmax = network_data.gen_pmax[g]
                
                if pg < pmin - tolerance or pg > pmax + tolerance:
                    violations['generation_limits'].append({
                        'scenario': s,
                        'stage': t,
                        'generator': g,
                        'value': pg,
                        'bounds': (pmin, pmax)
                    })
            
            # Check load limits
            for l, dl in solution['load_served'][s][t].items():
                demand = network_data.load_nominal[l]
                
                if dl < -tolerance or dl > demand + tolerance:
                    violations['load_limits'].append({
                        'scenario': s,
                        'stage': t,
                        'load': l,
                        'value': dl,
                        'demand': demand
                    })
            
            # Check line flow limits
            flows = compute_power_flows(solution, network_data, scenario_id=s)
            if t in flows:
                for e, flow in flows[t].items():
                    fmin = network_data.line_flow_min[e]
                    fmax = network_data.line_flow_max[e]
                    
                    if flow < fmin - tolerance or flow > fmax + tolerance:
                        violations['line_limits'].append({
                            'scenario': s,
                            'stage': t,
                            'line': e,
                            'value': flow,
                            'bounds': (fmin, fmax)
                        })
    
    # Count violations
    n_violations = sum(len(v) for v in violations.values())
    
    validation = {
        'feasible': n_violations == 0,
        'n_violations': n_violations,
        'violations': violations
    }
    
    return validation


def print_validation_report(validation):
    """Print formatted validation report."""
    print("=" * 70)
    print("SOLUTION VALIDATION REPORT")
    print("=" * 70)
    
    if validation['feasible']:
        print("✓ Solution is FEASIBLE")
    else:
        print("✗ Solution has VIOLATIONS")
        print(f"  Total violations: {validation['n_violations']}")
        
        for constraint_type, viols in validation['violations'].items():
            if viols:
                print(f"\n  {constraint_type.replace('_', ' ').title()}:")
                for v in viols[:5]:  # Show first 5
                    print(f"    {v}")
                if len(viols) > 5:
                    print(f"    ... and {len(viols) - 5} more")
    
    print("=" * 70)


def compute_solution_statistics(solution, network_data, model_params):
    """
    Per-stage generation and load-shedding statistics and expected repair counts.

    Scenarios (leaf paths in tree mode) are weighted equally, which matches
    their probabilities.

    Parameters:
    -----------
    solution : dict
        Solution dictionary (integer keys, as returned by SAASolver.solve)
    network_data : NetworkData
        Network data
    model_params : ModelParameters
        Model parameters (only T is used)

    Returns:
    --------
    stats : dict
        'mean_generation', 'std_generation', 'mean_shedding', 'std_shedding':
        arrays of length T (MW), mean and std over scenarios of the stage total.
        'total_gen_repairs', 'total_line_repairs', 'total_repairs': expected
        number of repairs over the horizon.
    """
    n_scenarios = len(solution['generation'])
    T = model_params.T
    
    # Generation statistics
    total_gen = np.zeros((n_scenarios, T))
    for s in range(n_scenarios):
        for t in range(1, T + 1):
            total_gen[s, t-1] = sum(solution['generation'][s][t].values())
    
    # Load shedding statistics
    total_shedding = np.zeros((n_scenarios, T))
    for s in range(n_scenarios):
        for t in range(1, T + 1):
            for l in range(network_data.n_loads):
                demand = network_data.load_nominal[l]
                served = solution['load_served'][s][t][l]
                total_shedding[s, t-1] += (demand - served)
    
    # Expected repair counts: solution['repairs'] holds the per-stage mean of the
    # binary repair decisions, so the sum over stages and components is the
    # expected number of repairs over the horizon.
    expected_gen_repairs = sum(
        sum(r for r in repairs.values())
        for repairs in solution['repairs']['gen'].values()
    )
    expected_line_repairs = sum(
        sum(r for r in repairs.values())
        for repairs in solution['repairs']['line'].values()
    )

    stats = {
        'mean_generation': np.mean(total_gen, axis=0),
        'std_generation': np.std(total_gen, axis=0),
        'mean_shedding': np.mean(total_shedding, axis=0),
        'std_shedding': np.std(total_shedding, axis=0),
        'total_gen_repairs': expected_gen_repairs,
        'total_line_repairs': expected_line_repairs,
        'total_repairs': expected_gen_repairs + expected_line_repairs,
    }
    
    return stats


def print_statistics_report(stats, model_params):
    """Print formatted statistics report."""
    print("=" * 70)
    print("SOLUTION STATISTICS")
    print("=" * 70)
    
    print("\nGeneration by Stage (MW):")
    print("-" * 70)
    print(f"{'Stage':<10}{'Mean':<15}{'Std Dev':<15}")
    print("-" * 70)
    for t in range(model_params.T):
        print(f"{t+1:<10}{stats['mean_generation'][t]:<15.2f}{stats['std_generation'][t]:<15.2f}")
    
    print("\nLoad Shedding by Stage (MW):")
    print("-" * 70)
    print(f"{'Stage':<10}{'Mean':<15}{'Std Dev':<15}")
    print("-" * 70)
    for t in range(model_params.T):
        print(f"{t+1:<10}{stats['mean_shedding'][t]:<15.2f}{stats['std_shedding'][t]:<15.2f}")
    
    print("\nExpected Repair Summary:")
    print("-" * 70)
    print(f"  Generator repairs: {stats['total_gen_repairs']:.2f}")
    print(f"  Line repairs:      {stats['total_line_repairs']:.2f}")
    print(f"  Total repairs:     {stats['total_repairs']:.2f}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Self-test: round-trip a dict of numpy types through save/load.
    print("Testing utility functions...")
    
    test_dict = {
        'int': np.int64(42),
        'float': np.float64(3.14),
        'array': np.array([1, 2, 3]),
        'nested': {
            'value': np.float32(2.71)
        }
    }
    
    converted = save_solution(test_dict, '/tmp/test_solution.json')
    loaded = load_solution('/tmp/test_solution.json')
    
    print("Type conversion test passed!")
