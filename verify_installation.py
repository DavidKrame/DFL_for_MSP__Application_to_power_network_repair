"""
Installation check
==================
Quick checks on the toy (5-bus) network: imports the core modules, builds the
network, model parameters, scenarios, availability trajectories and the
extensive-form MIP (without solving it), and lists the available Pyomo MIP
solvers and gurobipy. The checks assume the config.py defaults T = 14 and
TAU = 1. Exits with status 1 if any check fails.

Usage:
    python verify_installation.py
"""

import sys
import traceback


def test_imports():
    """Test that the core modules import (DFL and SDDiP modules are not checked)."""
    print("=" * 70)
    print("TESTING MODULE IMPORTS")
    print("=" * 70)
    
    modules = [
        'network_data',
        'model_parameters',
        'scenario_generator',
        'state_dynamics',
        'problem_formulation',
        'saa_solver',
        'utils',
        'visualization'
    ]
    
    all_passed = True
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:30s} - OK")
        except Exception as e:
            print(f"✗ {module_name:30s} - FAILED")
            print(f"  Error: {e}")
            all_passed = False
    
    return all_passed


def test_network_data():
    """Test the toy network's dimensions and matrix shapes."""
    print("\n" + "=" * 70)
    print("TESTING NETWORK DATA")
    print("=" * 70)
    
    try:
        from network_data import create_network
        
        network = create_network()
        
        # Check basic properties
        assert network.n_buses == 5, "Wrong number of buses"
        assert network.n_lines == 5, "Wrong number of lines"
        assert network.n_generators == 2, "Wrong number of generators"
        assert network.n_loads == 3, "Wrong number of loads"
        
        # Check matrices
        assert network.C.shape == (5, 5), "Wrong incidence matrix shape"
        assert network.B.shape == (5, 5), "Wrong susceptance matrix shape"
        
        print("✓ Network data creation - OK")
        return True
        
    except Exception as e:
        print(f"✗ Network data creation - FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_model_parameters():
    """Test model parameters creation (expects T = 14 and TAU = 1)."""
    print("\n" + "=" * 70)
    print("TESTING MODEL PARAMETERS")
    print("=" * 70)
    
    try:
        from network_data import create_network
        from model_parameters import create_model_parameters
        
        network = create_network()
        params = create_model_parameters(network, seed=42)
        
        # Check basic properties
        assert params.T == 14, "Wrong time horizon"
        assert params.tau == 1, "Wrong repair delay"
        assert params.K > 0, "K must be positive"

        # Check failure probabilities
        assert params.lambda_g.shape == (2, 14), "Wrong generator failure prob shape"
        assert params.lambda_e.shape == (5, 14), "Wrong line failure prob shape"
        
        # Check probability bounds
        assert (params.lambda_g >= 0).all() and (params.lambda_g <= 1).all(), \
            "Failure probabilities out of bounds"
        
        print("✓ Model parameters creation - OK")
        return True
        
    except Exception as e:
        print(f"✗ Model parameters creation - FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_scenario_generation():
    """Test scenario generation: array shapes and equal probabilities 1/N."""
    print("\n" + "=" * 70)
    print("TESTING SCENARIO GENERATION")
    print("=" * 70)
    
    try:
        from network_data import create_network
        from model_parameters import create_model_parameters
        from scenario_generator import create_scenario_generator
        
        network = create_network()
        params = create_model_parameters(network, seed=42)
        gen = create_scenario_generator(params, seed=42)
        
        # Generate scenarios
        scenarios = gen.generate_scenarios(n_scenarios=10)
        
        # Check scenarios
        assert len(scenarios) == 10, "Wrong number of scenarios"
        
        for s in scenarios:
            assert s.gen_failures.shape == (14, 2), "Wrong gen failures shape"
            assert s.line_failures.shape == (14, 5), "Wrong line failures shape"
            assert s.probability == 0.1, "Wrong scenario probability"
        
        print("✓ Scenario generation - OK")
        return True
        
    except Exception as e:
        print(f"✗ Scenario generation - FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_state_dynamics():
    """Test the availability trajectories of one scenario without repairs."""
    print("\n" + "=" * 70)
    print("TESTING STATE DYNAMICS")
    print("=" * 70)
    
    try:
        from network_data import create_network
        from model_parameters import create_model_parameters
        from scenario_generator import create_scenario_generator
        from state_dynamics import create_state_dynamics
        
        network = create_network()
        params = create_model_parameters(network, seed=42)
        gen = create_scenario_generator(params, seed=42)
        scenarios = gen.generate_scenarios(n_scenarios=5)
        
        state_dyn = create_state_dynamics(network, params)
        
        # Compute availability
        availability = state_dyn.compute_availability_trajectories(scenarios[0])
        
        # Check availability
        assert 'gen' in availability, "Missing generator availability"
        assert 'line' in availability, "Missing line availability"
        assert availability['gen'].shape == (14, 2), "Wrong gen availability shape"
        assert availability['line'].shape == (14, 5), "Wrong line availability shape"

        # Initial availability reflects stage-1 failure draws: values in {0, 1}
        assert set(availability['gen'][0, :].tolist()).issubset({0, 1}), \
            "Initial gen availability must be binary"
        assert set(availability['line'][0, :].tolist()).issubset({0, 1}), \
            "Initial line availability must be binary"
        
        print("✓ State dynamics - OK")
        return True
        
    except Exception as e:
        print(f"✗ State dynamics - FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_problem_formulation():
    """Test that the scenario-flat MIP (EDOPFProblem) builds for 3 scenarios (no solve)."""
    print("\n" + "=" * 70)
    print("TESTING PROBLEM FORMULATION")
    print("=" * 70)
    
    try:
        from network_data import create_network
        from model_parameters import create_model_parameters
        from scenario_generator import create_scenario_generator
        from state_dynamics import create_state_dynamics
        from problem_formulation import create_problem
        
        network = create_network()
        params = create_model_parameters(network, seed=42)
        gen = create_scenario_generator(params, seed=42)
        scenarios = gen.generate_scenarios(n_scenarios=3)
        state_dyn = create_state_dynamics(network, params)
        
        # Create problem
        problem = create_problem(network, params, scenarios, state_dyn)
        problem.build_model()
        
        # Check model was built
        assert problem.model is not None, "Model not built"
        assert problem.model.nvariables() > 0, "No variables in model"
        assert problem.model.nconstraints() > 0, "No constraints in model"
        
        print("✓ Problem formulation - OK")
        print(f"  Variables: {problem.model.nvariables()}")
        print(f"  Constraints: {problem.model.nconstraints()}")
        return True
        
    except Exception as e:
        print(f"✗ Problem formulation - FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def test_solver_availability():
    """Test which Pyomo MIP solvers are available; fails if none is."""
    print("\n" + "=" * 70)
    print("TESTING SOLVER AVAILABILITY")
    print("=" * 70)
    
    import pyomo.environ  # noqa: F401  (side effect: registers solver plugins)
    from pyomo.opt import SolverFactory

    # Print the active selection from solver_config and probe every registered
    # MIP solver; the literal list is the fallback if solver_config fails to import.
    try:
        import solver_config
        print(solver_config.describe())
        solvers_to_test = sorted(solver_config.MIP_SOLVERS)
    except Exception:
        solvers_to_test = ['appsi_highs', 'highs', 'glpk', 'cbc', 'gurobi',
                            'gurobi_direct', 'appsi_gurobi', 'cplex']
    print()

    available_solvers = []

    for solver_name in solvers_to_test:
        try:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"✓ {solver_name:15s} - AVAILABLE")
                available_solvers.append(solver_name)
            else:
                print(f"  {solver_name:15s} - Not available")
        except Exception:
            print(f"  {solver_name:15s} - Not available")

    # gurobipy is used directly by the per-stage LP path (Layer C) and by the
    # in-memory Pyomo interfaces (gurobi_direct, appsi_gurobi), so it is reported
    # separately from the Pyomo probe. Its absence does not fail the check.
    try:
        import gurobipy as gp
        print(f"✓ {'gurobipy':15s} - AVAILABLE (Gurobi "
              f"{'.'.join(map(str, gp.gurobi.version()))})")
    except Exception:
        print(f"  {'gurobipy':15s} - Not available (fine unless using Gurobi)")

    if not available_solvers:
        print("\n⚠ WARNING: No solvers available!")
        print("  Please install at least one solver (HiGHS through highspy is recommended)")
        return False
    else:
        print(f"\n✓ Found {len(available_solvers)} available solver(s)")
        return True


def run_all_tests():
    """Run all checks, print a summary and return True if every check passed."""
    print("\n" + "=" * 70)
    print("INSTALLATION VERIFICATION")
    print("=" * 70)
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Network Data", test_network_data),
        ("Model Parameters", test_model_parameters),
        ("Scenario Generation", test_scenario_generation),
        ("State Dynamics", test_state_dynamics),
        ("Problem Formulation", test_problem_formulation),
        ("Solver Availability", test_solver_availability),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} - UNEXPECTED ERROR")
            print(f"  {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Installation is working correctly.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
