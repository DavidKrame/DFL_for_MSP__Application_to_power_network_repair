"""
Shared fixtures for the ED-OPF test suite.

All fixtures are session-scoped (built once per test run) except
`large_sample`, which is module-scoped (built once in each test module that
requests it).
"""

import sys
import os

# Ensure the project root is importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from network_data import create_network
from model_parameters import create_model_parameters
from scenario_generator import create_scenario_generator
from state_dynamics import create_state_dynamics
from Pseudo_DFL_stage_opf import create_stagewise_opf


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (they solve small MIPs). "
        "Deselect with: pytest -m 'not slow'",
    )


# ---------------------------------------------------------------------------
# Core shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def network():
    """Toy (5-bus) network."""
    return create_network()


@pytest.fixture(scope="session")
def params(network):
    """Model parameters (failure probabilities, costs) with seed 42 and the
    config.py defaults T=14, tau=1."""
    return create_model_parameters(network)


@pytest.fixture(scope="session")
def dynamics(network, params):
    """State dynamics (availability transitions with repair delay tau)."""
    return create_state_dynamics(network, params)


@pytest.fixture(scope="session")
def opf(network, params):
    """Single-stage DC-OPF LP (StagewiseOPF) on the solver_config.LP_SOLVER
    backend, scipy HiGHS by default."""
    return create_stagewise_opf(network, params)


# ---------------------------------------------------------------------------
# Large scenario sample (law-of-large-numbers tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def large_sample(params):
    """
    10,000 scenarios drawn with seed 42. Module-scoped, so the sampling runs
    once per test module that requests the fixture.
    """
    gen = create_scenario_generator(params, seed=42)
    return gen.generate_scenarios(10_000)


