"""The budget certificate must distinguish unique and nonunique checkpoints."""

import pytest
from gurobipy import GRB

from diagnose_valid_inequalities import checkpoint_diagnostics
from utils import load_gurobi_env


@pytest.fixture(scope="module")
def solver_env():
    with load_gurobi_env() as env:
        yield env


@pytest.mark.parametrize("second_cost,unique,distance", [(2, True, 0), (1, False, 2)])
def test_checkpoint_uniqueness(solver_env, second_cost, unique, distance):
    instance = {
        "nodes": [0, 1, 2], "budget": 1,
        "edges": [
            {"tail": 0, "head": 1, "checkpoint_cost": 1},
            {"tail": 1, "head": 2, "checkpoint_cost": second_cost},
        ],
        "intruders": [{"id": 0, "source": 0, "target": 2}],
        "known_feasible_checkpoints": [[0, 1]],
    }
    result = checkpoint_diagnostics(instance, solver_env)
    assert result["minimum_cost_status"] == result["max_distance_status"] == GRB.OPTIMAL
    assert result["minimum_cost_lp"] == pytest.approx(1)
    assert result["max_certificate_distance"] == pytest.approx(distance)
    assert result["unique_checkpoint_vector"] is unique
