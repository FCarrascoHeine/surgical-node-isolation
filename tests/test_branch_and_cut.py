import itertools
import math
from pathlib import Path

import pytest
from gurobipy import GurobiError

from branch_and_cut import directed_min_cut, separate_solution
from formulations import solve_instance
from instances import prepare_instance
from utils import load_gurobi_env
from validation import validate_cut

SMALL_INSTANCE = Path(__file__).resolve().parents[1] / "instances" / "small_instance.json"


@pytest.fixture(scope="module")
def solver_env():
    try:
        env = load_gurobi_env()
    except GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required: {error}")
    yield env
    env.dispose()


def _zero_master_values(data):
    x = {edge: 0.0 for edge in data["edges"]}
    alpha = {
        (journeyer, edge): 0.0
        for journeyer in data["journeyers"]
        for edge in data["edges"]
    }
    phi = {journeyer: 0.0 for journeyer in data["journeyers"]}
    return x, alpha, phi


def test_directed_min_cut_handles_antiparallel_edges():
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    capacities = {(0, 1): 0.25, (1, 0): 1.0, (1, 2): 1.0, (2, 1): 1.0}
    value, cut_set, cut_edges = directed_min_cut(
        nodes, edges, capacities, source=0, target=2
    )
    assert math.isclose(value, 0.25, abs_tol=1e-9)
    assert cut_set == {0}
    assert cut_edges == [(0, 1)]


def test_zero_solution_generates_valid_cut_families(solver_env):
    data = prepare_instance(SMALL_INSTANCE)
    x, alpha, phi = _zero_master_values(data)
    cuts = separate_solution(data, x, alpha, phi, env=solver_env)
    families = {cut["family"] for cut in cuts}
    assert "intruder" in families
    assert "optimality" in families
    for cut in cuts:
        result = validate_cut(
            data["instance"],
            cut,
            x_values=x,
            alpha_values=alpha,
            phi_values=phi,
        )
        assert result["valid"], result["errors"]


def test_closed_edges_generate_journeyer_feasibility_cuts():
    data = prepare_instance(SMALL_INSTANCE)
    x = {edge: 1.0 for edge in data["edges"]}
    alpha = {
        (journeyer, edge): 0.0
        for journeyer in data["journeyers"]
        for edge in data["edges"]
    }
    phi = {journeyer: 0.0 for journeyer in data["journeyers"]}
    cuts = separate_solution(data, x, alpha, phi)
    feasibility = [cut for cut in cuts if cut["family"] == "feasibility"]
    assert len(feasibility) == len(data["journeyers"])


def test_feasibility_cut_is_valid_for_every_binary_capacity_state():
    data = prepare_instance(SMALL_INSTANCE)
    x = {edge: 1.0 for edge in data["edges"]}
    alpha = {
        (journeyer, edge): 0.0
        for journeyer in data["journeyers"]
        for edge in data["edges"]
    }
    phi = {journeyer: 0.0 for journeyer in data["journeyers"]}
    cut = next(
        cut
        for cut in separate_solution(data, x, alpha, phi)
        if cut["family"] == "feasibility" and cut["agent"] == 0
    )

    for state_values in itertools.product([0, 1, 2], repeat=len(data["edges"])):
        states = dict(zip(data["edges"], state_values))
        capacities = {
            edge: float(states[edge] in (0, 2)) for edge in data["edges"]
        }
        value, _, _ = directed_min_cut(
            data["nodes"],
            data["edges"],
            capacities,
            data["journeyer_source"][0],
            data["journeyer_target"][0],
        )
        if value >= 1 - 1e-9:
            assert sum(capacities[edge] for edge in cut["cut_edges"]) >= 1


def test_final_branch_and_cut_solution_has_no_violations(solver_env):
    result = solve_instance(
        SMALL_INSTANCE, formulation=4, relax=False, env=solver_env
    )
    variables = result["variables"]
    cuts = separate_solution(
        SMALL_INSTANCE,
        variables["x"],
        variables["alpha"],
        variables["phi"],
        env=solver_env,
    )
    assert result["status_name"] == "OPTIMAL"
    assert result["separation_complete"]
    assert cuts == []


def test_relaxation_iteration_limit_is_reported(solver_env):
    result = solve_instance(
        SMALL_INSTANCE,
        formulation=4,
        relax=True,
        max_iterations=0,
        env=solver_env,
    )
    assert not result["separation_complete"]
    assert result["limit_reached"]
    assert result["cut_iterations"] == 0
    assert result["master_solves"] == 1
