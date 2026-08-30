import copy
import random
from pathlib import Path

import pytest
from gurobipy import GurobiError

from create_complex_instances import (
    assign_checkpoint_costs,
    build_minimum_checkpoint_model,
    complex_instance_path,
    create_complex_instance,
    minimum_feasible_budget,
)
from formulations import BUILDERS
from instances import load_instance, save_instance, validate_instance
from utils import load_gurobi_env
from validation import evaluate_allocation


SMALL_INSTANCE = (
    Path(__file__).resolve().parents[1] / "instances" / "small_instance.json"
)


@pytest.fixture(scope="module")
def solver_env():
    try:
        env = load_gurobi_env()
    except GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required: {error}")
    yield env
    env.dispose()


def _weighted_small_instance():
    instance = copy.deepcopy(load_instance(SMALL_INSTANCE))
    instance.pop("known_optimum")
    instance.pop("known_optimal_checkpoints")
    for position, edge in enumerate(instance["edges"], start=1):
        edge["checkpoint_cost"] = 10 * position
    instance["budget"] = 50
    return instance


def test_cost_generation_is_integer_bounded_and_reproducible():
    instance = load_instance(SMALL_INSTANCE)
    first = assign_checkpoint_costs(instance, random.Random(17))
    second = assign_checkpoint_costs(instance, random.Random(17))

    first_costs = [edge["checkpoint_cost"] for edge in first["edges"]]
    second_costs = [edge["checkpoint_cost"] for edge in second["edges"]]
    assert first_costs == second_costs
    assert all(type(cost) is int and 10 <= cost <= 50 for cost in first_costs)
    assert all(edge["checkpoint_cost"] == 1.0 for edge in instance["edges"])


def test_complex_path_replaces_only_terminal_dir_marker():
    source = Path("instances/50_272_10_3_1dir.json")
    assert complex_instance_path(source).name == "50_272_10_3_1c.json"


def test_instance_and_allocation_validation_use_weighted_budget():
    instance = _weighted_small_instance()
    assert instance["budget"] > len(instance["edges"])
    assert validate_instance(instance)

    x_values = {
        (edge["tail"], edge["head"]): 0.0 for edge in instance["edges"]
    }
    x_values[1, 3] = 1.0
    x_values[2, 3] = 1.0
    assert not evaluate_allocation(instance, x_values)["valid"]


def test_minimum_budget_master_uses_only_x_and_lazy_intruder_cuts(solver_env):
    instance = _weighted_small_instance()
    model, x, costs = build_minimum_checkpoint_model(
        instance,
        budget_model="intruder-cuts",
        env=solver_env,
    )

    assert model.NumVars == len(instance["edges"])
    assert model.NumConstrs == 0
    assert model.Params.LazyConstraints == 1
    assert model._budget_model == "intruder-cuts"
    assert len(x) == len(costs) == len(instance["edges"])
    model.dispose()


def test_auto_budget_model_uses_problem_size_threshold(solver_env):
    instance = _weighted_small_instance()
    compact, _, _ = build_minimum_checkpoint_model(instance, env=solver_env)
    cuts, _, _ = build_minimum_checkpoint_model(
        instance,
        cut_threshold=1,
        env=solver_env,
    )

    assert compact._budget_model == "disaggregated"
    assert compact.NumConstrs > 0
    assert cuts._budget_model == "intruder-cuts"
    assert cuts.NumConstrs == 0
    compact.dispose()
    cuts.dispose()


@pytest.mark.parametrize(
    "formulation,constraint_name",
    [(1, "budget_1"), (2, "budget_1"), (3, "budget_11"), (4, "budget_23")],
)
def test_formulation_budget_has_checkpoint_cost_coefficients(
    solver_env,
    formulation,
    constraint_name,
):
    instance = _weighted_small_instance()
    model, variables = BUILDERS[formulation](instance, env=solver_env)
    constraint = model.getConstrByName(constraint_name)
    costs = {
        (edge["tail"], edge["head"]): edge["checkpoint_cost"]
        for edge in instance["edges"]
    }
    for edge, variable in variables["x"].items():
        assert model.getCoeff(constraint, variable) == costs[edge]
    model.dispose()


def test_small_complex_instance_has_minimum_feasible_budget(tmp_path, solver_env):
    source = tmp_path / "toy_dir.json"
    save_instance(load_instance(SMALL_INSTANCE), source)
    instance, result = create_complex_instance(
        source,
        rng=random.Random(7),
        budget_model="intruder-cuts",
        env=solver_env,
    )

    assert result["optimal"]
    assert result["budget_model"] == "intruder-cuts"
    assert result["budget"] == 32
    assert result["intruder_cuts"] > 0
    assert result["lazy_additions"] >= result["intruder_cuts"]
    assert result["callback_calls"] > 0
    assert instance["name"] == "toy_c"
    assert instance["budget"] == 32
    assert instance["known_feasible_checkpoints"] == [[1, 3], [2, 3]]
    assert validate_instance(instance)

    disaggregated_result = minimum_feasible_budget(
        instance,
        budget_model="disaggregated",
        env=solver_env,
    )
    assert disaggregated_result["optimal"]
    assert disaggregated_result["budget"] == result["budget"]
