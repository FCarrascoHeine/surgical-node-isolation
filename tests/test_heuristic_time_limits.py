from types import SimpleNamespace

import pytest
from gurobipy import GRB

import heuristics as heuristics_module
import time_budget
from heuristics import (
    solve_heuristic,
    solve_single_intruder_heuristic,
    solve_standard_heuristic,
)
from instances import load_instance, prepare_instance
from time_budget import BudgetExpired
from validation import evaluate_allocation


@pytest.fixture
def instance():
    return load_instance("instances/small_instance.json")


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time_budget.time, "perf_counter", lambda: value.now)
    return value


@pytest.mark.parametrize("solve", [solve_standard_heuristic, solve_single_intruder_heuristic])
def test_zero_budget_stops_before_preparation(monkeypatch, clock, instance, solve):
    def unexpected(*args, **kwargs):
        pytest.fail("Preparation should not start with an exhausted budget")

    monkeypatch.setattr(heuristics_module, "_prepared_data", unexpected)
    result = solve(instance, time_limit=0)

    assert result["status_name"] == "TIME_LIMIT"
    assert not result["has_solution"]
    assert result["solution_type"] == "none"
    assert result["objective_value"] is None
    assert result["variables"] == {}
    assert result["minimum_cut_solves"] == 0
    assert result["auxiliary_solves"] == 0


@pytest.mark.parametrize("method", ["ah", "ash", "auto"])
def test_input_preparation_uses_dispatch_budget(monkeypatch, clock, instance, method):
    prepared = prepare_instance(instance)

    def slow_preparation(_instance):
        clock.now += 2.0
        return prepared

    monkeypatch.setattr(heuristics_module, "_prepared_data", slow_preparation)
    result = solve_heuristic(instance, method=method, time_limit=1)

    assert result["status_name"] == "TIME_LIMIT"
    assert result["runtime"] == 2.0
    assert not result["has_solution"]
    assert result["auxiliary_solves"] == result["minimum_cut_solves"] == 0


def test_ah_path_overrun_does_not_start_subproblem(monkeypatch, clock, instance):
    real_paths = heuristics_module._journeyer_paths

    def slow_paths(*args, **kwargs):
        paths = real_paths(*args, **kwargs)
        clock.now += 2.0
        return paths

    def unexpected(*args, **kwargs):
        pytest.fail("An auxiliary model must not start after path finding expires")

    monkeypatch.setattr(heuristics_module, "_journeyer_paths", slow_paths)
    monkeypatch.setattr(heuristics_module, "_solve_standard_subproblem", unexpected)
    result = solve_standard_heuristic(instance, time_limit=1)

    assert result["status_name"] == "TIME_LIMIT"
    assert result["runtime"] == 2.0
    assert not result["has_solution"]


def test_ah_timeout_returns_best_even_when_terminal_requested(monkeypatch, clock, instance):
    outcomes = iter([
        ({(1, 3), (2, 3)}, "OPTIMAL"),
        ({(0, 1), (0, 2)}, "TIME_LIMIT"),
    ])

    def fake_subproblem(*args, **kwargs):
        selected, status = next(outcomes)
        return {
            "status_name": status, "selected": selected, "objective_value": 0.0,
            "runtime": 0.0, "num_variables": 0, "num_constraints": 0,
            "nodes_explored": 0.0, "simplex_iterations": 0.0,
        }

    monkeypatch.setattr(heuristics_module, "_solve_standard_subproblem", fake_subproblem)
    result = solve_standard_heuristic(instance, time_limit=1, return_best=False)

    assert result["status_name"] == "TIME_LIMIT"
    assert result["objective_value"] == 4.0
    assert result["terminal_objective"] == 7.5
    assert result["returned_best_candidate"]
    assert result["has_solution"]
    assert result["solution_type"] == "integer"


@pytest.mark.parametrize("iterations,expected_status", [(1, "TIME_LIMIT"), (2, "CONVERGED")])
def test_ah_atomic_overrun_distinguishes_convergence_from_iteration_cap(
    monkeypatch, clock, instance, iterations, expected_status,
):
    calls = []

    def fake_subproblem(*args, **kwargs):
        calls.append(None)
        if len(calls) == iterations:
            clock.now = 2.0
        return {
            "status_name": "OPTIMAL", "selected": {(1, 3), (2, 3)},
            "objective_value": 0.0, "runtime": 0.0, "num_variables": 0,
            "num_constraints": 0, "nodes_explored": 0.0,
            "simplex_iterations": 0.0,
        }

    monkeypatch.setattr(heuristics_module, "_solve_standard_subproblem", fake_subproblem)
    result = solve_standard_heuristic(instance, time_limit=1, max_iterations=iterations)

    assert result["status_name"] == expected_status
    assert result["runtime"] == 2.0
    assert result["objective_value"] == 4.0
    assert result["has_solution"]


def _fake_model(monkeypatch, clock, *, variable_seconds=0.0, update_seconds=0.0):
    model = SimpleNamespace(
        Params=SimpleNamespace(), NumVars=0, NumConstrs=0, Status=GRB.LOADED,
        Runtime=0.0, NodeCount=0.0, IterCount=0.0, SolCount=0,
        allowances=[], disposed=False,
    )

    def add_var(**kwargs):
        clock.now += variable_seconds
        model.NumVars += 1
        return 0.0

    def add_constraint(*args, **kwargs):
        model.NumConstrs += 1

    def update():
        clock.now += update_seconds

    def optimize():
        model.allowances.append(model.Params.TimeLimit)
        model.Status = GRB.TIME_LIMIT

    model.addVar = add_var
    model.addConstr = add_constraint
    model.setObjective = lambda *args: None
    model.update = update
    model.optimize = optimize
    model.dispose = lambda: setattr(model, "disposed", True)
    monkeypatch.setattr(heuristics_module, "Model", lambda *args, **kwargs: model)
    return model


def test_ah_applies_remaining_allowance_after_model_update(monkeypatch, clock, instance):
    model = _fake_model(monkeypatch, clock, update_seconds=2.0)
    result = heuristics_module._solve_standard_subproblem(
        prepare_instance(instance), set(), time_limit=5, output_flag=0,
        solver_seed=0, threads=1, env=None,
    )

    assert model.allowances == [3.0]
    assert model.disposed
    assert result["status_name"] == "TIME_LIMIT"


def test_ah_stops_building_and_disposes_model_when_budget_expires(monkeypatch, clock, instance):
    model = _fake_model(monkeypatch, clock, variable_seconds=2.0)
    with pytest.raises(BudgetExpired):
        heuristics_module._solve_standard_subproblem(
            prepare_instance(instance), set(), time_limit=1, output_flag=0,
            solver_seed=0, threads=1, env=None,
        )

    assert model.NumVars == 1
    assert model.allowances == []
    assert model.disposed


def test_ah_does_not_optimize_when_model_update_exhausts_budget(monkeypatch, clock, instance):
    model = _fake_model(monkeypatch, clock, update_seconds=2.0)
    with pytest.raises(BudgetExpired):
        heuristics_module._solve_standard_subproblem(
            prepare_instance(instance), set(), time_limit=1, output_flag=0,
            solver_seed=0, threads=1, env=None,
        )

    assert model.allowances == []
    assert model.disposed


def test_auto_zero_budget_still_reports_selected_method(clock, instance):
    result = solve_heuristic(instance, time_limit=0)

    assert result["method"] == "ash"
    assert result["status_name"] == "TIME_LIMIT"
    assert not result["has_solution"]


@pytest.mark.parametrize("slow_stage", ["initial_cut", "paths", "binary_cut"])
def test_ash_timeout_retains_feasible_work(monkeypatch, clock, instance, slow_stage):
    real_cut = heuristics_module.directed_min_cut
    real_paths = heuristics_module._journeyer_paths
    cut_calls = []

    def cut(*args, **kwargs):
        result = real_cut(*args, **kwargs)
        cut_calls.append(result)
        if (slow_stage == "initial_cut" and len(cut_calls) == 1) or (
            slow_stage == "binary_cut" and len(cut_calls) == 3
        ):
            clock.now += 2.0
        return result

    def paths(*args, **kwargs):
        result = real_paths(*args, **kwargs)
        if slow_stage == "paths":
            clock.now += 2.0
        return result

    monkeypatch.setattr(heuristics_module, "directed_min_cut", cut)
    monkeypatch.setattr(heuristics_module, "_journeyer_paths", paths)
    result = solve_single_intruder_heuristic(instance, time_limit=1, return_best=False)

    assert result["status_name"] == "TIME_LIMIT"
    assert result["runtime"] == 2.0
    assert result["has_solution"]
    assert result["solution_type"] == "integer"
    assert result["returned_best_candidate"]
    assert evaluate_allocation(instance, result["variables"]["x"])["valid"]
    assert result["minimum_cut_solves"] == (3 if slow_stage == "binary_cut" else 1)


def test_ash_completed_natural_convergence_survives_atomic_overrun(monkeypatch, clock, instance):
    instance["budget"] = int(sum(edge["checkpoint_cost"] for edge in instance["edges"]))
    real_cut = heuristics_module.directed_min_cut
    cut_calls = []

    def cut(*args, **kwargs):
        result = real_cut(*args, **kwargs)
        cut_calls.append(result)
        if len(cut_calls) == 2:
            clock.now += 2.0
        return result

    monkeypatch.setattr(heuristics_module, "directed_min_cut", cut)
    result = solve_single_intruder_heuristic(instance, time_limit=1)

    assert result["status_name"] == "CONVERGED"
    assert result["convergence_reason"] == "impact_cut_feasible"
    assert result["runtime"] == 2.0
    assert result["has_solution"]
