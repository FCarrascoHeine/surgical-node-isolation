"""Small real-Gurobi checks, including equality with the existing direct API."""

import copy
import csv

import gurobipy as gp
import pytest

import experiment_supervisor as supervisor
from branch_and_cut import solve_journeyer_dual
from formulations import build_model, solve_instance
from heuristics import _solve_standard_subproblem
from instances import load_instance, prepare_instance
from run import run_comparison
from utils import MemoryLimitReached, load_gurobi_env


@pytest.fixture(scope="module")
def env():
    try:
        environment = load_gurobi_env()
    except gp.GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required: {error}")
    yield environment
    environment.dispose()


def test_all_methods_and_modes_match_direct_execution_with_reused_worker(env, tmp_path):
    instance = load_instance("instances/small_instance.json")
    scaled = copy.deepcopy(instance)
    scaled["name"] = "scaled_instance"
    scaled["known_optimum"] *= 2
    for edge in scaled["edges"]:
        edge["transit_time"] *= 2
        edge["inspection_time"] *= 2
    instances = [instance, scaled]
    direct = {
        item["name"]: run_comparison(item, heuristics=("ah", "ash"), env=env, time_limit=30)
        for item in instances
    }
    filename = tmp_path / "supervised.csv"
    supervised = supervisor.run_supervised_experiments(
        instances, repetitions=2, heuristics=("ah", "ash"), time_limit=30,
        memory_limit_gb=1, csv_filename=filename,
    )
    assert len(supervised["rows"]) == 40
    for row in supervised["rows"]:
        expected = next(r for r in direct[row["instance"]]["rows"]
                        if (r["method"], r["mode"]) == (row["method"], row["mode"]))
        for field in (
            "status", "has_solution", "solution_type", "num_variables", "num_constraints",
            "validation_passed", "separation_complete",
        ):
            assert row[field] == expected[field], (row["method"], row["mode"], field)
        for field in ("objective_value", "dual_bound", "gap", "original_objective", "reference_objective", "reference_gap"):
            if expected[field] is None:
                assert row[field] is None
            else:
                assert row[field] == pytest.approx(expected[field], abs=1e-6)
        assert row["error_type"] is None
        assert row["memory_limit_gb"] == 1
    with filename.open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 40


@pytest.mark.parametrize("formulation", [1, 2, 3, 4])
def test_models_inherit_environment_memory_limit(env, formulation):
    env.setParam("SoftMemLimit", 1)
    try:
        model, _ = build_model("instances/small_instance.json", formulation, env=env)
        try:
            assert model.Params.SoftMemLimit == 1
        finally:
            model.dispose()
    finally:
        env.setParam("SoftMemLimit", gp.GRB.INFINITY)


def test_real_soft_memory_stop_and_next_model_still_solve(env):
    env.setParam("SoftMemLimit", 1e-9)
    try:
        result = solve_instance("instances/small_instance.json", 2, env=env)
        assert result["status_name"] == "MEM_LIMIT"
        assert not result["has_solution"]
        subproblem = _solve_standard_subproblem(
            prepare_instance("instances/small_instance.json"), set(), env=env,
            time_limit=None, output_flag=0, solver_seed=0, threads=1,
        )
        assert subproblem["status_name"] == "MEM_LIMIT"
        assert subproblem["selected"] is None
        data = prepare_instance("instances/small_instance.json")
        with pytest.raises(MemoryLimitReached):
            solve_journeyer_dual(data, data["journeyers"][0],
                                {edge: 1 for edge in data["edges"]}, env=env)
    finally:
        env.setParam("SoftMemLimit", gp.GRB.INFINITY)
    result = solve_instance("instances/small_instance.json", 2, env=env)
    assert result["status_name"] == "OPTIMAL"
    assert result["objective_value"] == pytest.approx(4)


def test_supervised_memory_limit_then_graph_heuristic_continues(env, tmp_path):
    # A deliberately tiny allowance safely provokes Gurobi's real MEM_LIMIT;
    # it does not exhaust machine memory. ASH is outside this solver limit.
    rows = supervisor.run_supervised_experiments(
        ["instances/small_instance.json"], formulations=(2,), heuristics=("ah", "ash"),
        mode="integer", memory_limit_gb=1e-9, csv_filename=tmp_path / "limited.csv",
    )["rows"]
    assert [row["status"] for row in rows] == ["MEM_LIMIT", "MEM_LIMIT", "CONVERGED"]
    assert rows[-1]["validation_passed"]
    assert rows[-1]["objective_value"] == pytest.approx(4)
