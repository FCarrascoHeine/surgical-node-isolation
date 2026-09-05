"""Deterministic deadline and reporting checks; no Gurobi license needed."""

import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
from gurobipy import GRB, GurobiError

import branch_and_cut
import formulations
import run
import time_budget
from instances import prepare_instance
from time_budget import BudgetExpired, TimeBudget
from utils import collect_model_result, empty_model_result, load_gurobi_env
from validation import validate_integer_result

SMALL_INSTANCE = Path(__file__).resolve().parents[1] / "instances/small_instance.json"


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time_budget.time, "perf_counter", lambda: value.now)
    return value


class FakeModel:
    def __init__(self, clock, duration=2.0, status=GRB.OPTIMAL, has_solution=True):
        self.clock = clock
        self.duration = duration
        self.final_status = status
        self.final_has_solution = has_solution
        self.Params = SimpleNamespace(TimeLimit=None)
        self.Status = GRB.LOADED
        self.SolCount = 0
        self.Runtime = 0.0
        self.NodeCount = self.IterCount = 0.0
        self.NumVars = self.NumConstrs = 1
        self.NumQConstrs = 0
        self.calls = 0
        self.disposed = False

    @property
    def ObjBound(self):
        raise AttributeError("No LP objective bound available")

    def optimize(self):
        self.calls += 1
        self.allowance = self.Params.TimeLimit
        self.clock.now += self.duration
        self.Status = self.final_status
        self.SolCount = int(self.final_has_solution)
        self.Runtime = self.duration
        if self.SolCount:
            self.ObjVal = 4.0

    def dispose(self):
        self.disposed = True


def test_budget_is_shared_and_refreshes_solver_allowance(clock):
    budget = TimeBudget(60)
    model = SimpleNamespace(Params=SimpleNamespace())
    clock.now = 20
    budget.apply_to(model)
    assert model.Params.TimeLimit == 40
    clock.now = 55
    budget.apply_to(model)
    assert model.Params.TimeLimit == 5
    clock.now = 65
    assert budget.elapsed() == 65
    assert budget.remaining() == 0
    with pytest.raises(BudgetExpired):
        budget.apply_to(model)


@pytest.mark.parametrize("limit", [None, float("inf")])
def test_unlimited_budget(clock, limit):
    budget = TimeBudget(limit)
    clock.now = 1e12
    assert budget.remaining() is None
    assert not budget.expired()
    budget.check()


@pytest.mark.parametrize("limit", [-1, float("-inf"), float("nan")])
def test_invalid_budget_rejected_before_method_work(monkeypatch, limit):
    monkeypatch.setattr(formulations, "build_model", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(ValueError, match="time_limit"):
        formulations.solve_instance(SMALL_INSTANCE, 1, time_limit=limit)
    with pytest.raises(ValueError, match="time_limit"):
        run.run_comparison(SMALL_INSTANCE, time_limit=limit)


@pytest.mark.parametrize("formulation", [1, 2, 3])
@pytest.mark.parametrize("relax", [False, True])
def test_compact_preparation_and_build_consume_solver_budget(monkeypatch, clock, formulation, relax):
    model = FakeModel(clock)

    def build(*args, **kwargs):
        clock.now += 20
        return model, {"x": {(0, 1): SimpleNamespace(X=1.0)}}

    monkeypatch.setattr(formulations, "build_model", build)
    result = formulations.solve_instance(SMALL_INSTANCE, formulation, relax=relax, time_limit=60)
    assert model.allowance == 40
    assert result["runtime"] == 22
    assert result["solver_runtime"] == 2
    assert result["status_name"] == "OPTIMAL"
    assert result["has_solution"]
    assert result["solution_type"] == ("relaxation" if relax else "integer")
    assert result["variables"]["x"] == {(0, 1): 1.0}
    assert model.disposed


@pytest.mark.parametrize("formulation", [1, 2, 3])
def test_zero_budget_does_not_build_model(monkeypatch, clock, formulation):
    monkeypatch.setattr(formulations, "build_model", lambda *a, **k: pytest.fail("built"))
    result = formulations.solve_instance(SMALL_INSTANCE, formulation, time_limit=0)
    assert result["status_name"] == "TIME_LIMIT"
    assert not result["has_solution"]
    assert result["variables"] == {}
    assert result["objective_value"] is result["dual_bound"] is result["gap"] is None
    assert result["master_solves"] == 0


def test_expired_build_skips_optimization_and_disposes_model(monkeypatch, clock):
    model = FakeModel(clock)

    def build(*args, **kwargs):
        clock.now += 65
        return model, {}

    monkeypatch.setattr(formulations, "build_model", build)
    result = formulations.solve_instance(SMALL_INSTANCE, 2, time_limit=60)
    assert model.calls == 0
    assert model.disposed
    assert result["runtime"] == 65
    assert result["solver_runtime"] == 0
    assert result["status_name"] == "TIME_LIMIT"
    assert result["num_variables"] == 1


@pytest.mark.parametrize("has_solution", [False, True])
def test_solver_timeout_preserves_available_incumbent(monkeypatch, clock, has_solution):
    model = FakeModel(clock, status=GRB.TIME_LIMIT, has_solution=has_solution)
    monkeypatch.setattr(formulations, "build_model", lambda *a, **k: (
        model, {"x": {(0, 1): SimpleNamespace(X=1.0)}},
    ))
    result = formulations.solve_instance(SMALL_INSTANCE, 2, time_limit=2)
    assert result["status_name"] == "TIME_LIMIT"
    assert result["has_solution"] == has_solution
    assert bool(result["variables"]) == has_solution
    assert result["objective_value"] == (4.0 if has_solution else None)
    assert result["dual_bound"] is None
    assert model.disposed


def test_completed_solver_overrun_keeps_natural_status(monkeypatch, clock):
    model = FakeModel(clock, duration=3)
    monkeypatch.setattr(formulations, "build_model", lambda *a, **k: (model, {}))
    result = formulations.solve_instance(SMALL_INSTANCE, 1, time_limit=2)
    assert result["status_name"] == "OPTIMAL"
    assert result["runtime"] == 3


def test_interrupted_lp_primal_objective_is_not_a_lower_bound(clock):
    model = FakeModel(clock, status=GRB.TIME_LIMIT)
    model.optimize()
    result = collect_model_result(model, 2, True)
    assert result["objective_value"] == 4
    assert result["dual_bound"] is None
    assert result["gap"] is None


def test_strict_runner_writes_all_zero_budget_rows_without_validation_failure(tmp_path):
    output = tmp_path / "timeouts.csv"
    experiment = run.run_experiments(
        [SMALL_INSTANCE], repetitions=2, csv_filename=output,
        time_limit=0, heuristics=("ah", "ash"),
    )
    assert len(experiment["rows"]) == 20
    for row in experiment["rows"]:
        assert row["status"] == "TIME_LIMIT"
        assert not row["has_solution"]
        assert row["solution_type"] == "none"
        assert row["objective_value"] is row["gap"] is None
        assert row["validation_passed"] is None
    with output.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 20
    assert all(row["objective_value"] == "" for row in saved)
    assert all(row["has_solution"] == "False" for row in saved)


def test_runner_reporting_does_not_use_up_next_method_budget(monkeypatch, clock):
    allowances = []

    def solve(instance, formulation, **kwargs):
        budget = TimeBudget(kwargs["time_limit"])
        allowances.append(budget.remaining())
        clock.now += 3
        result = empty_model_result(formulation, kwargs["relax"])
        result["runtime"] = budget.elapsed()
        return result

    def report(row):
        clock.now += 100

    monkeypatch.setattr(run, "solve_instance", solve)
    result = run.run_comparison(
        SMALL_INSTANCE, formulations=(1, 2), time_limit=60, row_callback=report,
    )
    assert allowances == [60] * 4
    assert all(row["runtime"] == 3 for row in result["rows"])


def _nonoptimal_feasible_result():
    data = prepare_instance(SMALL_INSTANCE)
    x = {edge: float(edge in {(1, 3), (2, 3)}) for edge in data["edges"]}
    z = {(j, e): float(e in ({(0, 1), (1, 3)} if j == 0 else {(0, 1)}))
         for j in data["journeyers"] for e in data["edges"]}
    alpha = {(j, e): x[e] * z[j, e] for j, e in z}
    alpha[0, (2, 3)] = 1.0
    result = empty_model_result(2, False)
    result.update(
        objective_value=10.0, has_solution=True, solution_type="integer",
        variables={"x": x, "z": z, "alpha": alpha,
                   "y": {(0, n): float(n == 3) for n in data["nodes"]}},
    )
    return result


def test_feasible_nonoptimal_incumbent_need_not_use_shortest_paths(monkeypatch):
    result = _nonoptimal_feasible_result()
    validation = validate_integer_result(SMALL_INSTANCE, result)
    assert validation["valid"], validation["errors"]
    assert validation["original_objective"] == 4
    monkeypatch.setattr(run, "solve_instance", lambda *a, **k: copy.deepcopy(result))
    row = run.run_comparison(SMALL_INSTANCE, formulations=(2,), mode="integer")["rows"][0]
    assert row["validation_passed"]
    assert row["objective_value"] == 10
    assert row["original_objective"] == 4
    result["status_name"] = "OPTIMAL"
    assert not validate_integer_result(SMALL_INSTANCE, result)["valid"]


@pytest.mark.parametrize("scope,objective", [("none", None), ("restricted_master", 3.0)])
def test_console_shows_solution_scope(capsys, scope, objective):
    result = empty_model_result(4, True)
    result.update(solution_type=scope, has_solution=objective is not None, objective_value=objective)
    row = run._row_from_result(result, {"name": "test"}, 1, 0, 1)
    run.print_results([row])
    output = capsys.readouterr().out
    assert "Solution" in output
    assert scope in output
    assert "TIME_LIMIT" in output
    assert ("3.000000" in output) == (objective is not None)


def test_console_columns_expand_consistently_and_compact_large_values(capsys):
    rows = [
        {
            "instance": "test",
            "repetition": 1,
            "method": "f3",
            "mode": "integer",
            "status": "TIME_LIMIT",
            "solution_type": "integer",
            "objective_value": 273915.334291,
            "dual_bound": 269221.244892,
            "runtime": 60.0704,
            "cuts": 0,
        },
        {
            "instance": "test",
            "repetition": 1,
            "method": "f4",
            "mode": "integer",
            "status": "VALIDATION_FAILED",
            "solution_type": "none",
            "objective_value": 123456789012345.0,
            "dual_bound": None,
            "runtime": 0.0,
            "cuts": None,
        },
    ]

    run.print_results(rows)

    lines = capsys.readouterr().out.splitlines()
    assert len({len(line) for line in lines}) == 1
    assert "273915.334291" in lines[2]
    assert "269221.244892" in lines[2]
    assert "1.234568e+14" in lines[3]
    objective_end = lines[0].index("Objective") + len("Objective")
    for line, value in ((lines[2], "273915.334291"), (lines[3], "1.234568e+14")):
        assert line.index(value) + len(value) == objective_end


@pytest.fixture(scope="module")
def solver_env():
    try:
        env = load_gurobi_env()
    except GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required (error {error.errno})")
    yield env
    env.dispose()


@pytest.mark.parametrize("formulation", [1, 2, 3, 4])
@pytest.mark.parametrize("relax", [False, True])
def test_actual_gurobi_model_can_timeout_before_first_optimize(
    monkeypatch, clock, solver_env, formulation, relax,
):
    # Exercise real Gurobi attributes on an unoptimized model, without relying
    # on machine speed to hit a particular point in construction.
    module = branch_and_cut if formulation == 4 else formulations
    builder_name = "build_formulation_4" if formulation == 4 else "build_model"
    original = getattr(module, builder_name)
    built = []

    def build(*args, **kwargs):
        model, variables = original(*args, **kwargs)
        built.append(model)
        clock.now += 2
        return model, variables

    monkeypatch.setattr(module, builder_name, build)
    result = formulations.solve_instance(
        SMALL_INSTANCE, formulation, relax=relax, time_limit=1, env=solver_env,
    )
    assert result["status_name"] == "TIME_LIMIT"
    assert result["runtime"] == 2
    assert result["num_variables"] > 0
    assert result["master_solves"] == 0
    assert not result["has_solution"]
    assert result["objective_value"] is result["dual_bound"] is None
    assert result["variables"] == {}
    with pytest.raises(GurobiError):
        built[0].getVars()  # Explicit disposal also occurs on this exit path.
