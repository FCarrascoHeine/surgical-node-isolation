"""Memory stops must retain only mathematically justified results."""

from types import SimpleNamespace

import pytest
from gurobipy import GRB, GurobiError

import branch_and_cut as branch
import formulations
import heuristics
import run
from instances import load_instance
from time_budget import TimeBudget
from utils import MemoryLimitReached
from validation import evaluate_allocation
from test_branch_and_cut_time_limits import Dual, EDGE, Master, dual_data
from test_time_limits import FakeModel


@pytest.mark.parametrize("formulation", [1, 2, 3])
@pytest.mark.parametrize("has_solution", [False, True])
def test_compact_soft_memory_stop_preserves_available_information(monkeypatch, formulation, has_solution):
    model = FakeModel(SimpleNamespace(now=0), status=GRB.MEM_LIMIT, has_solution=has_solution)
    monkeypatch.setattr(formulations, "build_model", lambda *args, **kwargs: (
        model, {"x": {EDGE: SimpleNamespace(X=1.0)}},
    ))
    result = formulations.solve_instance({}, formulation)
    assert result["status_name"] == "MEM_LIMIT"
    assert result["has_solution"] == has_solution
    assert bool(result["variables"]) == has_solution
    assert result["dual_bound"] is None
    assert model.disposed


@pytest.mark.parametrize("formulation", [1, 2, 3, 4])
@pytest.mark.parametrize("stage", ["variables", "constraints"])
def test_partial_build_is_disposed_even_before_builder_returns(monkeypatch, formulation, stage):
    class Model:
        Params = SimpleNamespace()
        disposed = False

        def addVar(self, **kwargs):
            if stage == "variables":
                raise MemoryError("during variable construction")
            return 0.0

        def update(self):
            pass

        def setObjective(self, *args):
            raise GurobiError(GRB.Error.OUT_OF_MEMORY, "during objective construction")

        def dispose(self):
            self.disposed = True

    model = Model()
    monkeypatch.setattr(formulations, "Model", lambda *args, **kwargs: model)
    with pytest.raises((MemoryError, GurobiError)):
        formulations.build_model("instances/small_instance.json", formulation)
    assert model.disposed


def test_memory_limited_dual_cannot_be_used_for_separation(monkeypatch):
    dual = Dual(GRB.MEM_LIMIT)
    monkeypatch.setattr(branch, "Model", lambda *args, **kwargs: dual)
    monkeypatch.setattr(branch, "quicksum", sum)
    with pytest.raises(MemoryLimitReached):
        branch.solve_journeyer_dual(dual_data(), 0, {EDGE: 1}, budget=TimeBudget(30))
    assert dual.disposed


@pytest.mark.parametrize("previous_solution", [False, True])
def test_lazy_memory_interruption_never_accepts_unchecked_incumbent(monkeypatch, previous_solution):
    def optimize(model, callback):
        if previous_solution:
            model.ObjVal = 10
            callback(model, GRB.Callback.MIPSOL)
        model.ObjVal = model.ObjBound = 1
        model.variables["x"][EDGE].X = 0
        callback(model, GRB.Callback.MIPSOL)
        model.Status = GRB.OPTIMAL  # Even a misleading master status must be overridden.

    model = Master(optimize)
    monkeypatch.setattr(branch, "build_formulation_4", lambda *args, **kwargs: (model, model.variables))

    def separate(data, x, *args, **kwargs):
        if x[EDGE] == 0:
            raise MemoryLimitReached()
        return []

    monkeypatch.setattr(branch, "separate_solution", separate)
    result = branch.solve_instance({})
    assert result["status_name"] == "MEM_LIMIT"
    assert result["objective_value"] == (10 if previous_solution else None)
    assert result["has_solution"] == previous_solution
    assert result["dual_bound"] == 1
    assert not result["separation_complete"]
    assert result["limit_reached"]
    assert model.disposed and model.terminated


@pytest.mark.parametrize("error", [MemoryError("python"), GurobiError(10001, "gurobi")])
def test_raw_callback_oom_keeps_original_type_for_supervisor(monkeypatch, error):
    model = Master()
    monkeypatch.setattr(branch, "build_formulation_4", lambda *args, **kwargs: (model, model.variables))

    def separate(*args, **kwargs):
        raise error

    monkeypatch.setattr(branch, "separate_solution", separate)
    with pytest.raises(type(error)):
        branch.solve_instance({})
    assert model.disposed


@pytest.mark.parametrize("stage", ["master", "separation"])
def test_relaxation_memory_stop_keeps_bound_but_not_full_feasibility_claim(monkeypatch, stage):
    def optimize(model, callback):
        if stage == "master":
            model.Status = GRB.MEM_LIMIT

    model = Master(optimize)
    monkeypatch.setattr(branch, "build_formulation_4", lambda *args, **kwargs: (model, model.variables))

    def separate(*args, **kwargs):
        raise MemoryLimitReached()

    monkeypatch.setattr(branch, "separate_solution", separate)
    result = branch.solve_instance({}, relax=True)
    assert result["status_name"] == "MEM_LIMIT"
    assert result["solution_type"] == "restricted_master"
    assert result["dual_bound"] == 3
    assert result["gap"] is None
    assert not result["separation_complete"]
    assert model.disposed


@pytest.mark.parametrize("last_candidate", [None, {(0, 1), (0, 2)}])
def test_ah_memory_limit_returns_best_even_when_terminal_requested(monkeypatch, last_candidate):
    outcomes = iter([({(1, 3), (2, 3)}, "OPTIMAL"), (last_candidate, "MEM_LIMIT")])

    def subproblem(*args, **kwargs):
        selected, status = next(outcomes)
        return {
            "status_name": status, "selected": selected, "objective_value": 0.0,
            "runtime": 0.0, "num_variables": 0, "num_constraints": 0,
            "nodes_explored": 0.0, "simplex_iterations": 0.0,
        }

    monkeypatch.setattr(heuristics, "_solve_standard_subproblem", subproblem)
    instance = load_instance("instances/small_instance.json")
    result = heuristics.solve_standard_heuristic(instance, return_best=False)
    assert result["status_name"] == "MEM_LIMIT"
    assert result["objective_value"] == 4
    assert result["returned_best_candidate"]
    assert not result["subproblems_optimal"]
    assert evaluate_allocation(instance, result["variables"]["x"])["valid"]


def test_ah_memory_limit_without_candidate_is_not_infeasibility(monkeypatch):
    monkeypatch.setattr(heuristics, "_solve_standard_subproblem", lambda *args, **kwargs: {
        "status_name": "MEM_LIMIT", "selected": None, "objective_value": None,
        "runtime": 0.0, "num_variables": 0, "num_constraints": 0,
        "nodes_explored": 0.0, "simplex_iterations": 0.0,
    })
    result = heuristics.solve_standard_heuristic("instances/small_instance.json")
    assert result["status_name"] == "MEM_LIMIT"
    assert not result["has_solution"]
    assert not result["subproblems_optimal"]


def test_final_validation_memory_limit_retains_previously_separated_snapshot(monkeypatch):
    # A fully separated F4 candidate can survive a memory stop in the runner's
    # additional separation check, but that check must not be reported as passed.
    result = {
        "status_name": "OPTIMAL", "formulation": 4, "relax": False,
        "runtime": 0, "objective_value": 4, "has_solution": True,
        "dual_bound": 4, "gap": 0, "variables": {"x": {}, "alpha": {}, "phi": {}},
    }
    monkeypatch.setattr(run, "solve_instance", lambda *args, **kwargs: result)
    monkeypatch.setattr(run, "validate_integer_result", lambda *args, **kwargs: {
        "valid": True, "original_objective": 4, "errors": [],
    })

    def separate(*args, **kwargs):
        raise MemoryLimitReached()

    monkeypatch.setattr(run, "separate_solution", separate)
    row = run.run_comparison("instances/small_instance.json", formulations=(4,), mode="integer")["rows"][0]
    assert row["status"] == "MEM_LIMIT"
    assert row["has_solution"]
    assert row["validation_passed"] is None
    assert not row["separation_complete"]
