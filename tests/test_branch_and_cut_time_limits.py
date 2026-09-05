from types import SimpleNamespace

import pytest
from gurobipy import GRB

import branch_and_cut as branch
import time_budget
from time_budget import BudgetExpired, TimeBudget

EDGE = (0, 1)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    clock_module = SimpleNamespace(perf_counter=clock)
    monkeypatch.setattr(time_budget, "time", clock_module)
    monkeypatch.setattr(branch, "time", clock_module)
    return clock


class Master:
    def __init__(self, action=None):
        self.Params = SimpleNamespace(TimeLimit=None)
        self.NumConstrs = 1
        self.NumQConstrs = 0
        self.NumVars = 3
        self.Status = GRB.LOADED
        self.SolCount = 0
        self.ObjVal = 3.0
        self.ObjBound = 3.0
        self.Runtime = 0.0
        self.NodeCount = 0.0
        self.IterCount = 0.0
        self.variables = {
            "x": {EDGE: SimpleNamespace(X=1.0)},
            "alpha": {(0, EDGE): SimpleNamespace(X=1.0)},
            "phi": {0: SimpleNamespace(X=2.0)},
        }
        self._data = {"edges": [EDGE], "journeyers": [0]}
        self._solve_env = None
        self.action = action
        self.allowances = []
        self.disposed = False
        self.terminated = False

    def optimize(self, callback=None):
        self.allowances.append(self.Params.TimeLimit)
        self.Status = GRB.OPTIMAL
        self.SolCount = 1
        if self.action is not None:
            self.action(self, callback)
        elif callback is not None:
            callback(self, GRB.Callback.MIPSOL)

    def cbGetSolution(self, variable):
        return variable.X

    def cbGet(self, what):
        assert what == GRB.Callback.MIPSOL_OBJ
        return self.ObjVal

    def terminate(self):
        self.terminated = True

    def update(self):
        pass

    def dispose(self):
        self.disposed = True


def install_master(monkeypatch, clock, model, build_duration=0):
    def build(*args, **kwargs):
        clock.advance(build_duration)
        return model, model.variables

    monkeypatch.setattr(branch, "build_formulation_4", build)


@pytest.mark.parametrize("relax", [False, True])
def test_zero_budget_stops_before_master_construction(monkeypatch, clock, relax):
    def unexpected_build(*args, **kwargs):
        pytest.fail("A zero budget must not start constructing a model")

    monkeypatch.setattr(branch, "build_formulation_4", unexpected_build)
    result = branch.solve_instance({}, relax=relax, time_limit=0)
    assert result["status_name"] == "TIME_LIMIT"
    assert not result["has_solution"]
    assert result["solution_type"] == "none"
    assert result["master_solves"] == 0


@pytest.mark.parametrize("relax", [False, True])
def test_master_build_consumes_budget_and_is_disposed(monkeypatch, clock, relax):
    model = Master()
    install_master(monkeypatch, clock, model, build_duration=6)
    result = branch.solve_instance({}, relax=relax, time_limit=5)
    assert result["runtime"] == 6
    assert result["status_name"] == "TIME_LIMIT"
    assert result["num_variables"] == 3
    assert model.allowances == []
    assert model.disposed


@pytest.mark.parametrize("relax", [False, True])
def test_master_receives_remaining_budget_after_build(monkeypatch, clock, relax):
    model = Master()
    install_master(monkeypatch, clock, model, build_duration=3)
    separation_budgets = []

    def separate(*args, budget, **kwargs):
        separation_budgets.append(budget.remaining())
        return []

    monkeypatch.setattr(branch, "separate_solution", separate)
    result = branch.solve_instance({}, relax=relax, time_limit=10)
    assert model.allowances == [7]
    assert separation_budgets == [7]
    assert result["status_name"] == "OPTIMAL"
    assert result["separation_complete"]
    assert result["runtime"] == 3


@pytest.mark.parametrize("relax", [False, True])
def test_complete_final_separation_can_finish_naturally_after_deadline(monkeypatch, clock, relax):
    def optimize(model, callback):
        if callback is not None:
            callback(model, GRB.Callback.MIPSOL)
            callback(model, GRB.Callback.MESSAGE)

    model = Master(optimize)
    install_master(monkeypatch, clock, model)

    def separate(*args, **kwargs):
        clock.advance(11)
        return []

    monkeypatch.setattr(branch, "separate_solution", separate)
    result = branch.solve_instance({}, relax=relax, time_limit=10)
    assert result["status_name"] == "OPTIMAL"
    assert result["separation_complete"]
    assert not result["limit_reached"]
    assert result["runtime"] == 11


@pytest.mark.parametrize("previous_solution", [False, True])
def test_interrupted_lazy_separation_never_exposes_unchecked_incumbent(
    monkeypatch, clock, previous_solution
):
    def optimize(model, callback):
        if previous_solution:
            model.ObjVal = 10.0
            callback(model, GRB.Callback.MIPSOL)
        model.ObjVal = 1.0
        model.ObjBound = 1.0
        model.variables["x"][EDGE].X = 0.0
        callback(model, GRB.Callback.MIPSOL)
        # A terminated callback need not prevent Gurobi from retaining this
        # unsafe incumbent or even marking the restricted model optimal.
        model.Status = GRB.OPTIMAL

    model = Master(optimize)
    install_master(monkeypatch, clock, model)

    def separate(data, x, alpha, phi, budget, **kwargs):
        if x[EDGE] == 0:
            clock.advance(10)
            budget.check()
        return []

    monkeypatch.setattr(branch, "separate_solution", separate)
    result = branch.solve_instance({}, time_limit=10)
    assert result["status_name"] == "TIME_LIMIT"
    assert not result["separation_complete"]
    assert model.terminated
    assert model.disposed
    assert result["dual_bound"] == 1
    if previous_solution:
        assert result["objective_value"] == 10
        assert result["variables"]["x"][EDGE] == 1
        assert result["gap"] == pytest.approx(0.9)
        assert result["has_solution"]
    else:
        assert result["objective_value"] is None
        assert result["variables"] == {}
        assert result["gap"] is None
        assert not result["has_solution"]


def test_integer_retains_best_fully_separated_candidate(monkeypatch, clock):
    def optimize(model, callback):
        for objective in [10.0, 12.0, 8.0]:
            model.ObjVal = objective
            callback(model, GRB.Callback.MIPSOL)
        clock.advance(10)
        callback(model, GRB.Callback.MIP)

    model = Master(optimize)
    install_master(monkeypatch, clock, model)
    monkeypatch.setattr(branch, "separate_solution", lambda *args, **kwargs: [])
    result = branch.solve_instance({}, time_limit=10)
    assert result["status_name"] == "TIME_LIMIT"
    assert result["objective_value"] == 8


def test_relaxation_separation_timeout_retains_master_as_lower_bound(monkeypatch, clock):
    model = Master()
    install_master(monkeypatch, clock, model)

    def separate(*args, budget, **kwargs):
        clock.advance(10)
        budget.check()

    monkeypatch.setattr(branch, "separate_solution", separate)
    result = branch.solve_instance({}, relax=True, time_limit=10)
    assert result["status_name"] == "TIME_LIMIT"
    assert result["solution_type"] == "restricted_master"
    assert result["objective_value"] == result["dual_bound"] == 3
    assert result["gap"] is None
    assert result["variables"]["x"][EDGE] == 1
    assert not result["separation_complete"]
    assert result["separation_time"] == result["runtime"] == 10


def test_relaxation_retains_snapshot_when_cut_update_exhausts_budget(monkeypatch, clock):
    model = Master()
    install_master(monkeypatch, clock, model)
    monkeypatch.setattr(branch, "separate_solution", lambda *args, **kwargs: [
        {"family": "intruder", "agent": 0, "path": (EDGE,)}
    ])
    monkeypatch.setattr(branch, "_model_constraint", lambda *args: None)

    def update():
        model.SolCount = 0
        model.Status = GRB.LOADED
        model.variables["x"][EDGE].X = -99
        clock.advance(10)

    model.update = update
    result = branch.solve_instance({}, relax=True, time_limit=10)
    assert model.allowances == [10]
    assert result["master_solves"] == 1
    assert result["objective_value"] == result["dual_bound"] == 3
    assert result["variables"]["x"][EDGE] == 1
    assert result["solution_type"] == "restricted_master"
    assert result["status_name"] == "TIME_LIMIT"
    assert result["gap"] is None


def test_relaxation_retains_previous_bound_when_next_master_has_no_solution(monkeypatch, clock):
    def optimize(model, callback):
        if len(model.allowances) == 2:
            model.Status = GRB.TIME_LIMIT
            model.SolCount = 0
            model.ObjBound = float("-inf")

    model = Master(optimize)
    install_master(monkeypatch, clock, model)
    monkeypatch.setattr(branch, "separate_solution", lambda *args, **kwargs: [
        {"family": "intruder", "agent": 0, "path": (EDGE,)}
    ])
    monkeypatch.setattr(branch, "_model_constraint", lambda *args: None)
    result = branch.solve_instance({}, relax=True, time_limit=10)
    assert result["master_solves"] == 2
    assert result["objective_value"] == result["dual_bound"] == 3
    assert result["solution_type"] == "restricted_master"
    assert result["status_name"] == "TIME_LIMIT"
    assert result["gap"] is None


def test_interrupted_lp_primal_does_not_replace_previous_lower_bound(monkeypatch, clock):
    class LP(Master):
        @property
        def ObjBound(self):
            raise AttributeError("LPs do not expose ObjBound")

        @ObjBound.setter
        def ObjBound(self, value):
            pass

    def optimize(model, callback):
        if len(model.allowances) == 2:
            model.Status = GRB.TIME_LIMIT
            model.ObjVal = 7.0
            model.variables["x"][EDGE].X = 0.5

    model = LP(optimize)
    install_master(monkeypatch, clock, model)
    monkeypatch.setattr(branch, "separate_solution", lambda *args, **kwargs: [
        {"family": "intruder", "agent": 0, "path": (EDGE,)}
    ])
    monkeypatch.setattr(branch, "_model_constraint", lambda *args: None)
    result = branch.solve_instance({}, relax=True, time_limit=10)
    assert result["master_solves"] == 2
    assert result["objective_value"] == 7
    assert result["dual_bound"] == 3
    assert result["variables"]["x"][EDGE] == 0.5
    assert result["solution_type"] == "restricted_master"
    assert result["gap"] is None


def test_iteration_limited_master_does_not_claim_full_optimality(monkeypatch, clock):
    model = Master()
    install_master(monkeypatch, clock, model)
    monkeypatch.setattr(branch, "separate_solution", lambda *args, **kwargs: [
        {"family": "intruder", "agent": 0, "path": (EDGE,)}
    ])
    result = branch.solve_instance({}, relax=True, max_iterations=0)
    assert result["status_name"] == "ITERATION_LIMIT"
    assert result["solution_type"] == "restricted_master"
    assert result["dual_bound"] == 3
    assert result["gap"] is None


class Expression:
    X = 0.0

    def __add__(self, other):
        return self

    __radd__ = __add__
    __sub__ = __add__
    __mul__ = __add__
    __le__ = __add__


class Dual:
    def __init__(self, status):
        self.Params = SimpleNamespace()
        self.Status = status
        self.allowances = []
        self.disposed = False

    def addVar(self, **kwargs):
        return Expression()

    def update(self):
        pass

    def setObjective(self, *args):
        pass

    def addConstr(self, *args, **kwargs):
        pass

    def optimize(self):
        self.allowances.append(self.Params.TimeLimit)

    def dispose(self):
        self.disposed = True


def dual_data():
    return {
        "nodes": [0, 1], "edges": [EDGE], "tau": {EDGE: 1.0},
        "balance": {}, "journeyer_source": {0: 0}, "journeyer_target": {0: 1},
    }


@pytest.mark.parametrize("build_duration", [3, 10])
def test_dual_uses_shared_remaining_budget_and_disposes_on_timeout(monkeypatch, clock, build_duration):
    dual = Dual(GRB.TIME_LIMIT)

    def build(*args, **kwargs):
        clock.advance(build_duration)
        return dual

    monkeypatch.setattr(branch, "Model", build)
    monkeypatch.setattr(branch, "quicksum", sum)
    with pytest.raises(BudgetExpired):
        branch.solve_journeyer_dual(dual_data(), 0, {EDGE: 1}, budget=TimeBudget(10))
    assert dual.allowances == ([7] if build_duration == 3 else [])
    assert dual.disposed


def test_separation_stops_between_graph_operations(monkeypatch, clock):
    data = dual_data()
    data.update({
        "intruders": [0, 1], "journeyers": [0],
        "intruder_source": {0: 0, 1: 0}, "intruder_target": {0: 1, 1: 1},
    })
    calls = []

    def shortest_path(*args):
        calls.append("shortest_path")
        clock.advance(10)
        return 1, [EDGE]

    monkeypatch.setattr(branch, "shortest_path", shortest_path)
    with pytest.raises(BudgetExpired):
        branch.separate_solution(data, {EDGE: 1}, {(0, EDGE): 1}, {0: 1}, budget=TimeBudget(10))
    assert calls == ["shortest_path"]
