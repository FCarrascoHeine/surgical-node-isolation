"""Subset separation, root-only callbacks, cut budgets, and LP bound comparisons."""

import csv
import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from gurobipy import GRB, GurobiError

import formulation_3_VI as vi
import formulations
import testing_valid_inequalities as comparison
import time_budget
from instances import generate_instance, prepare_instance
from run import _row_from_result, finalize_comparison, run_experiments
from time_budget import TimeBudget
from utils import empty_model_result, load_gurobi_env
from validation import validate_integer_result

SMALL_INSTANCE = Path(__file__).resolve().parents[1] / "instances/small_instance.json"


@pytest.fixture(scope="module")
def solver_env():
    try:
        env = load_gurobi_env()
    except GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required: {error}")
    yield env
    env.dispose()


def test_separator_matches_exhaustive_subset_search():
    rng = random.Random(17)
    journeyers = (2, 7, 9, 18)
    edges = ((0, 1), (1, 2), (2, 3))
    for _ in range(30):
        x = {edge: rng.random() for edge in edges}
        z = {(j, edge): rng.random() for j in journeyers for edge in edges}
        # A fractional point satisfying the aggregated lower bounds.
        beta = {edge: max(0, sum(z[j, edge] for j in journeyers)
                          - len(journeyers) * (1 - x[edge])) for edge in edges}
        cuts = vi.separate_valid_inequalities(edges, journeyers, x, z, beta)
        violations = {cut["edge"]: cut["violation"] for cut in cuts}
        for edge in edges:
            exhaustive = max(
                sum(z[j, edge] for j in subset) - len(subset) * (1 - x[edge]) - beta[edge]
                for size in range(len(journeyers) + 1)
                for subset in itertools.combinations(journeyers, size)
            )
            assert violations.get(edge, 0) == pytest.approx(max(0, exhaustive))
        assert [c["violation"] for c in cuts] == sorted(violations.values(), reverse=True)


def test_subset_uses_strict_positivity_not_violation_tolerance():
    edge = (0, 1)
    cuts = vi.separate_valid_inequalities(
        [edge], [0, 1, 2, 3], {edge: 0.5},
        {(0, edge): 1, (1, edge): 0.5 + 1e-9, (2, edge): 0.5, (3, edge): 0},
        {edge: 0}, tolerance=1e-6,
    )
    assert cuts[0]["journeyers"] == (0, 1)
    assert vi.separate_valid_inequalities(
        [edge], [0, 1], {edge: 0.5}, {(0, edge): 0.5000001, (1, edge): 0},
        {edge: 0}, tolerance=1e-6,
    ) == []


def test_every_subset_inequality_is_valid_for_integer_products():
    for x, *z in itertools.product((0, 1), repeat=5):
        beta = x * sum(z)
        for size in range(len(z) + 1):
            for subset in itertools.combinations(range(len(z)), size):
                assert beta >= sum(z[j] for j in subset) - size * (1 - x)


def _callback_fixture(monkeypatch, **limits):
    edges = ((0, 1), (1, 2))
    variables = {
        "x": {e: ("x", e) for e in edges},
        "z": {(j, e): ("z", j, e) for j in (0, 1) for e in edges},
        "beta": {e: ("beta", e) for e in edges},
    }
    values = {v: (0.5 if v[0] == "x" else 0.0) for group in variables.values() for v in group.values()}
    values["z", 0, edges[0]] = 0.8
    values["z", 0, edges[1]] = 1.0
    settings = dict(max_cuts=10, max_cuts_per_round=10, max_iterations=10, tolerance=1e-6)
    settings.update(limits)
    separator = vi.RootSeparator(variables, TimeBudget(), **settings)
    model = SimpleNamespace(node_count=0, status=GRB.OPTIMAL, cuts=[], terminated=False)
    model.cbGet = lambda code: model.node_count if code == GRB.Callback.MIPNODE_NODCNT else model.status
    model.cbGetNodeRel = lambda requested: [values[v] for v in requested]
    model.cbCut = model.cuts.append
    model.terminate = lambda: setattr(model, "terminated", True)
    monkeypatch.setattr(vi, "_constraint", lambda variables, cut: cut)
    return separator, model


@pytest.mark.parametrize("event,node,status", [
    (GRB.Callback.MIPSOL, 0, GRB.OPTIMAL),
    (GRB.Callback.MIPNODE, 1, GRB.OPTIMAL),
    (GRB.Callback.MIPNODE, 20, GRB.OPTIMAL),
    (GRB.Callback.MIPNODE, 0, GRB.INFEASIBLE),
])
def test_callback_skips_every_event_except_optimal_root_nodes(monkeypatch, event, node, status):
    separator, model = _callback_fixture(monkeypatch)
    model.node_count, model.status = node, status
    model.cbGetNodeRel = lambda requested: pytest.fail("read a non-root relaxation")
    separator.callback(model, event)
    assert not model.cuts
    assert not separator.keys


def test_callback_prioritizes_violation_and_obeys_total_cap(monkeypatch):
    separator, model = _callback_fixture(monkeypatch, max_cuts=1)
    for _ in range(3):
        separator.callback(model, GRB.Callback.MIPNODE)
    assert len(model.cuts) == 1
    assert model.cuts[0]["edge"] == (1, 2)
    assert separator.reason == "cut_limit"


def test_callback_round_limit_and_duplicate_suppression(monkeypatch):
    separator, model = _callback_fixture(monkeypatch, max_cuts_per_round=1, max_iterations=2)
    for _ in range(4):
        separator.callback(model, GRB.Callback.MIPNODE)
    assert [cut["edge"] for cut in model.cuts] == [(1, 2), (0, 1)]
    assert separator.cut_history == [1, 1]
    assert separator.reason == "iteration_limit"


def test_callback_errors_are_retained_for_propagation(monkeypatch):
    separator, model = _callback_fixture(monkeypatch)
    def fail(_requested):
        raise RuntimeError("failed callback")
    model.cbGetNodeRel = fail
    separator.callback(model, GRB.Callback.MIPNODE)
    assert isinstance(separator.error, RuntimeError)
    assert model.terminated


@pytest.mark.parametrize("options", [
    {"max_cuts": -1}, {"max_cuts": None}, {"max_cuts": 1.5},
    {"max_cuts_per_round": -1}, {"max_iterations": -1},
    {"tolerance": 0}, {"tolerance": float("nan")},
])
def test_invalid_limits_fail_before_build(monkeypatch, options):
    monkeypatch.setattr(vi, "build_formulation_3_VI", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(ValueError):
        formulations.solve_instance(SMALL_INSTANCE, "3_VI", **options)


@pytest.mark.parametrize("relax", [False, True])
def test_zero_time_budget_does_not_build(monkeypatch, relax):
    monkeypatch.setattr(vi, "build_formulation_3_VI", lambda *a, **k: pytest.fail("built"))
    result = formulations.solve_instance(SMALL_INSTANCE, "3_VI", relax=relax, time_limit=0)
    assert result["status_name"] == "TIME_LIMIT"
    assert not result["has_solution"]
    assert result["cuts"] == 0


@pytest.mark.parametrize("limits", [
    {"max_cuts": 0}, {"max_cuts": 1}, {"max_cuts": 2},
    {"max_cuts": 100}, {"max_cuts_per_round": 0}, {"max_iterations": 0},
    {"max_cuts_per_round": 1, "max_iterations": 1},
])
def test_lp_bounds_and_cut_limits(solver_env, limits):
    result = formulations.solve_instance(SMALL_INSTANCE, "formulation_3_VI",
                                         relax=True, env=solver_env, **limits)
    assert result["status_name"] == "OPTIMAL"
    assert 3.0 - 1e-6 <= result["objective_value"] <= 3.25 + 1e-6
    assert result["cuts"] <= result["max_cuts"]
    assert result["cut_iterations"] <= result["max_iterations"]
    assert sum(result["cut_history"]) == result["cuts"]
    assert all(n <= result["max_cuts_per_round"] for n in result["cut_history"])
    assert result["master_solves"] == result["cut_iterations"] + 1
    assert result["num_constraints"] == 35 + result["cuts"]
    history = result["objective_history"]
    assert all(a <= b + 1e-6 for a, b in zip(history, history[1:]))
    if any(value == 0 for value in limits.values()):
        assert result["objective_value"] == pytest.approx(3.0)
        assert result["cuts"] == 0
        assert not result["separation_complete"]
    if limits == {"max_cuts": 100}:
        assert result["objective_value"] == pytest.approx(3.25)
        assert result["separation_complete"]
        data, values = prepare_instance(SMALL_INSTANCE), result["variables"]
        assert not vi.separate_valid_inequalities(
            data["edges"], data["journeyers"], values["x"], values["z"], values["beta"],
        )


def test_integer_solution_remains_valid(solver_env):
    result = formulations.solve_instance(SMALL_INSTANCE, "3_VI", env=solver_env, max_cuts=2)
    assert result["status_name"] == "OPTIMAL"
    assert result["objective_value"] == pytest.approx(4.0)
    assert result["cuts"] <= 2
    assert validate_integer_result(SMALL_INSTANCE, result)["valid"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complete_separation_matches_disaggregation_on_generated_instances(solver_env, seed):
    instance = generate_instance(num_nodes=5, num_edges=10, num_intruders=2,
                                 num_journeyers=4, seed=seed)
    baseline = formulations.solve_instance(instance, 3, relax=True, env=solver_env)
    disaggregated = formulations.solve_instance(instance, 2, relax=True, env=solver_env)
    strengthened = formulations.solve_instance(
        instance, "3_VI", relax=True, env=solver_env, max_cuts=1000, max_cuts_per_round=3,
    )
    assert strengthened["status_name"] == "OPTIMAL"
    assert strengthened["separation_complete"]
    assert baseline["objective_value"] <= strengthened["objective_value"] + 1e-6
    assert strengthened["objective_value"] == pytest.approx(disaggregated["objective_value"])


def test_real_root_callback_adds_cuts(solver_env):
    model, variables = formulations.build_model(SMALL_INSTANCE, "3_VI", env=solver_env)
    try:
        model.Params.Presolve = 0
        model.Params.Heuristics = 0
        model.Params.Cuts = 0
        assert model.Params.PreCrush == 1
        separator = vi.RootSeparator(variables, TimeBudget(), 10, 2, 10, 1e-6)
        model.optimize(separator.callback)
        assert separator.error is None
        assert model.Status == GRB.OPTIMAL
        assert model.ObjVal == pytest.approx(4.0)
        assert 0 < len(separator.keys) <= 10
    finally:
        model.dispose()


def test_lp_comparison_csv_contains_three_relaxations(solver_env, tmp_path):
    filename = tmp_path / "valid_inequalities.csv"
    experiment = run_experiments(
        [SMALL_INSTANCE], formulations=comparison.LP_FORMULATIONS, mode="relaxation",
        env=solver_env, max_cuts=100, max_cuts_per_round=1, csv_filename=filename,
    )
    rows = experiment["rows"]
    assert [row["formulation"] for row in rows] == [3, 2, "3_VI"]
    assert [row["objective_value"] for row in rows] == pytest.approx([3.0, 3.25, 3.25])
    assert [row["lp_improvement"] for row in rows] == pytest.approx([0, 0.25, 0.25])
    assert [row["lp_improved"] for row in rows] == [False, True, True]
    assert rows[2]["lp_improvement_percent"] == pytest.approx(100 * 0.25 / 3)
    assert rows[2]["max_cuts"] == 100
    assert rows[2]["max_cuts_per_round"] == 1
    assert rows[2]["cuts"] > 0
    with filename.open(newline="") as file:
        saved = list(csv.DictReader(file))
    assert [row["formulation"] for row in saved] == ["3", "2", "3_VI"]


def test_comparison_cli_schedules_three_lps(monkeypatch, tmp_path):
    calls = []
    def fake_run(paths, **kwargs):
        calls.append((paths, kwargs))
        return {"rows": []}
    monkeypatch.setattr(comparison, "run_supervised_experiments", fake_run)
    comparison.main([str(SMALL_INSTANCE), "--max-cuts", "7", "--max-cuts-per-round", "2",
                     "--csv", str(tmp_path / "comparison.csv")])
    assert calls[0][1]["formulations"] == (3, 2, "3_VI")
    assert calls[0][1]["mode"] == "relaxation"
    assert calls[0][1]["max_cuts"] == 7
    assert calls[0][1]["max_cuts_per_round"] == 2


@pytest.mark.parametrize("baseline_status,variant_status", [
    ("TIME_LIMIT", "OPTIMAL"), ("OPTIMAL", "TIME_LIMIT"), ("OPTIMAL", "MEM_LIMIT"),
])
def test_lp_improvement_does_not_use_interrupted_primal_values(baseline_status, variant_status):
    rows, results = [], {}
    for formulation, status, objective in ((3, baseline_status, 3.0), ("3_VI", variant_status, 3.25)):
        result = empty_model_result(formulation, True)
        result.update(status_name=status, objective_value=objective, has_solution=True,
                      solution_type="relaxation")
        rows.append(_row_from_result(result, {"name": "test"}, 1, 0, 1))
        results[formulation, "relaxation"] = result
    finalize_comparison(rows, results, None, (3, "3_VI"), False, True)
    assert rows[1]["lp_improvement"] is None
    assert rows[1]["lp_improvement_percent"] is None
    assert rows[1]["lp_improved"] is None


def test_zero_baseline_and_revalidation_of_lp_improvement():
    rows, results = [], {}
    for formulation, objective in ((3, 0.0), ("3_VI", 1.0)):
        result = empty_model_result(formulation, True)
        result.update(status_name="OPTIMAL", objective_value=objective, has_solution=True,
                      solution_type="relaxation")
        rows.append(_row_from_result(result, {"name": "test"}, 1, 0, 1))
        results[formulation, "relaxation"] = result
    finalize_comparison(rows, results, None, (3, "3_VI"), False, True)
    assert rows[1]["lp_improvement"] == 1.0
    assert rows[1]["lp_improvement_percent"] is None
    assert rows[1]["lp_improved"]
    rows[0]["validation_passed"] = False
    finalize_comparison(rows, results, None, (3, "3_VI"), False, True)
    assert rows[1]["lp_improvement"] is None
    assert rows[1]["lp_improved"] is None


@pytest.mark.parametrize("stop", ["before_reoptimization", GRB.TIME_LIMIT, GRB.MEM_LIMIT])
def test_interrupted_lp_keeps_previous_bound_and_marks_old_snapshot(monkeypatch, stop):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time_budget, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    edge = (0, 1)
    variables = {
        "x": {edge: SimpleNamespace(X=0.5)},
        "z": {(0, edge): SimpleNamespace(X=1.0), (1, edge): SimpleNamespace(X=0.0)},
        "beta": {edge: SimpleNamespace(X=0.0)},
    }

    class Model:
        Params = SimpleNamespace(TimeLimit=None)
        NumConstrs, NumQConstrs, NumVars = 1, 0, 4
        Status, SolCount = GRB.LOADED, 0
        NodeCount, IterCount, Runtime = 0, 0, 0
        ObjVal, ObjBound = 3.0, 3.0
        calls, disposed = 0, False

        def optimize(self):
            self.calls += 1
            self.Status, self.SolCount = GRB.OPTIMAL, 1
            self.Runtime = 1.0
            clock.now += 1
            if self.calls > 1:
                self.Status, self.SolCount, self.ObjBound = stop, 0, float("inf")

        def addConstr(self, constraint, name):
            self.NumConstrs += 1

        def update(self):
            if stop == "before_reoptimization":
                clock.now = 10.0

        def dispose(self):
            self.disposed = True

    model = Model()
    monkeypatch.setattr(vi, "build_formulation_3_VI", lambda *a, **k: (model, variables))
    monkeypatch.setattr(vi, "_constraint", lambda *a: True)
    result = vi.solve_instance({}, relax=True, time_limit=10)
    assert result["status"] == (GRB.TIME_LIMIT if isinstance(stop, str) else stop)
    assert result["has_solution"]
    assert result["objective_value"] == result["dual_bound"] == 3.0
    assert result["solution_type"] == "restricted_master"
    assert result["gap"] is None
    assert result["cuts"] == 1
    assert not result["separation_complete"]
    assert result["master_solves"] == (1 if isinstance(stop, str) else 2)
    assert model.disposed
