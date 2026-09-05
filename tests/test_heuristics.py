import copy
import itertools
from pathlib import Path

import pytest
from gurobipy import GurobiError

import heuristics as heuristics_module
from graph_algorithms import directed_min_cut
from heuristics import (
    solve_heuristic,
    solve_single_intruder_heuristic,
    solve_standard_heuristic,
)
from instances import load_instance, prepare_instance
from run import run_comparison
from utils import load_gurobi_env
from validation import evaluate_allocation

SMALL_INSTANCE = (
    Path(__file__).resolve().parents[1] / "instances" / "small_instance.json"
)
REPRESENTATIVE_SINGLE_INSTANCE = (
    SMALL_INSTANCE.parent / "single50_272_10_3_1dir.json"
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


def _assert_valid_result(instance, result):
    assert result["status_name"] == "CONVERGED"
    assert result["variables"]["x"]
    evaluation = evaluate_allocation(instance, result["variables"]["x"])
    assert evaluation["valid"], evaluation["errors"]
    assert result["objective_value"] == pytest.approx(
        evaluation["objective_value"], abs=1e-9
    )
    assert result["checkpoint_cost"] <= instance["budget"] + 1e-9


def _minimum_surrogate_value(instance, support):
    data = prepare_instance(instance)
    values = []
    for size in range(len(data["edges"]) + 1):
        for chosen in itertools.combinations(data["edges"], size):
            x_values = {
                edge: float(edge in chosen) for edge in data["edges"]
            }
            if not evaluate_allocation(instance, x_values)["valid"]:
                continue
            values.append(
                sum(
                    data["inspection_time"][edge]
                    for edge in chosen
                    if edge in support
                )
            )
    return min(values)


def test_single_intruder_heuristic_reproduces_hand_checked_optimum():
    instance = load_instance(SMALL_INSTANCE)
    result = solve_single_intruder_heuristic(instance)

    _assert_valid_result(instance, result)
    assert result["selected_edges"] == ((1, 3), (2, 3))
    assert result["objective_value"] == pytest.approx(4.0, abs=1e-9)
    assert result["convergence_reason"] == "stable_allocation"
    assert result["heuristic_iterations"] == 2
    assert result["minimum_cut_solves"] == 30


def test_single_intruder_heuristic_matches_representative_exact_objective():
    instance = load_instance(REPRESENTATIVE_SINGLE_INSTANCE)
    result = solve_single_intruder_heuristic(instance)

    _assert_valid_result(instance, result)
    assert result["objective_value"] == pytest.approx(
        7779.018444281898, abs=1e-6
    )
    assert result["checkpoint_cost"] == 15.0
    assert len(result["selected_edges"]) == 15
    assert result["heuristic_iterations"] == 2


def test_single_intruder_scalarized_cuts_match_their_mathematical_definition():
    instance = _weighted_small_instance()
    data = prepare_instance(instance)
    result = solve_single_intruder_heuristic(instance)

    _assert_valid_result(instance, result)
    assert result["selected_edges"] == ((0, 1), (0, 2))
    assert result["checkpoint_cost"] == 40.0
    assert result["objective_value"] == pytest.approx(7.5, abs=1e-9)

    intruder = data["intruders"][0]
    for record in result["iteration_history"]:
        support = set(record["path_support_edges"])
        alpha = record["alpha"]
        capacities = {
            edge: alpha * data["checkpoint_cost"][edge]
            + (1 - alpha)
            * (data["inspection_time"][edge] if edge in support else 0.0)
            for edge in data["edges"]
        }
        minimum_value, _, _ = directed_min_cut(
            data["nodes"],
            data["edges"],
            capacities,
            data["intruder_source"][intruder],
            data["intruder_target"][intruder],
        )
        selected_value = sum(capacities[edge] for edge in record["selected_edges"])
        assert selected_value == pytest.approx(minimum_value, abs=1e-9)
        assert record["alpha_upper"] - record["alpha_lower"] < 1e-4

        lower = record["alpha_lower"]
        lower_capacities = {
            edge: lower * data["checkpoint_cost"][edge]
            + (1 - lower)
            * (data["inspection_time"][edge] if edge in support else 0.0)
            for edge in data["edges"]
        }
        _, _, lower_cut = directed_min_cut(
            data["nodes"],
            data["edges"],
            lower_capacities,
            data["intruder_source"][intruder],
            data["intruder_target"][intruder],
        )
        lower_cost = sum(data["checkpoint_cost"][edge] for edge in lower_cut)
        assert lower == 0.0 or lower_cost > data["budget"]


def test_single_intruder_heuristic_reports_infeasible_weighted_budget():
    instance = copy.deepcopy(load_instance(SMALL_INSTANCE))
    instance.pop("known_optimum")
    instance.pop("known_optimal_checkpoints")
    instance["budget"] = 0

    result = solve_single_intruder_heuristic(instance)

    assert result["status_name"] == "INFEASIBLE"
    assert result["convergence_reason"] == "minimum_cost_cut_exceeds_budget"
    assert result["objective_value"] is None
    assert result["minimum_cut_solves"] == 1


def test_standard_heuristic_reproduces_hand_checked_optimum(solver_env):
    instance = load_instance(SMALL_INSTANCE)
    result = solve_standard_heuristic(instance, env=solver_env)

    _assert_valid_result(instance, result)
    assert result["selected_edges"] == ((1, 3), (2, 3))
    assert result["objective_value"] == pytest.approx(4.0, abs=1e-9)
    assert result["convergence_reason"] == "stable_allocation"
    assert result["subproblems_optimal"]


def test_standard_heuristic_handles_multiple_intruders(solver_env):
    instance = copy.deepcopy(load_instance(SMALL_INSTANCE))
    instance.pop("known_optimum")
    instance.pop("known_optimal_checkpoints")
    instance["intruders"].append({"id": 1, "source": 1, "target": 3})

    result = solve_standard_heuristic(instance, env=solver_env)

    _assert_valid_result(instance, result)
    assert result["selected_edges"] == ((1, 3), (2, 3))
    assert result["auxiliary_solves"] == result["heuristic_iterations"]


def test_standard_subproblems_use_weighted_budget_and_path_support(solver_env):
    instance = _weighted_small_instance()
    data = prepare_instance(instance)
    result = solve_standard_heuristic(instance, env=solver_env)

    _assert_valid_result(instance, result)
    assert result["selected_edges"] == ((0, 1), (0, 2))
    assert result["checkpoint_cost"] == 40.0

    support_sizes = [
        record["path_support_size"] for record in result["iteration_history"]
    ]
    assert support_sizes == sorted(support_sizes)
    for record in result["iteration_history"]:
        support = set(record["path_support_edges"])
        expected_surrogate = sum(
            data["inspection_time"][edge]
            for edge in record["selected_edges"]
            if edge in support
        )
        assert record["auxiliary_objective"] == pytest.approx(
            expected_surrogate, abs=1e-9
        )
        assert record["auxiliary_objective"] == pytest.approx(
            _minimum_surrogate_value(instance, support), abs=1e-9
        )


def test_best_candidate_tracking_does_not_change_the_search(monkeypatch):
    instance = load_instance(SMALL_INSTANCE)
    best_cut = {(1, 3), (2, 3)}
    terminal_cut = {(0, 1), (0, 2)}

    def run(return_best):
        candidates = iter((best_cut, terminal_cut, terminal_cut))

        def fake_subproblem(*_args, **_kwargs):
            return {
                "status": 2,
                "status_name": "OPTIMAL",
                "selected": set(next(candidates)),
                "objective_value": 0.0,
                "runtime": 0.0,
                "num_variables": 0,
                "num_constraints": 0,
                "nodes_explored": 0.0,
                "simplex_iterations": 0.0,
            }

        monkeypatch.setattr(
            heuristics_module, "_solve_standard_subproblem", fake_subproblem
        )
        return solve_standard_heuristic(instance, return_best=return_best)

    best_result = run(True)
    terminal_result = run(False)

    assert best_result["best_iteration"] == 1
    assert best_result["selected_edges"] == ((1, 3), (2, 3))
    assert best_result["objective_value"] == 4.0
    assert best_result["terminal_objective"] == 7.5
    assert terminal_result["selected_edges"] == ((0, 1), (0, 2))
    assert terminal_result["objective_value"] == 7.5
    assert [
        record["selected_edges"] for record in best_result["iteration_history"]
    ] == [
        record["selected_edges"] for record in terminal_result["iteration_history"]
    ]


def test_auto_dispatch_selects_paper_single_intruder_method():
    result = solve_heuristic(SMALL_INSTANCE, method="auto")
    assert result["method"] == "ash"


def test_unified_runner_compares_formulation_and_both_heuristics(solver_env):
    comparison = run_comparison(
        SMALL_INSTANCE,
        mode="integer",
        formulations=(2,),
        heuristics=("ah", "ash"),
        env=solver_env,
    )

    assert [row["method"] for row in comparison["rows"]] == ["f2", "ah", "ash"]
    assert {row["mode"] for row in comparison["rows"]} == {
        "integer",
        "heuristic",
    }
    assert all(row["validation_passed"] for row in comparison["rows"])
    assert all(row["objective_value"] == pytest.approx(4.0) for row in comparison["rows"])
    assert all(row["reference_objective"] == pytest.approx(4.0) for row in comparison["rows"])
    assert all(row["reference_gap"] == pytest.approx(0.0) for row in comparison["rows"])
    assert ("ah", "heuristic") in comparison["results"]
    assert ("ash", "heuristic") in comparison["results"]


def test_unified_runner_supports_heuristic_only_and_not_applicable_rows():
    single = run_comparison(
        SMALL_INSTANCE,
        mode="integer",
        formulations=(),
        heuristics=("ash",),
    )
    assert len(single["rows"]) == 1
    assert single["rows"][0]["status"] == "CONVERGED"
    assert single["rows"][0]["reference_gap"] == pytest.approx(0.0)

    multiple = copy.deepcopy(load_instance(SMALL_INSTANCE))
    multiple.pop("known_optimum")
    multiple.pop("known_optimal_checkpoints")
    multiple["intruders"].append({"id": 1, "source": 1, "target": 3})
    skipped = run_comparison(
        multiple,
        mode="integer",
        formulations=(),
        heuristics=("ash",),
    )
    assert skipped["rows"][0]["status"] == "NOT_APPLICABLE"
    assert skipped["rows"][0]["convergence_reason"] == (
        "requires_exactly_one_intruder"
    )


def test_heuristic_option_validation():
    with pytest.raises(ValueError, match="exactly one intruder"):
        instance = copy.deepcopy(load_instance(SMALL_INSTANCE))
        instance["intruders"].append(
            {"id": 1, "source": 1, "target": 3}
        )
        solve_single_intruder_heuristic(instance)
    with pytest.raises(ValueError, match="between zero and one"):
        solve_single_intruder_heuristic(
            SMALL_INSTANCE, binary_search_tolerance=1.0
        )
    with pytest.raises(ValueError, match="at least one"):
        solve_single_intruder_heuristic(SMALL_INSTANCE, max_iterations=0)
    with pytest.raises(ValueError, match="positive"):
        solve_single_intruder_heuristic(SMALL_INSTANCE, tolerance=0)
