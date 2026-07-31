import copy
import heapq
import itertools
import math
import tempfile
from pathlib import Path

import pytest
from gurobipy import GRB, GurobiError

from formulations import BUILDERS, solve_instance
from instances import generate_instance, load_instance, save_instance, validate_instance
from run import run_experiments
from utils import load_gurobi_env

SMALL_INSTANCE = Path(__file__).resolve().parents[1] / "instances" / "small_instance.json"
RESULTS_DIR = SMALL_INSTANCE.parents[1] / "results"
EXPECTED_SIZES = {1: (22, 17), 2: (34, 29), 3: (28, 35), 4: (20, 13)}


@pytest.fixture(scope="module")
def solver_env():
    try:
        env = load_gurobi_env()
    except GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required: {error}")
    yield env
    env.dispose()


@pytest.fixture
def workspace_tmp_dir():
    RESULTS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_", dir=RESULTS_DIR) as directory:
        yield Path(directory)


@pytest.fixture(scope="module")
def integer_results(solver_env):
    return {
        formulation: solve_instance(
            SMALL_INSTANCE, formulation=formulation, relax=False, env=solver_env
        )
        for formulation in BUILDERS
    }


@pytest.fixture(scope="module")
def relaxation_results(solver_env):
    return {
        formulation: solve_instance(
            SMALL_INSTANCE, formulation=formulation, relax=True, env=solver_env
        )
        for formulation in BUILDERS
    }


def _path_exists(nodes, edges, source, target, blocked):
    outgoing = {node: [] for node in nodes}
    for edge in edges:
        if edge not in blocked:
            outgoing[edge[0]].append(edge[1])
    visited = {source}
    pending = [source]
    while pending:
        node = pending.pop()
        if node == target:
            return True
        for successor in outgoing[node]:
            if successor not in visited:
                visited.add(successor)
                pending.append(successor)
    return False


def _distance(nodes, edges, costs, source, target):
    outgoing = {node: [] for node in nodes}
    for edge in edges:
        outgoing[edge[0]].append(edge)
    distances = {node: math.inf for node in nodes}
    distances[source] = 0.0
    pending = [(0.0, source)]
    while pending:
        distance, node = heapq.heappop(pending)
        if distance > distances[node] + 1e-12:
            continue
        if node == target:
            return distance
        for edge in outgoing[node]:
            candidate = distance + costs[edge]
            if candidate < distances[edge[1]] - 1e-12:
                distances[edge[1]] = candidate
                heapq.heappush(pending, (candidate, edge[1]))
    return math.inf


def _independent_oracle(instance):
    nodes = instance["nodes"]
    edges = [(edge["tail"], edge["head"]) for edge in instance["edges"]]
    transit = {
        edge: instance["edges"][position]["transit_time"]
        for position, edge in enumerate(edges)
    }
    inspection = {
        edge: instance["edges"][position]["inspection_time"]
        for position, edge in enumerate(edges)
    }
    feasible = []
    for size in range(instance["budget"] + 1):
        for allocation in itertools.combinations(edges, size):
            selected = set(allocation)
            if any(
                _path_exists(
                    nodes,
                    edges,
                    intruder["source"],
                    intruder["target"],
                    selected,
                )
                for intruder in instance["intruders"]
            ):
                continue
            costs = {
                edge: transit[edge] + inspection[edge] * float(edge in selected)
                for edge in edges
            }
            objective = sum(
                _distance(
                    nodes,
                    edges,
                    costs,
                    journeyer["source"],
                    journeyer["target"],
                )
                for journeyer in instance["journeyers"]
            )
            feasible.append((objective, tuple(sorted(selected))))
    optimum = min(value for value, _ in feasible)
    optimal = [
        allocation
        for value, allocation in feasible
        if abs(value - optimum) <= 1e-9
    ]
    return optimum, optimal


def test_small_instance_has_independently_verified_unique_optimum():
    instance = load_instance(SMALL_INSTANCE)
    objective, allocations = _independent_oracle(instance)
    assert validate_instance(instance)
    assert math.isclose(objective, 4.0, abs_tol=1e-9)
    assert allocations == [((1, 3), (2, 3))]


@pytest.mark.parametrize("formulation,builder", sorted(BUILDERS.items()))
def test_model_structure_matches_formulation_document(solver_env, formulation, builder):
    model, variables = builder(SMALL_INSTANCE, env=solver_env)
    expected_variables, expected_constraints = EXPECTED_SIZES[formulation]
    assert model.NumVars == expected_variables
    assert model.NumConstrs == expected_constraints
    if formulation == 1:
        assert model.NumQNZs > 0
        assert model.Params.NonConvex == 2
        assert set(variables) == {"x", "y", "z"}
    elif formulation in (2, 3):
        assert model.NumQNZs == 0
    else:
        assert model.Params.LazyConstraints == 1
    model.dispose()


@pytest.mark.parametrize("formulation,builder", sorted(BUILDERS.items()))
def test_relaxation_makes_discrete_variables_continuous(
    solver_env, formulation, builder
):
    model, _ = builder(SMALL_INSTANCE, relax=True, env=solver_env)
    assert all(variable.VType == GRB.CONTINUOUS for variable in model.getVars())
    assert all(variable.LB >= -1e-12 for variable in model.getVars())
    model.dispose()


def test_all_integer_formulations_reproduce_known_solution(integer_results):
    for result in integer_results.values():
        selected = {
            edge
            for edge, value in result["variables"]["x"].items()
            if value >= 0.5
        }
        assert result["status_name"] == "OPTIMAL"
        assert math.isclose(result["objective_value"], 4.0, abs_tol=1e-6)
        assert selected == {(1, 3), (2, 3)}


def test_relaxation_values_are_stable(relaxation_results):
    expected = {1: 4.0, 2: 3.25, 3: 3.0, 4: 3.25}
    for formulation, result in relaxation_results.items():
        assert result["status_name"] == "OPTIMAL"
        assert math.isclose(
            result["objective_value"], expected[formulation], abs_tol=1e-6
        )


def test_unified_runner_writes_reproducible_metadata(
    solver_env, workspace_tmp_dir
):
    output = workspace_tmp_dir / "comparison.csv"
    experiment = run_experiments(
        [SMALL_INSTANCE],
        repetitions=1,
        csv_filename=output,
        mode="both",
        env=solver_env,
    )
    assert len(experiment["rows"]) == 8
    assert output.exists()
    assert all(row["instance"] == "small_instance" for row in experiment["rows"])
    assert all(row["validation_passed"] for row in experiment["rows"])
    assert {"python_version", "gurobi_version", "solver_seed", "threads"}.issubset(
        experiment["rows"][0]
    )


def test_instance_generation_is_reproducible_and_round_trips(workspace_tmp_dir):
    first = generate_instance(num_nodes=8, num_edges=18, seed=17)
    second = generate_instance(num_nodes=8, num_edges=18, seed=17)
    output = workspace_tmp_dir / "instance.json"
    save_instance(first, output)
    assert first == second
    assert load_instance(output) == first


def test_instance_validator_rejects_inconsistent_edge():
    invalid = copy.deepcopy(generate_instance(seed=3))
    invalid["edges"][0]["inspection_time"] = -1
    with pytest.raises(ValueError):
        validate_instance(invalid)
