import copy
import itertools
import json
import random
from pathlib import Path

import pytest
from gurobipy import GRB, GurobiError

import create_complex_instances as complex_generator
from create_complex_instances import (
    assign_checkpoint_costs,
    build_minimum_checkpoint_model,
    complex_instance_path,
    create_complex_instance,
    create_complex_instances,
    discover_complex_sources,
    minimum_feasible_budget,
)
from formulations import BUILDERS
from generate_grid_instances import build_grid_instance, build_grid_master
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


def _auxiliary_chain():
    return {
        "name": "single_chain_dir",
        "directed": True,
        "nodes": [0, 1, 2, 3],
        "edges": [
            {"tail": 0, "head": 1, "transit_time": 0, "inspection_time": 0,
             "checkpoint_cost": 2},
            {"tail": 1, "head": 2, "transit_time": 1, "inspection_time": 2,
             "checkpoint_cost": 1},
            {"tail": 2, "head": 3, "transit_time": 0, "inspection_time": 0,
             "checkpoint_cost": 2},
        ],
        "intruders": [{"id": 0, "source": 0, "target": 3}],
        "journeyers": [{"id": 0, "source": 1, "target": 2}],
        "budget": 1,
    }


@pytest.mark.parametrize("budget_model", ["disaggregated", "intruder-cuts"])
@pytest.mark.parametrize("ordinary_cost", [0, 10])
def test_auxiliary_arcs_stay_uncheckable_after_budget_changes(
    tmp_path, solver_env, budget_model, ordinary_cost
):
    original = _auxiliary_chain()
    source = tmp_path / "single_chain_dir.json"
    save_instance(original, source)
    weighted, result = create_complex_instance(
        source, cost_min=ordinary_cost, cost_max=ordinary_cost,
        budget_model=budget_model, env=solver_env,
    )
    assert result["optimal"]
    assert result["budget"] == ordinary_cost
    assert weighted["known_feasible_checkpoints"] == [[1, 2]]
    assert [e["checkpoint_cost"] for e in weighted["edges"]] == [
        ordinary_cost + 1, ordinary_cost, ordinary_cost + 1,
    ]
    assert weighted["complex_generation"]["time_limit"] == 600
    assert weighted["complex_generation"]["uncheckable_edge_count"] == 2
    assert validate_instance(weighted)
    assert load_instance(source) == original


@pytest.mark.parametrize("budget_model", ["disaggregated", "intruder-cuts"])
def test_protected_start_and_budget_match_exhaustive_feasible_allocations(
    solver_env, budget_model
):
    instance = _weighted_small_instance()
    protected = {(0, 1)}
    model, x, _ = build_minimum_checkpoint_model(
        instance, forbidden_edges=protected, budget_model=budget_model, env=solver_env,
    )
    try:
        assert x[0, 1].UB == 0
        assert x[0, 1].Start == 0
        instance["budget"] = sum(e["checkpoint_cost"] for e in instance["edges"])
        assert evaluate_allocation(instance, {e: v.Start for e, v in x.items()})["valid"]
    finally:
        model.dispose()
    edges = [(e["tail"], e["head"]) for e in instance["edges"]]
    allowed = [e for e in edges if e not in protected]
    costs = {(e["tail"], e["head"]): e["checkpoint_cost"] for e in instance["edges"]}
    feasible_costs = []
    for mask in itertools.product((0, 1), repeat=len(allowed)):
        allocation = dict.fromkeys(edges, 0)
        allocation.update(zip(allowed, mask))
        if evaluate_allocation(instance, allocation)["valid"]:
            feasible_costs.append(sum(costs[e] * allocation[e] for e in edges))
    result = minimum_feasible_budget(
        instance, forbidden_edges=protected, budget_model=budget_model, env=solver_env,
    )
    assert result["optimal"]
    assert result["budget"] == min(feasible_costs)
    assert protected.isdisjoint(result["selected_edges"])


def test_uncheckable_only_path_fails_before_constructing_solver(monkeypatch):
    def unexpected_model(*args, **kwargs):
        pytest.fail("An infeasible protected graph must be detected before the MIP")

    monkeypatch.setattr(complex_generator, "Model", unexpected_model)
    with pytest.raises(ValueError, match="path of only uncheckable edges"):
        minimum_feasible_budget(
            _auxiliary_chain(), forbidden_edges={(0, 1), (1, 2), (2, 3)},
        )


def test_collection_discovery_paths_and_overwrite_preflight(tmp_path, monkeypatch):
    originals = [
        tmp_path / "single_chain_dir.json",
        tmp_path / "extra_large" / "single_large_dir.json",
        tmp_path / "grid_collection" / "grid_5x10_i1_j10_seed0.json",
    ]
    for path in originals:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "legacy_dir.json").write_text("{}", encoding="utf-8")
    destinations = [complex_instance_path(path) for path in originals]
    assert [p.name for p in destinations] == [
        "single_chain_c.json", "single_large_c.json", "grid_5x10_i1_j10_seed0_c.json",
    ]
    assert all(a.parent == b.parent for a, b in zip(originals, destinations))
    destinations[-1].write_text("already generated", encoding="utf-8")
    for percentage in (120, 150):
        variant = destinations[-1].with_name(f"{destinations[-1].stem}_b{percentage}.json")
        variant.write_text("budget variant", encoding="utf-8")
    assert set(discover_complex_sources(tmp_path, collection="single-grid")) == set(originals)
    assert discover_complex_sources(
        tmp_path / "grid_collection", pattern="grid_*.json"
    ) == [originals[-1]]
    def unexpected_load(*args, **kwargs):
        pytest.fail("Overwrite conflicts must be detected before loading/solving")
    monkeypatch.setattr(complex_generator, "load_instance", unexpected_load)
    with pytest.raises(FileExistsError, match="--overwrite"):
        create_complex_instances(tmp_path, collection="single-grid")
    assert not destinations[0].exists()
    assert destinations[-1].read_text(encoding="utf-8") == "already generated"


def _cost_map(instance):
    return {(e["tail"], e["head"]): e["checkpoint_cost"] for e in instance["edges"]}


def test_grid_costs_and_budgets_shared_across_populations_and_subsets(tmp_path, solver_env):
    master = build_grid_master(5, 10, 5)
    for intruders, journeyers in [(1, 10), (1, 50), (5, 10)]:
        instance = build_grid_instance(master, intruders, journeyers)
        if journeyers == 50:
            instance["edges"].reverse()
        save_instance(instance, tmp_path / f"{instance['name']}.json")
    summaries = create_complex_instances(
        tmp_path, pattern="grid_*.json", seed=19, env=solver_env,
    )
    weighted = [load_instance(s["destination"]) for s in summaries]
    maps = [_cost_map(instance) for instance in weighted]
    assert all(costs == maps[0] for costs in maps)
    assert any(value != maps[0][head, tail] for (tail, head), value in maps[0].items())
    assert weighted[0]["budget"] == weighted[1]["budget"] <= weighted[2]["budget"]
    assert summaries[1]["budget_reused_from"] == summaries[0]["source"].name
    assert summaries[2]["budget_reused_from"] is None
    assert not weighted[0]["generation_parameters"]["symmetric_opposite_arcs"]
    assert weighted[0]["generation_parameters"]["symmetric_opposite_arc_times"]
    standalone, _ = create_complex_instance(summaries[1]["source"], seed=19, env=solver_env)
    assert _cost_map(standalone) == maps[0]
    changed, _ = create_complex_instance(summaries[1]["source"], seed=20, env=solver_env)
    assert _cost_map(changed) != maps[0]


def test_single_pairs_share_costs_and_solve_but_not_journeyers(tmp_path, solver_env):
    first = _auxiliary_chain()
    second = copy.deepcopy(first)
    second["journeyers"].append({"id": 1, "source": 1, "target": 2})
    second["edges"].reverse()
    save_instance(first, tmp_path / "single_first_dir.json")
    save_instance(second, tmp_path / "single_second_dir.json")
    summaries = create_complex_instances(tmp_path, seed=31, env=solver_env)
    a, b = [load_instance(s["destination"]) for s in summaries]
    assert _cost_map(a) == _cost_map(b)
    assert a["budget"] == b["budget"]
    assert a["journeyers"] == first["journeyers"]
    assert b["journeyers"] == second["journeyers"]
    assert summaries[1]["budget_reused_from"] == "single_first_dir.json"


def test_rejects_grid_topology_mismatch(tmp_path):
    grid = build_grid_instance(build_grid_master(5, 10, 5), 1, 10)
    grid["edges"].pop()
    source = tmp_path / f"{grid['name']}.json"
    save_instance(grid, source)
    with pytest.raises(ValueError, match="Grid topology"):
        create_complex_instance(source)


def test_time_limited_result_retains_feasible_budget_and_provenance(tmp_path, monkeypatch):
    source = tmp_path / "single_chain_dir.json"
    save_instance(_auxiliary_chain(), source)
    def limited_solve(instance, **kwargs):
        assert kwargs["time_limit"] == 600
        assert kwargs["forbidden_edges"] == {(0, 1), (2, 3)}
        return {
            "budget": 10, "optimal": False, "status": GRB.TIME_LIMIT,
            "budget_model": "disaggregated", "objective_bound": 5,
            "selected_edges": [(1, 2)], "solver_runtime": 600.01,
        }
    monkeypatch.setattr(complex_generator, "minimum_feasible_budget", limited_solve)
    instance, _ = create_complex_instance(source, cost_min=10, cost_max=10)
    metadata = instance["complex_generation"]
    assert not metadata["optimal"]
    assert metadata["status"] == GRB.TIME_LIMIT
    assert metadata["objective_bound"] == 5
    assert validate_instance(instance)
    assert json.loads(json.dumps(instance)) == instance


def test_materialized_single_and_grid_collection():
    """Audit the delivered JSON files against their originals without a solver."""
    sources = discover_complex_sources(SMALL_INSTANCE.parent, collection="single-grid")
    assert len(sources) == 87
    groups = {}
    solves = 0
    changed_fields = {
        "name", "budget", "edges", "generation_parameters",
        "known_feasible_checkpoints", "known_optimal_checkpoints", "known_optimum",
    }
    for source in sources:
        original = json.loads(source.read_text(encoding="utf-8"))
        destination = complex_instance_path(source)
        weighted = json.loads(destination.read_text(encoding="utf-8"))
        assert weighted["name"] == destination.stem
        for key in original.keys() - changed_fields:
            assert weighted[key] == original[key], (destination.name, key)
        assert len(original["edges"]) == len(weighted["edges"])
        protected = set()
        for old, new in zip(original["edges"], weighted["edges"]):
            assert {k: v for k, v in old.items() if k != "checkpoint_cost"} == {
                k: v for k, v in new.items() if k != "checkpoint_cost"
            }
            assert type(new["checkpoint_cost"]) is int
            if old["checkpoint_cost"] > original["budget"]:
                protected.add((old["tail"], old["head"]))
                assert new["checkpoint_cost"] == weighted["budget"] + 1
            else:
                assert 10 <= new["checkpoint_cost"] <= 50
        costs = _cost_map(weighted)
        certificate = {tuple(e) for e in weighted["known_feasible_checkpoints"]}
        assert certificate <= costs.keys()
        assert certificate.isdisjoint(protected)
        assert sum(costs[e] for e in certificate) == weighted["budget"]
        outgoing = {node: [] for node in weighted["nodes"]}
        for tail, head in costs.keys() - certificate:
            outgoing[tail].append(head)
        for intruder in weighted["intruders"]:
            visited, pending = {intruder["source"]}, [intruder["source"]]
            while pending:
                for head in outgoing[pending.pop()]:
                    if head not in visited:
                        visited.add(head)
                        pending.append(head)
            assert intruder["target"] not in visited, destination.name
        metadata = weighted["complex_generation"]
        assert metadata["source"] == source.name
        assert metadata["cost_seed"] == 0
        assert metadata["time_limit"] == 600
        assert metadata["budget"] == weighted["budget"]
        assert metadata["uncheckable_edge_count"] == len(protected)
        assert metadata["status"] in (GRB.OPTIMAL, GRB.TIME_LIMIT)
        assert metadata["optimal"] == (metadata["status"] == GRB.OPTIMAL)
        assert metadata["objective_bound"] <= weighted["budget"] + 1e-6
        if metadata["optimal"]:
            assert metadata["objective_bound"] == pytest.approx(weighted["budget"])
        solves += metadata["budget_reused_from"] is None
        group = groups.setdefault(metadata["cost_group"], [])
        if group:
            assert costs == group[0], destination.name
        group.append(costs)
    assert solves == 36
    assert sorted(len(group) for group in groups.values()) == [2] * 21 + [9] * 5
