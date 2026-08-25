import copy
import tempfile
from pathlib import Path

import pytest

from generate_grid_instances import (
    CHECKPOINT_COST,
    EPSILON,
    GRID_SPECS,
    INSPECTION_TIME_MAX,
    INSPECTION_TIME_MIN,
    INTRUDER_COUNTS,
    JOURNEYER_COUNTS,
    MAX_INTRUDERS,
    MAX_JOURNEYERS,
    TRANSIT_TIME_MAX,
    build_grid_instance,
    build_grid_master,
    generate_grid_collection,
    save_grid_collection,
)
from instances import load_instance, prepare_instance, validate_instance

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def collection():
    return generate_grid_collection()


@pytest.fixture
def workspace_tmp_dir():
    RESULTS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test_grid_", dir=RESULTS_DIR) as directory:
        yield Path(directory)


def _edge_lookup(edges):
    return {(edge["tail"], edge["head"]): edge for edge in edges}


def _path_exists(instance, source, target, blocked):
    outgoing = {node: [] for node in instance["nodes"]}
    for edge in instance["edges"]:
        pair = (edge["tail"], edge["head"])
        if pair not in blocked:
            outgoing[pair[0]].append(pair[1])

    pending = [source]
    visited = {source}
    while pending:
        node = pending.pop()
        if node == target:
            return True
        for neighbor in outgoing[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return False


@pytest.mark.parametrize("rows,columns,budget", GRID_SPECS)
def test_master_grid_has_expected_structure(rows, columns, budget):
    master = build_grid_master(rows, columns, budget)
    expected_physical_edges = rows * (columns - 1) + columns * (rows - 1)

    assert len(master["nodes"]) == rows * columns
    assert len(master["edges"]) == 2 * expected_physical_edges
    assert len(_edge_lookup(master["edges"])) == len(master["edges"])
    assert master["budget"] == rows == budget

    for edge in master["edges"]:
        tail_row, tail_column = divmod(edge["tail"], columns)
        head_row, head_column = divmod(edge["head"], columns)
        assert abs(tail_row - head_row) + abs(tail_column - head_column) == 1


@pytest.mark.parametrize("rows,columns,budget", GRID_SPECS)
def test_opposite_arcs_share_valid_parameters(rows, columns, budget):
    master = build_grid_master(rows, columns, budget)
    lookup = _edge_lookup(master["edges"])

    for (tail, head), edge in lookup.items():
        reverse = lookup[head, tail]
        assert EPSILON <= edge["transit_time"] <= TRANSIT_TIME_MAX
        assert INSPECTION_TIME_MIN <= edge["inspection_time"] <= INSPECTION_TIME_MAX
        assert edge["checkpoint_cost"] == CHECKPOINT_COST
        assert edge["transit_time"] == reverse["transit_time"]
        assert edge["inspection_time"] == reverse["inspection_time"]
        assert edge["checkpoint_cost"] == reverse["checkpoint_cost"]


@pytest.mark.parametrize("rows,columns,budget", GRID_SPECS)
def test_intruder_terminals_are_distinct_and_in_opposite_halves(
    rows, columns, budget
):
    master = build_grid_master(rows, columns, budget)
    sources = [agent["source"] for agent in master["intruders"]]
    targets = [agent["target"] for agent in master["intruders"]]

    assert len(master["intruders"]) == MAX_INTRUDERS
    assert len(set(sources + targets)) == 2 * MAX_INTRUDERS
    assert all(source % columns < columns // 2 for source in sources)
    assert all(target % columns >= (columns + 1) // 2 for target in targets)
    if columns % 2:
        middle_column = columns // 2
        assert all(node % columns != middle_column for node in sources + targets)


@pytest.mark.parametrize("rows,columns,budget", GRID_SPECS)
def test_master_journeyer_pairs_are_distinct(rows, columns, budget):
    master = build_grid_master(rows, columns, budget)
    pairs = [
        (agent["source"], agent["target"]) for agent in master["journeyers"]
    ]

    assert len(pairs) == MAX_JOURNEYERS
    assert len(set(pairs)) == MAX_JOURNEYERS
    assert all(source != target for source, target in pairs)


@pytest.mark.parametrize("rows,columns,budget", GRID_SPECS)
def test_row_budget_has_a_feasible_grid_cut(rows, columns, budget):
    master = build_grid_master(rows, columns, budget)
    instance = build_grid_instance(master, MAX_INTRUDERS, MAX_JOURNEYERS)
    left_boundary_column = columns // 2 - 1
    blocked = {
        (
            row * columns + left_boundary_column,
            row * columns + left_boundary_column + 1,
        )
        for row in range(rows)
    }

    assert len(blocked) == budget
    assert all(
        not _path_exists(
            instance,
            intruder["source"],
            intruder["target"],
            blocked,
        )
        for intruder in instance["intruders"]
    )


def test_collection_contains_45_valid_unique_instances(collection):
    expected_count = len(GRID_SPECS) * len(INTRUDER_COUNTS) * len(JOURNEYER_COUNTS)
    assert expected_count == 45
    assert len(collection) == expected_count
    assert len({instance["name"] for instance in collection}) == expected_count
    assert all(validate_instance(instance) for instance in collection)


@pytest.mark.parametrize("rows,columns,budget", GRID_SPECS)
def test_collection_uses_nested_population_prefixes(
    collection, rows, columns, budget
):
    variants = {
        (len(instance["intruders"]), len(instance["journeyers"])): instance
        for instance in collection
        if instance["grid"] == {"rows": rows, "columns": columns}
    }
    largest = variants[MAX_INTRUDERS, MAX_JOURNEYERS]

    assert len(variants) == len(INTRUDER_COUNTS) * len(JOURNEYER_COUNTS)
    for num_intruders in INTRUDER_COUNTS:
        for num_journeyers in JOURNEYER_COUNTS:
            instance = variants[num_intruders, num_journeyers]
            assert instance["edges"] == largest["edges"]
            assert instance["intruders"] == largest["intruders"][:num_intruders]
            assert instance["journeyers"] == largest["journeyers"][:num_journeyers]


def test_generation_is_reproducible_and_component_seeded():
    first = build_grid_master(5, 10, 5, seed=0)
    second = build_grid_master(5, 10, 5, seed=0)
    changed = build_grid_master(5, 10, 5, seed=1)

    assert first == second
    assert first["edges"] != changed["edges"]
    assert first["intruders"] != changed["intruders"]
    assert first["journeyers"] != changed["journeyers"]


def test_checkpoint_cost_is_required_validated_and_prepared():
    instance = build_grid_instance(build_grid_master(5, 10, 5), 1, 10)
    prepared = prepare_instance(instance)
    assert set(prepared["checkpoint_cost"].values()) == {CHECKPOINT_COST}

    missing = copy.deepcopy(instance)
    del missing["edges"][0]["checkpoint_cost"]
    with pytest.raises(ValueError, match="checkpoint_cost"):
        validate_instance(missing)

    negative = copy.deepcopy(instance)
    negative["edges"][0]["checkpoint_cost"] = -1.0
    with pytest.raises(ValueError, match="checkpoint_cost"):
        validate_instance(negative)


def test_collection_materialization_and_overwrite_protection(workspace_tmp_dir):
    instances = generate_grid_collection()[:2]
    destinations = save_grid_collection(instances, workspace_tmp_dir)

    assert len(destinations) == 2
    assert all(destination.exists() for destination in destinations)
    assert load_instance(destinations[0]) == instances[0]
    with pytest.raises(FileExistsError, match="--overwrite"):
        save_grid_collection(instances, workspace_tmp_dir)
    assert (
        save_grid_collection(instances, workspace_tmp_dir, overwrite=True)
        == destinations
    )


@pytest.mark.parametrize(
    "arguments,error",
    [
        ((5.0, 10, 5), TypeError),
        ((5, 10.0, 5), TypeError),
        ((5, 10, 5.0), TypeError),
        ((1, 10, 1), ValueError),
        ((2, 2, 2), ValueError),
    ],
)
def test_invalid_master_parameters_are_rejected(arguments, error):
    with pytest.raises(error):
        build_grid_master(*arguments)
