import json
import re
from pathlib import Path

import pytest
from gurobipy import GurobiError

from branch_and_cut import directed_min_cut
from formulations import solve_instance
from instances import load_instance
from utils import load_gurobi_env

INSTANCES_DIR = Path(__file__).resolve().parents[1] / "instances"
SINGLE_INSTANCE = INSTANCES_DIR / "single50_272_10_3_1dir.json"
SINGLE_INSTANCE_PATTERN = re.compile(
    r"^single(?P<nodes>\d+)_(?P<edges>\d+)_(?P<journeyers>\d+)_"
    r"(?P<cluster_pairs>\d+)_(?P<index>\d+)dir\.json$"
)


@pytest.fixture(scope="module")
def solver_env():
    try:
        env = load_gurobi_env()
    except GurobiError as error:
        pytest.skip(f"A usable Gurobi license is required: {error}")
    yield env
    env.dispose()


def test_single_intruder_collection_preserves_legacy_construction():
    paths = sorted(INSTANCES_DIR.glob("single*.json"))

    assert len(paths) == 42
    assert all(path.stem.endswith("dir") for path in paths)

    for path in paths:
        match = SINGLE_INSTANCE_PATTERN.fullmatch(path.name)
        assert match is not None

        original_node_count = int(match["nodes"])
        original_edge_count = int(match["edges"])
        expected_journeyers = int(match["journeyers"])
        cluster_pairs = int(match["cluster_pairs"])
        instance = json.loads(path.read_text(encoding="utf-8"))
        super_source = original_node_count
        super_target = original_node_count + 1

        assert instance["name"] == path.stem
        assert instance["directed"] is True
        assert instance["nodes"] == list(range(original_node_count + 2))
        assert instance["intruders"] == [
            {"id": 0, "source": super_source, "target": super_target}
        ]
        assert len(instance["journeyers"]) == expected_journeyers
        assert [agent["id"] for agent in instance["journeyers"]] == list(
            range(expected_journeyers)
        )
        assert all(
            agent["source"] < original_node_count
            and agent["target"] < original_node_count
            for agent in instance["journeyers"]
        )

        ordinary_edges = instance["edges"][:original_edge_count]
        auxiliary_edges = instance["edges"][original_edge_count:]
        outgoing_source = [
            edge for edge in auxiliary_edges if edge["tail"] == super_source
        ]
        incoming_target = [
            edge for edge in auxiliary_edges if edge["head"] == super_target
        ]

        assert ordinary_edges
        assert auxiliary_edges
        assert len(outgoing_source) <= cluster_pairs
        assert len(incoming_target) <= cluster_pairs
        assert len(auxiliary_edges) == len(outgoing_source) + len(incoming_target)
        assert all(
            edge["tail"] < original_node_count
            and edge["head"] < original_node_count
            and edge["checkpoint_cost"] == 1.0
            for edge in ordinary_edges
        )
        assert all(
            (
                edge["tail"] == super_source
                and edge["head"] < original_node_count
            )
            or (
                edge["tail"] < original_node_count
                and edge["head"] == super_target
            )
            for edge in auxiliary_edges
        )
        assert all(
            edge["transit_time"] == 0.0
            and edge["inspection_time"] == 0.0
            and edge["checkpoint_cost"] == instance["budget"] + 1
            for edge in auxiliary_edges
        )


def test_representative_single_intruder_instance_has_nontrivial_feasible_cut():
    instance = load_instance(SINGLE_INSTANCE)
    intruder = instance["intruders"][0]
    edges = [(edge["tail"], edge["head"]) for edge in instance["edges"]]
    capacities = {
        (edge["tail"], edge["head"]): edge["checkpoint_cost"]
        for edge in instance["edges"]
    }
    protected_edges = {
        edge
        for edge in edges
        if edge[0] == intruder["source"] or edge[1] == intruder["target"]
    }

    cut_value, _, cut_edges = directed_min_cut(
        instance["nodes"],
        edges,
        capacities,
        intruder["source"],
        intruder["target"],
    )

    assert len(instance["nodes"]) == 52
    assert len(instance["edges"]) == 278
    assert len(instance["journeyers"]) == 10
    assert instance["budget"] == 16
    assert len(protected_edges) == 6
    assert {capacities[edge] for edge in protected_edges} == {17.0}
    assert cut_value == 13.0
    assert cut_value <= instance["budget"]
    assert protected_edges.isdisjoint(cut_edges)


def test_representative_single_intruder_instance_reproduces_published_objective(
    solver_env,
):
    result = solve_instance(
        SINGLE_INSTANCE,
        formulation=2,
        relax=False,
        solver_seed=0,
        threads=1,
        env=solver_env,
    )

    assert result["status_name"] == "OPTIMAL"
    assert result["objective_value"] == pytest.approx(
        7779.018444281898,
        abs=1e-6,
    )
