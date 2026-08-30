import argparse
import json
import math
import random
from pathlib import Path


def _directed_path_exists(nodes, edges, source, target, blocked_edges=None):
    blocked_edges = set() if blocked_edges is None else set(blocked_edges)
    outgoing = {v: [] for v in nodes}

    for edge in edges:
        if edge not in blocked_edges:
            outgoing[edge[0]].append(edge[1])

    visited = {source}
    pending = [source]

    while pending:
        v = pending.pop()
        if v == target:
            return True

        for v_prime in outgoing[v]:
            if v_prime not in visited:
                visited.add(v_prime)
                pending.append(v_prime)

    return False


def _underlying_graph_is_connected(nodes, edges):
    adjacent = {v: [] for v in nodes}

    for v, v_prime in edges:
        adjacent[v].append(v_prime)
        adjacent[v_prime].append(v)

    visited = {nodes[0]}
    pending = [nodes[0]]

    while pending:
        v = pending.pop()

        for v_prime in adjacent[v]:
            if v_prime not in visited:
                visited.add(v_prime)
                pending.append(v_prime)

    return len(visited) == len(nodes)


def validate_instance(instance):
    required_fields = [
        "nodes",
        "edges",
        "intruders",
        "journeyers",
        "budget",
    ]

    for field in required_fields:
        if field not in instance:
            raise ValueError(f"Missing field: {field}")

    if instance.get("directed", True) is not True:
        raise ValueError("The four formulations expect a directed graph")

    nodes = instance["nodes"]
    if len(nodes) < 2 or len(nodes) != len(set(nodes)):
        raise ValueError("The node list must contain at least two distinct nodes")

    node_set = set(nodes)
    edges = []

    for edge_data in instance["edges"]:
        for field in [
            "tail",
            "head",
            "transit_time",
            "inspection_time",
            "checkpoint_cost",
        ]:
            if field not in edge_data:
                raise ValueError(f"Every edge must define {field}")

        edge = (edge_data["tail"], edge_data["head"])
        if edge[0] not in node_set or edge[1] not in node_set:
            raise ValueError(f"Edge {edge} has an unknown endpoint")
        if edge[0] == edge[1]:
            raise ValueError(f"Self loops are not supported: {edge}")
        if edge in edges:
            raise ValueError(f"Parallel directed edges are not supported: {edge}")
        for field in ["transit_time", "inspection_time", "checkpoint_cost"]:
            value = edge_data[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be numeric")  # noqa: TRY004
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")

        edges.append(edge)

    if not edges:
        raise ValueError("The instance must contain at least one edge")
    if not _underlying_graph_is_connected(nodes, edges):
        raise ValueError("The underlying undirected graph must be connected")

    checkpoint_cost = {
        (edge_data["tail"], edge_data["head"]): edge_data["checkpoint_cost"]
        for edge_data in instance["edges"]
    }
    maximum_budget = sum(checkpoint_cost.values())
    budget = instance["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ValueError("The budget must be an integer")  # noqa: TRY004
    if budget < 0 or budget > maximum_budget:
        raise ValueError(
            "The budget must be between 0 and the total checkpoint cost"
        )

    for group_name in ["intruders", "journeyers"]:
        ids = set()
        pairs = set()

        for agent in instance[group_name]:
            for field in ["id", "source", "target"]:
                if field not in agent:
                    raise ValueError(f"Every {group_name} must define {field}")

            if agent["id"] in ids:
                raise ValueError("Repeated {} id: {}".format(group_name, agent["id"]))
            if agent["source"] not in node_set or agent["target"] not in node_set:
                raise ValueError(f"{agent} has an unknown endpoint")
            if agent["source"] == agent["target"]:
                raise ValueError(f"{group_name} source and target must differ")

            pair = (agent["source"], agent["target"])
            # Repeated journeyer pairs represent separate travelers and therefore
            # contribute multiplicity to the objective. Some legacy instances
            # intentionally contain them. Repeated intruders are merely redundant.
            if group_name == "intruders" and pair in pairs:
                raise ValueError(f"Repeated ordered pair in {group_name}: {pair}")

            ids.add(agent["id"])
            pairs.add(pair)

            if (
                group_name == "journeyers"
                and not _directed_path_exists(nodes, edges, pair[0], pair[1])
            ):
                raise ValueError("No directed path exists for {} {}".format(group_name, agent["id"]))

    if "known_feasible_checkpoints" in instance:
        checkpoints = [tuple(edge) for edge in instance["known_feasible_checkpoints"]]

        if any(edge not in edges for edge in checkpoints):
            raise ValueError("The known feasible checkpoint set contains an unknown edge")
        if sum(checkpoint_cost[edge] for edge in checkpoints) > budget:
            raise ValueError("The known feasible checkpoint set exceeds the budget")

        for intruder in instance["intruders"]:
            if _directed_path_exists(
                nodes,
                edges,
                intruder["source"],
                intruder["target"],
                blocked_edges=checkpoints,
            ):
                raise ValueError("The known checkpoint set does not deter every intruder")

    if "known_optimal_checkpoints" in instance:
        checkpoints = [tuple(edge) for edge in instance["known_optimal_checkpoints"]]

        if any(edge not in edges for edge in checkpoints):
            raise ValueError("The known optimal checkpoint set contains an unknown edge")
        if sum(checkpoint_cost[edge] for edge in checkpoints) > budget:
            raise ValueError("The known optimal checkpoint set exceeds the budget")

        for intruder in instance["intruders"]:
            if _directed_path_exists(
                nodes,
                edges,
                intruder["source"],
                intruder["target"],
                blocked_edges=checkpoints,
            ):
                raise ValueError("The known optimal checkpoint set is not feasible")

    if "known_optimum" in instance:
        known_optimum = instance["known_optimum"]
        if (
            isinstance(known_optimum, bool)
            or not isinstance(known_optimum, (int, float))
            or not math.isfinite(known_optimum)
            or known_optimum < 0
        ):
            raise ValueError("The known optimum must be finite and nonnegative")

    return True


def load_instance(filename):
    with open(filename, "r", encoding="utf-8") as file:
        instance = json.load(file)

    validate_instance(instance)

    return instance


def save_instance(instance, filename):
    validate_instance(instance)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(instance, file, indent=4)
        file.write("\n")


def prepare_instance(instance):
    if isinstance(instance, (str, Path)):
        instance = load_instance(instance)
    else:
        validate_instance(instance)

    nodes = list(instance["nodes"])
    edges = [(edge["tail"], edge["head"]) for edge in instance["edges"]]
    intruders = [agent["id"] for agent in instance["intruders"]]
    journeyers = [agent["id"] for agent in instance["journeyers"]]

    tau = {
        edges[k]: float(instance["edges"][k]["transit_time"])
        for k in range(len(edges))
    }
    inspection_time = {
        edges[k]: float(instance["edges"][k]["inspection_time"])
        for k in range(len(edges))
    }
    checkpoint_cost = {
        edges[k]: float(instance["edges"][k]["checkpoint_cost"])
        for k in range(len(edges))
    }
    intruder_source = {
        agent["id"]: agent["source"]
        for agent in instance["intruders"]
    }
    intruder_target = {
        agent["id"]: agent["target"]
        for agent in instance["intruders"]
    }
    journeyer_source = {
        agent["id"]: agent["source"]
        for agent in instance["journeyers"]
    }
    journeyer_target = {
        agent["id"]: agent["target"]
        for agent in instance["journeyers"]
    }
    balance = {}

    for j in journeyers:
        for v in nodes:
            balance[j, v] = 0

        balance[j, journeyer_source[j]] = -1
        balance[j, journeyer_target[j]] = 1

    return {
        "instance": instance,
        "nodes": nodes,
        "edges": edges,
        "intruders": intruders,
        "journeyers": journeyers,
        "tau": tau,
        "inspection_time": inspection_time,
        "checkpoint_cost": checkpoint_cost,
        "intruder_source": intruder_source,
        "intruder_target": intruder_target,
        "journeyer_source": journeyer_source,
        "journeyer_target": journeyer_target,
        "balance": balance,
        "budget": instance["budget"],
    }


def generate_instance(
    num_nodes=10,
    num_edges=None,
    num_intruders=2,
    num_journeyers=3,
    budget=None,
    seed=0,
    name=None,
):
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2")
    if num_intruders < 1 or num_intruders > num_nodes - 1:
        raise ValueError("num_intruders must be between 1 and num_nodes - 1")
    if num_journeyers < 1 or num_journeyers > num_nodes * (num_nodes - 1):
        raise ValueError("Invalid number of journeyers")

    if num_edges is None:
        num_edges = min(3 * num_nodes, num_nodes * (num_nodes - 1))
    if num_edges < num_nodes or num_edges > num_nodes * (num_nodes - 1):
        raise ValueError(
            "num_edges must be between num_nodes and num_nodes * (num_nodes - 1)"
        )

    rng = random.Random(seed)
    nodes = list(range(num_nodes))
    coordinates = {
        v: (rng.random(), rng.random())
        for v in nodes
    }

    # A directed Hamiltonian cycle guarantees strong connectivity.
    cycle = list(nodes)
    rng.shuffle(cycle)
    edges = []

    for k in range(num_nodes):
        edge = (cycle[k], cycle[(k + 1) % num_nodes])
        edges.append(edge)

    candidates = [
        (v, v_prime)
        for v in nodes
        for v_prime in nodes
        if v != v_prime and (v, v_prime) not in edges
    ]
    edges.extend(rng.sample(candidates, num_edges - num_nodes))

    indegree = {
        v: sum(1 for edge in edges if edge[1] == v)
        for v in nodes
    }

    if budget is None:
        budget = max(min(indegree.values()), math.ceil(0.3 * num_edges))
        budget = min(budget, num_edges)
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ValueError("budget must be an integer")  # noqa: TRY004
    if budget < 0 or budget > num_edges:
        raise ValueError("budget must be between 0 and num_edges")

    possible_targets = [v for v in nodes if indegree[v] <= budget]
    if not possible_targets:
        raise ValueError(
            "The requested budget is too small for the generated feasibility certificate"
        )

    common_target = rng.choice(possible_targets)
    possible_sources = [v for v in nodes if v != common_target]
    intruder_sources = rng.sample(possible_sources, num_intruders)

    ordered_pairs = [
        (v, v_prime)
        for v in nodes
        for v_prime in nodes
        if v != v_prime
    ]
    journeyer_pairs = rng.sample(ordered_pairs, num_journeyers)

    edge_data = []
    for v, v_prime in edges:
        x_v, y_v = coordinates[v]
        x_prime, y_prime = coordinates[v_prime]
        transit_time = max(0.0001, math.hypot(x_v - x_prime, y_v - y_prime))
        transit_time = round(transit_time, 4)
        inspection_time = round(transit_time * rng.random(), 4)

        edge_data.append(
            {
                "tail": v,
                "head": v_prime,
                "transit_time": transit_time,
                "inspection_time": inspection_time,
                "checkpoint_cost": 1.0,
            }
        )

    intruders = [
        {
            "id": k,
            "source": intruder_sources[k],
            "target": common_target,
        }
        for k in range(num_intruders)
    ]
    journeyers = [
        {
            "id": k,
            "source": journeyer_pairs[k][0],
            "target": journeyer_pairs[k][1],
        }
        for k in range(num_journeyers)
    ]
    known_feasible_checkpoints = [
        [edge[0], edge[1]]
        for edge in edges
        if edge[1] == common_target
    ]

    if name is None:
        name = f"generated_{num_nodes}_{num_edges}_{num_intruders}_{num_journeyers}_seed_{seed}"

    instance = {
        "name": name,
        "seed": seed,
        "directed": True,
        "nodes": nodes,
        "coordinates": {
            str(v): [coordinates[v][0], coordinates[v][1]]
            for v in nodes
        },
        "edges": edge_data,
        "intruders": intruders,
        "journeyers": journeyers,
        "budget": budget,
        "known_feasible_checkpoints": known_feasible_checkpoints,
    }

    validate_instance(instance)

    return instance


def main():
    parser = argparse.ArgumentParser(description="Generate a reproducible SNI instance")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edges", type=int, default=None)
    parser.add_argument("--intruders", type=int, default=2)
    parser.add_argument("--journeyers", type=int, default=3)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default="instances/generated_instance.json")
    args = parser.parse_args()

    instance = generate_instance(
        num_nodes=args.nodes,
        num_edges=args.edges,
        num_intruders=args.intruders,
        num_journeyers=args.journeyers,
        budget=args.budget,
        seed=args.seed,
        name=args.name,
    )
    save_instance(instance, args.output)

    print(f"Instance saved in {args.output}")


if __name__ == "__main__":
    main()
