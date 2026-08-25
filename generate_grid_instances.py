import argparse
import hashlib
import random
from pathlib import Path

from instances import save_instance, validate_instance

EPSILON = 0.01
SEED = 0
TRANSIT_TIME_MAX = 1.0
INSPECTION_TIME_MIN = 1.0
INSPECTION_TIME_MAX = 5.0
CHECKPOINT_COST = 1.0
DECIMAL_PLACES = 6

GRID_SPECS = (
    (5, 10, 5),
    (10, 10, 10),
    (10, 20, 10),
    (20, 25, 20),
    (20, 50, 20),
)
INTRUDER_COUNTS = (1, 5, 10)
JOURNEYER_COUNTS = (10, 50, 100)
MAX_INTRUDERS = max(INTRUDER_COUNTS)
MAX_JOURNEYERS = max(JOURNEYER_COUNTS)


def node_id(row, column, columns):
    return row * columns + column


def _component_rng(seed, rows, columns, component):
    material = f"{seed}:{rows}:{columns}:{component}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived_seed)


def _validate_master_parameters(rows, columns, budget, seed):
    for name, value in [("rows", rows), ("columns", columns), ("budget", budget)]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if rows < 2 or columns < 2:
        raise ValueError("rows and columns must both be at least 2")
    if budget < 0:
        raise ValueError("budget must be nonnegative")

    source_pool_size = rows * (columns // 2)
    target_pool_size = rows * (columns // 2)
    if min(source_pool_size, target_pool_size) < MAX_INTRUDERS:
        raise ValueError(
            "Each half of the grid must contain at least "
            f"{MAX_INTRUDERS} nodes"
        )


def _physical_grid_edges(rows, columns):
    for row in range(rows):
        for column in range(columns):
            node = node_id(row, column, columns)
            if column + 1 < columns:
                yield node, node_id(row, column + 1, columns)
            if row + 1 < rows:
                yield node, node_id(row + 1, column, columns)


def _build_edges(rows, columns, seed):
    rng = _component_rng(seed, rows, columns, "edge_costs")
    edges = []

    for node, neighbor in _physical_grid_edges(rows, columns):
        transit_time = round(rng.uniform(EPSILON, TRANSIT_TIME_MAX), DECIMAL_PLACES)
        inspection_time = round(
            rng.uniform(INSPECTION_TIME_MIN, INSPECTION_TIME_MAX),
            DECIMAL_PLACES,
        )
        parameters = {
            "transit_time": transit_time,
            "inspection_time": inspection_time,
            "checkpoint_cost": CHECKPOINT_COST,
        }
        edges.append({"tail": node, "head": neighbor, **parameters})
        edges.append({"tail": neighbor, "head": node, **parameters})

    return edges


def _intruder_terminal_pools(rows, columns):
    source_columns = range(columns // 2)
    target_columns = range((columns + 1) // 2, columns)
    sources = [
        node_id(row, column, columns)
        for row in range(rows)
        for column in source_columns
    ]
    targets = [
        node_id(row, column, columns)
        for row in range(rows)
        for column in target_columns
    ]
    return sources, targets


def _build_intruders(rows, columns, seed):
    rng = _component_rng(seed, rows, columns, "intruders")
    source_pool, target_pool = _intruder_terminal_pools(rows, columns)
    sources = rng.sample(source_pool, MAX_INTRUDERS)
    targets = rng.sample(target_pool, MAX_INTRUDERS)
    return [
        {"id": agent_id, "source": source, "target": target}
        for agent_id, (source, target) in enumerate(zip(sources, targets))
    ]


def _build_journeyers(rows, columns, seed):
    rng = _component_rng(seed, rows, columns, "journeyers")
    num_nodes = rows * columns
    pairs = []
    used_pairs = set()

    while len(pairs) < MAX_JOURNEYERS:
        source = rng.randrange(num_nodes)
        target = rng.randrange(num_nodes - 1)
        if target >= source:
            target += 1
        pair = (source, target)
        if pair not in used_pairs:
            used_pairs.add(pair)
            pairs.append(pair)

    return [
        {"id": agent_id, "source": source, "target": target}
        for agent_id, (source, target) in enumerate(pairs)
    ]


def build_grid_master(rows, columns, budget, seed=SEED):
    """Create one grid and its ordered maximum-size agent populations."""
    _validate_master_parameters(rows, columns, budget, seed)
    nodes = list(range(rows * columns))
    return {
        "rows": rows,
        "columns": columns,
        "budget": budget,
        "seed": seed,
        "nodes": nodes,
        "coordinates": {
            str(node_id(row, column, columns)): [row, column]
            for row in range(rows)
            for column in range(columns)
        },
        "edges": _build_edges(rows, columns, seed),
        "intruders": _build_intruders(rows, columns, seed),
        "journeyers": _build_journeyers(rows, columns, seed),
    }


def _validate_population_size(name, value, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def build_grid_instance(master, num_intruders, num_journeyers):
    """Build an instance by taking prefixes of a grid's master populations."""
    _validate_population_size("num_intruders", num_intruders, len(master["intruders"]))
    _validate_population_size(
        "num_journeyers", num_journeyers, len(master["journeyers"])
    )

    rows = master["rows"]
    columns = master["columns"]
    seed = master["seed"]
    name = (
        f"grid_{rows}x{columns}_i{num_intruders}_j{num_journeyers}_seed{seed}"
    )
    instance = {
        "name": name,
        "seed": seed,
        "directed": True,
        "grid": {"rows": rows, "columns": columns},
        "generation_parameters": {
            "epsilon": EPSILON,
            "transit_time_range": [EPSILON, TRANSIT_TIME_MAX],
            "inspection_time_range": [INSPECTION_TIME_MIN, INSPECTION_TIME_MAX],
            "symmetric_opposite_arcs": True,
            "master_intruders": len(master["intruders"]),
            "master_journeyers": len(master["journeyers"]),
        },
        "nodes": list(master["nodes"]),
        "coordinates": {
            node: list(coordinates)
            for node, coordinates in master["coordinates"].items()
        },
        "edges": [dict(edge) for edge in master["edges"]],
        "intruders": [
            dict(agent) for agent in master["intruders"][:num_intruders]
        ],
        "journeyers": [
            dict(agent) for agent in master["journeyers"][:num_journeyers]
        ],
        "budget": master["budget"],
    }
    validate_instance(instance)
    return instance


def generate_grid_collection(seed=SEED):
    """Return the 45 agreed grid instances without writing files."""
    instances = []
    for rows, columns, budget in GRID_SPECS:
        master = build_grid_master(rows, columns, budget, seed=seed)
        for num_intruders in INTRUDER_COUNTS:
            for num_journeyers in JOURNEYER_COUNTS:
                instances.append(
                    build_grid_instance(master, num_intruders, num_journeyers)
                )
    return instances


def save_grid_collection(instances, output_directory, overwrite=False):
    output_directory = Path(output_directory)
    destinations = [
        output_directory / f"{instance['name']}.json" for instance in instances
    ]
    existing = [destination for destination in destinations if destination.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {existing[0]}; use --overwrite to replace files"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    for instance, destination in zip(instances, destinations):
        save_instance(instance, destination)
    return destinations


def main():
    parser = argparse.ArgumentParser(
        description="Generate the reproducible 45-instance rectangular-grid collection"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("instances/grid_collection"),
        help="Output directory (default: instances/grid_collection)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and validate every instance without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace collection files that already exist",
    )
    args = parser.parse_args()

    instances = generate_grid_collection(seed=args.seed)
    if args.dry_run:
        print(f"Constructed and validated {len(instances)} instances")
        return

    destinations = save_grid_collection(
        instances,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"Saved {len(destinations)} instances in {args.output}")


if __name__ == "__main__":
    main()
