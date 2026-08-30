"""Create weighted-cost sister instances for the SNI problem.

For every ``*dir.json`` file directly inside the input directory, this script
draws an integer checkpoint cost for every edge and solves an intruder-separation
model

    min  sum_e c_e x_e

Large models use dynamically separated intruder path cuts; smaller models use
the disaggregated x/y formulation.  The value of a best feasible solution
becomes the new instance budget.  If the solve reaches the time limit, the
incumbent is used; consequently the saved budget is feasible, although it may
not be the true minimum.
"""

import argparse
import copy
import random
from pathlib import Path

from gurobipy import GRB, Model, quicksum

from instances import load_instance, save_instance
from utils import load_gurobi_env


DEFAULT_COST_MIN = 10
DEFAULT_COST_MAX = 50
DEFAULT_TIME_LIMIT = 10 * 60
DEFAULT_CUT_THRESHOLD = 100_000
BUDGET_MODELS = ("auto", "disaggregated", "intruder-cuts")


def _complex_stem(stem):
    """Replace a terminal ``dir`` marker with ``c``."""
    if not stem.endswith("dir"):
        raise ValueError(f"The source instance name must end in 'dir': {stem}")
    return f"{stem[:-3]}c"


def complex_instance_path(source_path):
    """Return the requested sister path (for example, ``1dir.json`` -> ``1c.json``)."""
    source_path = Path(source_path)
    return source_path.with_name(f"{_complex_stem(source_path.stem)}.json")


def assign_checkpoint_costs(
    instance,
    rng,
    cost_min=DEFAULT_COST_MIN,
    cost_max=DEFAULT_COST_MAX,
):
    """Return a copy of *instance* with independently drawn integer edge costs."""
    if isinstance(cost_min, bool) or not isinstance(cost_min, int):
        raise ValueError("cost_min must be an integer")
    if isinstance(cost_max, bool) or not isinstance(cost_max, int):
        raise ValueError("cost_max must be an integer")
    if cost_min < 0 or cost_min > cost_max:
        raise ValueError(
            "cost_min and cost_max must satisfy 0 <= cost_min <= cost_max"
        )

    weighted_instance = copy.deepcopy(instance)
    for edge in weighted_instance["edges"]:
        edge["checkpoint_cost"] = rng.randint(cost_min, cost_max)
    return weighted_instance


def _directed_path_exists(nodes, edges, source, target, blocked_edges):
    """Return whether an unblocked directed source-target path exists."""
    outgoing = {node: [] for node in nodes}
    for tail, head in edges:
        if (tail, head) not in blocked_edges:
            outgoing[tail].append(head)

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


def _reachable_nodes(nodes, edges, source, blocked_edges):
    """Return nodes reachable from *source* after removing blocked edges."""
    outgoing = {node: [] for node in nodes}
    for edge in edges:
        if edge not in blocked_edges:
            outgoing[edge[0]].append(edge[1])

    reachable = {source}
    pending = [source]
    while pending:
        node = pending.pop()
        for successor in outgoing[node]:
            if successor not in reachable:
                reachable.add(successor)
                pending.append(successor)
    return reachable


def _initial_checkpoint_set(edges, intruders, costs):
    """Build a low-cost feasible checkpoint certificate for the MIP start."""
    sources = {intruder["source"] for intruder in intruders}
    targets = {intruder["target"] for intruder in intruders}
    outgoing_sources = {edge for edge in edges if edge[0] in sources}
    incoming_targets = {edge for edge in edges if edge[1] in targets}

    candidates = (outgoing_sources, incoming_targets)
    return min(candidates, key=lambda selected: sum(costs[e] for e in selected))


def _find_open_path(outgoing, source, target, x_values):
    """Find an s-t path containing only edges whose incumbent x value is zero."""
    previous = {source: None}
    pending = [source]
    position = 0

    while position < len(pending) and target not in previous:
        node = pending[position]
        position += 1
        for edge in outgoing[node]:
            successor = edge[1]
            if x_values[edge] >= 0.5 or successor in previous:
                continue
            previous[successor] = edge
            pending.append(successor)

    if target not in previous:
        return []

    path = []
    node = target
    while node != source:
        edge = previous[node]
        path.append(edge)
        node = edge[0]
    path.reverse()
    return path


def _resolve_budget_model(instance, budget_model, cut_threshold):
    if budget_model not in BUDGET_MODELS:
        raise ValueError(f"budget_model must be one of {', '.join(BUDGET_MODELS)}")
    if isinstance(cut_threshold, bool) or not isinstance(cut_threshold, int):
        raise ValueError("cut_threshold must be an integer")
    if cut_threshold < 0:
        raise ValueError("cut_threshold must be nonnegative")
    if budget_model != "auto":
        return budget_model

    problem_size = len(instance["intruders"]) * len(instance["edges"])
    if problem_size >= cut_threshold:
        return "intruder-cuts"
    return "disaggregated"


def build_minimum_checkpoint_model(
    instance,
    *,
    budget_model="auto",
    cut_threshold=DEFAULT_CUT_THRESHOLD,
    time_limit=DEFAULT_TIME_LIMIT,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    """Build the selected minimum-cost intruder-separation model."""
    nodes = list(instance["nodes"])
    edges = [(edge["tail"], edge["head"]) for edge in instance["edges"]]
    intruders = list(instance["intruders"])
    costs = {
        edge: int(edge_data["checkpoint_cost"])
        for edge, edge_data in zip(edges, instance["edges"])
    }
    selected_model = _resolve_budget_model(instance, budget_model, cut_threshold)

    model = Model(f"minimum_checkpoint_budget_{selected_model}", env=env)
    model.Params.OutputFlag = output_flag
    model.Params.TimeLimit = time_limit
    model.Params.Seed = solver_seed
    model.Params.Threads = threads
    if selected_model == "intruder-cuts":
        model.Params.LazyConstraints = 1

    x = {
        edge: model.addVar(vtype=GRB.BINARY, name="x[{},{}]".format(*edge))
        for edge in edges
    }

    model.setObjective(
        quicksum(costs[edge] * x[edge] for edge in edges),
        GRB.MINIMIZE,
    )

    initial_checkpoints = _initial_checkpoint_set(edges, intruders, costs)
    for edge, variable in x.items():
        variable.Start = float(edge in initial_checkpoints)

    if selected_model == "disaggregated":
        y = {
            (intruder["id"], node): model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"y[{intruder['id']},{node}]",
            )
            for intruder in intruders
            for node in nodes
        }
        for intruder in intruders:
            intruder_id = intruder["id"]
            source = intruder["source"]
            target = intruder["target"]
            model.addConstr(
                y[intruder_id, source] == 0,
                name=f"source[{intruder_id}]",
            )
            for tail, head in edges:
                model.addConstr(
                    y[intruder_id, head] - y[intruder_id, tail]
                    <= x[tail, head],
                    name=f"arc[{intruder_id},{tail},{head}]",
                )
            model.addConstr(
                y[intruder_id, target] >= 1,
                name=f"target[{intruder_id}]",
            )

            reachable = _reachable_nodes(
                nodes,
                edges,
                source,
                initial_checkpoints,
            )
            for node in nodes:
                y[intruder_id, node].Start = float(node not in reachable)

    model.update()
    outgoing = {node: [] for node in nodes}
    for edge in edges:
        outgoing[edge[0]].append(edge)

    model._x = x
    model._nodes = nodes
    model._edges = edges
    model._outgoing = outgoing
    model._intruders = intruders
    model._budget_model = selected_model
    model._initial_checkpoints = initial_checkpoints
    model._cut_tolerance = 1e-6
    model._cut_keys = set()
    model._intruder_cuts = 0
    model._lazy_additions = 0
    model._callback_calls = 0
    model._callback_error = None
    return model, x, costs


def _minimum_checkpoint_callback(model, where):
    """Separate violated intruder path cuts at integer incumbents."""
    if where != GRB.Callback.MIPSOL:
        return

    model._callback_calls += 1
    try:
        x_values = {
            edge: float(model.cbGetSolution(model._x[edge]))
            for edge in model._edges
        }

        for intruder in model._intruders:
            path = _find_open_path(
                model._outgoing,
                intruder["source"],
                intruder["target"],
                x_values,
            )
            path_value = sum(x_values[edge] for edge in path)
            if not path or path_value >= 1 - model._cut_tolerance:
                continue

            model.cbLazy(quicksum(model._x[edge] for edge in path) >= 1)
            model._lazy_additions += 1
            cut_key = (intruder["id"], tuple(path))
            if cut_key not in model._cut_keys:
                model._cut_keys.add(cut_key)
                model._intruder_cuts += 1
    except Exception as error:  # noqa: BLE001
        model._callback_error = error
        model.terminate()


def minimum_feasible_budget(
    instance,
    *,
    budget_model="auto",
    cut_threshold=DEFAULT_CUT_THRESHOLD,
    time_limit=DEFAULT_TIME_LIMIT,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    """Solve for a minimum budget and return budget/status information."""
    model, x, costs = build_minimum_checkpoint_model(
        instance,
        budget_model=budget_model,
        cut_threshold=cut_threshold,
        time_limit=time_limit,
        output_flag=output_flag,
        solver_seed=solver_seed,
        threads=threads,
        env=env,
    )
    try:
        if model._budget_model == "intruder-cuts":
            model.optimize(_minimum_checkpoint_callback)
        else:
            model.optimize()
        if model._callback_error is not None:
            raise RuntimeError(
                "Error while separating intruder path cuts"
            ) from model._callback_error
        if model.SolCount == 0:
            raise RuntimeError(
                "The minimum-budget model ended without a feasible solution "
                f"(Gurobi status {model.Status})"
            )
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(
                "The minimum-budget model did not finish normally "
                f"(Gurobi status {model.Status})"
            )

        selected_edges = [
            edge for edge, variable in x.items() if variable.X >= 0.5
        ]
        # Compute the integer value directly from the incumbent instead of
        # rounding ObjVal, avoiding any numerical-tolerance ambiguity.
        budget = sum(costs[edge] for edge in selected_edges)
        selected_edge_set = set(selected_edges)
        nodes = instance["nodes"]
        edges = list(costs)
        for intruder in instance["intruders"]:
            if _directed_path_exists(
                nodes,
                edges,
                intruder["source"],
                intruder["target"],
                selected_edge_set,
            ):
                raise RuntimeError(
                    "Gurobi returned an incumbent that does not capture "
                    f"intruder {intruder['id']}"
                )
        return {
            "budget": budget,
            "optimal": model.Status == GRB.OPTIMAL,
            "status": int(model.Status),
            "budget_model": model._budget_model,
            "objective_bound": float(model.ObjBound),
            "selected_edges": selected_edges,
            "intruder_cuts": model._intruder_cuts,
            "lazy_additions": model._lazy_additions,
            "callback_calls": model._callback_calls,
        }
    finally:
        model.dispose()


def create_complex_instance(
    source_path,
    *,
    rng,
    cost_min=DEFAULT_COST_MIN,
    cost_max=DEFAULT_COST_MAX,
    budget_model="auto",
    cut_threshold=DEFAULT_CUT_THRESHOLD,
    time_limit=DEFAULT_TIME_LIMIT,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    """Create one in-memory weighted sister instance and its solve summary."""
    source_path = Path(source_path)
    instance = load_instance(source_path)
    weighted_instance = assign_checkpoint_costs(instance, rng, cost_min, cost_max)
    result = minimum_feasible_budget(
        weighted_instance,
        budget_model=budget_model,
        cut_threshold=cut_threshold,
        time_limit=time_limit,
        output_flag=output_flag,
        solver_seed=solver_seed,
        threads=threads,
        env=env,
    )

    weighted_instance["name"] = _complex_stem(source_path.stem)
    weighted_instance["budget"] = result["budget"]

    # Certificates and known objective values from the unit-cost instance need
    # not remain valid under the newly computed minimum weighted budget.
    for field in (
        "known_feasible_checkpoints",
        "known_optimal_checkpoints",
        "known_optimum",
    ):
        weighted_instance.pop(field, None)
    weighted_instance["known_feasible_checkpoints"] = [
        list(edge) for edge in result["selected_edges"]
    ]

    return weighted_instance, result


def create_complex_instances(
    instances_directory=Path("instances"),
    *,
    pattern="*dir.json",
    seed=0,
    cost_min=DEFAULT_COST_MIN,
    cost_max=DEFAULT_COST_MAX,
    budget_model="auto",
    cut_threshold=DEFAULT_CUT_THRESHOLD,
    time_limit=DEFAULT_TIME_LIMIT,
    output_flag=0,
    solver_seed=0,
    threads=1,
    overwrite=False,
    env=None,
):
    """Generate and save every matching sister instance in a directory."""
    instances_directory = Path(instances_directory)
    sources = sorted(instances_directory.glob(pattern))
    if not sources:
        raise FileNotFoundError(
            f"No instances matching {pattern!r} found in {instances_directory}"
        )
    invalid_sources = [source for source in sources if not source.stem.endswith("dir")]
    if invalid_sources:
        raise ValueError(
            f"The selected source name must end in 'dir': {invalid_sources[0].name}"
        )

    destinations = [complex_instance_path(source) for source in sources]
    existing = [destination for destination in destinations if destination.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {existing[0]}; use --overwrite to replace files"
        )

    rng = random.Random(seed)
    summaries = []
    for source, destination in zip(sources, destinations):
        print(f"Processing {source.name} ...", flush=True)
        weighted_instance, result = create_complex_instance(
            source,
            rng=rng,
            cost_min=cost_min,
            cost_max=cost_max,
            budget_model=budget_model,
            cut_threshold=cut_threshold,
            time_limit=time_limit,
            output_flag=output_flag,
            solver_seed=solver_seed,
            threads=threads,
            env=env,
        )
        save_instance(weighted_instance, destination)
        if result["optimal"]:
            qualification = "minimum proven"
        else:
            qualification = (
                "feasible incumbent after time limit; minimum not proven, "
                f"lower bound {result['objective_bound']:.6g}"
            )
        print(
            f"Saved {destination.name}: budget={result['budget']} "
            f"({qualification}); model={result['budget_model']}; "
            f"intruder cuts={result['intruder_cuts']}",
            flush=True,
        )
        summaries.append({"source": source, "destination": destination, **result})

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create random weighted-cost sister instances and minimum "
            "feasible budgets"
        )
    )
    parser.add_argument(
        "--instances-directory",
        type=Path,
        default=Path("instances"),
        help="Directory containing *dir.json files (default: instances)",
    )
    parser.add_argument(
        "--pattern",
        default="*dir.json",
        help="Filename pattern inside the directory (default: *dir.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random checkpoint-cost seed",
    )
    parser.add_argument("--cost-min", type=int, default=DEFAULT_COST_MIN)
    parser.add_argument("--cost-max", type=int, default=DEFAULT_COST_MAX)
    parser.add_argument(
        "--budget-model",
        choices=BUDGET_MODELS,
        default="auto",
        help=(
            "Model used to compute the budget (default: auto; intruder cuts "
            "when |I||E| reaches --cut-threshold)"
        ),
    )
    parser.add_argument(
        "--cut-threshold",
        type=int,
        default=DEFAULT_CUT_THRESHOLD,
        help="Auto-mode threshold on |I||E| (default: 100000)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="Seconds allowed for each minimum-budget model (default: 600)",
    )
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", action="store_true", help="Display Gurobi logs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    if args.threads < 0:
        parser.error("--threads must be nonnegative")
    if args.cut_threshold < 0:
        parser.error("--cut-threshold must be nonnegative")

    with load_gurobi_env() as env:
        summaries = create_complex_instances(
            args.instances_directory,
            pattern=args.pattern,
            seed=args.seed,
            cost_min=args.cost_min,
            cost_max=args.cost_max,
            budget_model=args.budget_model,
            cut_threshold=args.cut_threshold,
            time_limit=args.time_limit,
            output_flag=int(args.output),
            solver_seed=args.solver_seed,
            threads=args.threads,
            overwrite=args.overwrite,
            env=env,
        )

    nonoptimal = sum(not summary["optimal"] for summary in summaries)
    print(
        f"Created {len(summaries)} complex instances "
        f"({nonoptimal} not proven optimal)"
    )


if __name__ == "__main__":
    main()
