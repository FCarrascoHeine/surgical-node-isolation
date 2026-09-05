import math
import time

from gurobipy import GRB, Model, quicksum

from graph_algorithms import directed_min_cut, shortest_path
from instances import prepare_instance
from time_budget import BudgetExpired, TimeBudget
from utils import STATUS_NAMES, configure_model
from validation import evaluate_allocation

HEURISTIC_NAMES = ("ah", "ash")


def _prepared_data(instance):
    if isinstance(instance, dict) and "tau" in instance and "balance" in instance:
        return instance
    return prepare_instance(instance)


def _validate_options(max_iterations, time_limit, tolerance):
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if time_limit is not None and time_limit < 0:
        raise ValueError("time_limit must be nonnegative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")


def _ordered_edges(edges, selected):
    return tuple(edge for edge in edges if edge in selected)


def _x_values(edges, selected):
    return {edge: float(edge in selected) for edge in edges}


def _allocation_cost(data, selected):
    return float(sum(data["checkpoint_cost"][edge] for edge in selected))


def _journeyer_paths(data, inspected, *, _budget=None):
    if _budget is not None:
        _budget.check()
    lengths = {
        edge: data["tau"][edge]
        + (data["inspection_time"][edge] if edge in inspected else 0.0)
        for edge in data["edges"]
    }
    paths = {}
    total_distance = 0.0

    for journeyer in data["journeyers"]:
        if _budget is not None:
            _budget.check()
        distance, path = shortest_path(
            data["nodes"],
            data["edges"],
            lengths,
            data["journeyer_source"][journeyer],
            data["journeyer_target"][journeyer],
        )
        if not math.isfinite(distance):
            raise RuntimeError(f"Journeyer {journeyer} has no directed path")
        paths[journeyer] = tuple(path)
        total_distance += distance

    return paths, float(total_distance)


def _evaluate_candidate(data, selected, tolerance):
    evaluation = evaluate_allocation(
        data["instance"],
        _x_values(data["edges"], selected),
        tolerance=tolerance,
    )
    if not evaluation["valid"]:
        raise RuntimeError(
            "The heuristic produced an invalid allocation: {}".format(
                evaluation["errors"]
            )
        )
    return evaluation


def _candidate_record(
    data,
    iteration,
    selected,
    evaluation,
    path_support,
    path_objective,
    **extra,
):
    return {
        "iteration": iteration,
        "selected_edges": _ordered_edges(data["edges"], selected),
        "checkpoint_cost": _allocation_cost(data, selected),
        "surrogate_path_objective": float(path_objective),
        "objective_value": evaluation["objective_value"],
        "journeyer_distances": dict(evaluation["journeyer_distances"]),
        "path_support_edges": _ordered_edges(data["edges"], path_support),
        "path_support_size": len(path_support),
        **extra,
    }


def _better_candidate(record, best, tolerance):
    if best is None:
        return True
    return record["objective_value"] < best["objective_value"] - tolerance * max(
        1.0,
        abs(best["objective_value"]),
    )


def _heuristic_result(
    data,
    method,
    start,
    status_name,
    convergence_reason,
    history,
    best,
    terminal,
    *,
    return_best,
    solver_runtime=0.0,
    auxiliary_solves=0,
    minimum_cut_solves=0,
    num_variables=0,
    num_constraints=0,
    nodes_explored=0.0,
    simplex_iterations=0.0,
    subproblems_optimal=True,
):
    runtime = float(time.perf_counter() - start)
    return_best = return_best or status_name in ("TIME_LIMIT", "MEM_LIMIT")
    chosen = best if return_best else terminal
    edges = () if data is None else data["edges"]
    selected = set() if chosen is None else set(chosen["selected_edges"])
    objective = None if chosen is None else float(chosen["objective_value"])
    checkpoint_cost = None if chosen is None else float(chosen["checkpoint_cost"])

    result = {
        "method": method,
        "formulation": None,
        "relax": False,
        "status": None,
        "status_name": status_name,
        "has_solution": chosen is not None,
        "solution_type": "integer" if chosen is not None else "none",
        "objective_value": objective,
        "dual_bound": None,
        "gap": None,
        "runtime": runtime,
        "solver_runtime": float(solver_runtime),
        "num_variables": int(num_variables),
        "num_constraints": int(num_constraints),
        "num_linear_constraints": int(num_constraints),
        "num_quadratic_constraints": 0,
        "nodes_explored": float(nodes_explored),
        "simplex_iterations": float(simplex_iterations),
        "cuts": 0,
        "cut_iterations": 0,
        "master_solves": int(auxiliary_solves),
        "separation_time": 0.0,
        "separation_complete": status_name == "CONVERGED",
        "variables": {"x": _x_values(edges, selected)} if chosen else {},
        "selected_edges": _ordered_edges(edges, selected),
        "checkpoint_cost": checkpoint_cost,
        "heuristic_iterations": len(history),
        "convergence_reason": convergence_reason,
        "auxiliary_solves": int(auxiliary_solves),
        "minimum_cut_solves": int(minimum_cut_solves),
        "best_iteration": None if best is None else best["iteration"],
        "terminal_objective": (
            None if terminal is None else float(terminal["objective_value"])
        ),
        "terminal_selected_edges": (
            () if terminal is None else tuple(terminal["selected_edges"])
        ),
        "returned_best_candidate": bool(return_best),
        "subproblems_optimal": bool(subproblems_optimal),
        "iteration_history": history,
    }
    return result


def _solve_standard_subproblem(
    data,
    path_support,
    *,
    time_limit,
    output_flag,
    solver_seed,
    threads,
    env,
    _budget=None,
):
    budget = _budget if _budget is not None else TimeBudget(time_limit)
    budget.check()
    model = Model("SNI_standard_heuristic_subproblem", env=env)
    try:
        configure_model(
            model,
            output_flag=output_flag,
            solver_seed=solver_seed,
            threads=threads,
        )
        budget.check()
        selected = {}
        for edge in data["edges"]:
            budget.check()
            selected[edge] = model.addVar(
                vtype=GRB.BINARY,
                name="x[{},{}]".format(*edge),
            )
        source_side = {}
        for intruder in data["intruders"]:
            for node in data["nodes"]:
                budget.check()
                source_side[intruder, node] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y[{intruder},{node}]",
                )
        budget.check()
        model.setObjective(
            quicksum(
                (
                    data["inspection_time"][edge]
                    if edge in path_support
                    else 0.0
                )
                * selected[edge]
                for edge in data["edges"]
            ),
            GRB.MINIMIZE,
        )
        budget.check()
        model.addConstr(
            quicksum(
                data["checkpoint_cost"][edge] * selected[edge]
                for edge in data["edges"]
            )
            <= data["budget"],
            name="budget",
        )
        for intruder in data["intruders"]:
            budget.check()
            source = data["intruder_source"][intruder]
            target = data["intruder_target"][intruder]
            model.addConstr(
                source_side[intruder, source]
                - source_side[intruder, target]
                == 1,
                name=f"terminals[{intruder}]",
            )
            for tail, head in data["edges"]:
                budget.check()
                model.addConstr(
                    selected[tail, head]
                    >= source_side[intruder, tail]
                    - source_side[intruder, head],
                    name=f"cut[{intruder},{tail},{head}]",
                )

        model.Params.FeasibilityTol = 1e-8
        budget.check()
        model.update()
        budget.apply_to(model)
        model.optimize()
        result = {
            "status": int(model.Status),
            "status_name": STATUS_NAMES.get(
                model.Status, f"STATUS_{model.Status}"
            ),
            "selected": None,
            "objective_value": None,
            "runtime": float(model.Runtime),
            "num_variables": int(model.NumVars),
            "num_constraints": int(model.NumConstrs),
            "nodes_explored": float(model.NodeCount),
            "simplex_iterations": float(model.IterCount),
        }
        if model.SolCount > 0:
            result["selected"] = {
                edge for edge in data["edges"] if selected[edge].X >= 0.5
            }
            result["objective_value"] = float(model.ObjVal)
        return result
    finally:
        model.dispose()


def solve_standard_heuristic(
    instance,
    *,
    max_iterations=100,
    time_limit=None,
    tolerance=1e-6,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
    return_best=True,
    _budget=None,
):
    """Run A_H with one elapsed-time budget including input/model preparation."""
    budget = _budget if _budget is not None else TimeBudget(time_limit)
    _validate_options(max_iterations, time_limit, tolerance)
    data = None
    inspected_history = set()
    path_support = set()
    previous = None
    seen = set()
    history = []
    best = None
    terminal = None
    solver_runtime = 0.0
    auxiliary_solves = 0
    subproblems_optimal = True
    num_variables = 0
    num_constraints = 0
    nodes_explored = 0.0
    simplex_iterations = 0.0
    status_name = "ITERATION_LIMIT"
    convergence_reason = "iteration_limit"

    try:
        budget.check()
        data = _prepared_data(instance)
        for iteration in range(1, max_iterations + 1):
            budget.check()
            paths, path_objective = _journeyer_paths(
                data, inspected_history, _budget=budget
            )
            budget.check()
            for path in paths.values():
                path_support.update(path)

            subproblem = _solve_standard_subproblem(
                data,
                path_support,
                time_limit=budget.remaining(),
                output_flag=output_flag,
                solver_seed=solver_seed,
                threads=threads,
                env=env,
                _budget=budget,
            )
            auxiliary_solves += 1
            solver_runtime += subproblem["runtime"]
            num_variables = max(num_variables, subproblem["num_variables"])
            num_constraints = max(num_constraints, subproblem["num_constraints"])
            nodes_explored += subproblem["nodes_explored"]
            simplex_iterations += subproblem["simplex_iterations"]

            if subproblem["selected"] is None:
                status_name = subproblem["status_name"]
                convergence_reason = f"auxiliary_{status_name.lower()}"
                subproblems_optimal = False
                break

            # Finish evaluating an incumbent even when its solve used the budget.
            selected = subproblem["selected"]
            evaluation = _evaluate_candidate(data, selected, tolerance)
            terminal = _candidate_record(
                data,
                iteration,
                selected,
                evaluation,
                path_support,
                path_objective,
                auxiliary_objective=subproblem["objective_value"],
                auxiliary_status=subproblem["status_name"],
            )
            history.append(terminal)
            if _better_candidate(terminal, best, tolerance):
                best = terminal

            if subproblem["status_name"] != "OPTIMAL":
                status_name = subproblem["status_name"]
                convergence_reason = f"auxiliary_{status_name.lower()}"
                subproblems_optimal = False
                break

            key = frozenset(selected)
            if previous is not None and selected == previous:
                status_name = "CONVERGED"
                convergence_reason = "stable_allocation"
                break
            if key in seen:
                status_name = "CONVERGED"
                convergence_reason = "repeated_allocation"
                break

            # A completed convergence condition may stand after an atomic overrun;
            # starting another iteration (or reporting its cap) may not.
            budget.check()
            seen.add(key)
            previous = set(selected)
            inspected_history.update(selected)
    except BudgetExpired:
        status_name = "TIME_LIMIT"
        convergence_reason = "time_limit"

    return _heuristic_result(
        data,
        "ah",
        budget.start,
        status_name,
        convergence_reason,
        history,
        best,
        terminal,
        return_best=return_best,
        solver_runtime=solver_runtime,
        auxiliary_solves=auxiliary_solves,
        num_variables=num_variables,
        num_constraints=num_constraints,
        nodes_explored=nodes_explored,
        simplex_iterations=simplex_iterations,
        subproblems_optimal=subproblems_optimal,
    )


def solve_single_intruder_heuristic(
    instance,
    *,
    max_iterations=100,
    binary_search_tolerance=1e-4,
    time_limit=None,
    tolerance=1e-6,
    return_best=True,
    _budget=None,
):
    """Run A_SH with one elapsed-time budget including input preparation."""
    budget = _budget if _budget is not None else TimeBudget(time_limit)
    _validate_options(max_iterations, time_limit, tolerance)
    if not 0 < binary_search_tolerance < 1:
        raise ValueError("binary_search_tolerance must be between zero and one")
    data = None
    minimum_cut_solves = 0
    minimum_cost_cut = None
    inspected_history = set()
    path_support = set()
    previous = None
    seen = set()
    history = []
    records = {}
    best = None
    terminal = None
    pending = None
    status_name = "ITERATION_LIMIT"
    convergence_reason = "iteration_limit"

    def record_candidate(selected, iteration, path_objective, **extra):
        nonlocal best, terminal
        evaluation = _evaluate_candidate(data, selected, tolerance)
        record = _candidate_record(
            data, iteration, selected, evaluation, path_support,
            path_objective, **extra,
        )
        records[frozenset(selected)] = record
        terminal = record
        if iteration > 0:
            history.append(record)
        if _better_candidate(record, best, tolerance):
            best = record
        return record

    try:
        budget.check()
        data = _prepared_data(instance)
        if len(data["intruders"]) != 1:
            raise ValueError("The single-intruder heuristic requires exactly one intruder")
        intruder = data["intruders"][0]
        source = data["intruder_source"][intruder]
        target = data["intruder_target"][intruder]
        flow_tolerance = min(1e-9, tolerance * 0.01)

        budget.check()
        _, _, minimum_cost_edges = directed_min_cut(
            data["nodes"], data["edges"], data["checkpoint_cost"], source,
            target, tolerance=flow_tolerance,
        )
        minimum_cut_solves += 1
        initial_cut = set(minimum_cost_edges)
        if _allocation_cost(data, initial_cut) > data["budget"] + tolerance:
            # The completed minimum cut proves infeasibility, even if that
            # indivisible operation returned slightly after the deadline.
            return _heuristic_result(
                data, "ash", budget.start, "INFEASIBLE",
                "minimum_cost_cut_exceeds_budget", [], None, None,
                return_best=return_best, minimum_cut_solves=minimum_cut_solves,
            )
        minimum_cost_cut = initial_cut

        for iteration in range(1, max_iterations + 1):
            budget.check()
            paths, path_objective = _journeyer_paths(
                data, inspected_history, _budget=budget
            )
            budget.check()
            for path in paths.values():
                path_support.update(path)
            impact = {
                edge: data["inspection_time"][edge] if edge in path_support else 0.0
                for edge in data["edges"]
            }
            budget.check()

            if iteration == 1:
                _, _, impact_edges = directed_min_cut(
                    data["nodes"], data["edges"], impact, source, target,
                    tolerance=flow_tolerance,
                )
                minimum_cut_solves += 1
                impact_cut = set(impact_edges)
                if _allocation_cost(data, impact_cut) <= data["budget"] + tolerance:
                    record_candidate(
                        impact_cut, iteration, path_objective, alpha=0.0,
                        binary_search_iterations=0,
                    )
                    status_name = "CONVERGED"
                    convergence_reason = "impact_cut_feasible"
                    break

            lower = 0.0
            upper = 1.0
            pending = {
                "selected": set(minimum_cost_cut),
                "iteration": iteration,
                "path_objective": path_objective,
                "alpha": 1.0,
                "alpha_lower": lower,
                "alpha_upper": upper,
                "binary_search_iterations": 0,
            }
            # Binary-search trial cuts are auxiliary states. Only the current
            # feasible cut becomes an outer candidate on completion or timeout.
            while upper - lower >= binary_search_tolerance:
                budget.check()
                alpha = (lower + upper) / 2.0
                capacities = {
                    edge: alpha * data["checkpoint_cost"][edge]
                    + (1.0 - alpha) * impact[edge]
                    for edge in data["edges"]
                }
                budget.check()
                _, _, cut_edges = directed_min_cut(
                    data["nodes"], data["edges"], capacities, source, target,
                    tolerance=flow_tolerance,
                )
                minimum_cut_solves += 1
                pending["binary_search_iterations"] += 1
                candidate = set(cut_edges)
                if _allocation_cost(data, candidate) <= data["budget"] + tolerance:
                    upper = alpha
                    pending["selected"] = candidate
                    pending["alpha"] = alpha
                else:
                    lower = alpha
                pending["alpha_lower"] = lower
                pending["alpha_upper"] = upper

            selected = pending["selected"]
            record_candidate(**pending)
            pending = None
            key = frozenset(selected)
            if previous is not None and selected == previous:
                status_name = "CONVERGED"
                convergence_reason = "stable_allocation"
                break
            if key in seen:
                status_name = "CONVERGED"
                convergence_reason = "repeated_allocation"
                break
            budget.check()
            seen.add(key)
            previous = set(selected)
            inspected_history.update(selected)
    except BudgetExpired:
        status_name = "TIME_LIMIT"
        convergence_reason = "time_limit"
        # Evaluation finishes already obtained feasible candidates. It does not
        # start another cut or path-support search after the deadline.
        if pending is not None:
            record_candidate(**pending)
        if minimum_cost_cut is not None:
            fallback = records.get(frozenset(minimum_cost_cut))
            if fallback is None:
                prior_terminal = terminal
                fallback = record_candidate(
                    minimum_cost_cut, 0, 0.0, initial_feasible_fallback=True,
                )
                if prior_terminal is not None:
                    terminal = prior_terminal
            if _better_candidate(fallback, best, tolerance):
                best = fallback

    return _heuristic_result(
        data, "ash", budget.start, status_name, convergence_reason, history,
        best, terminal, return_best=return_best,
        minimum_cut_solves=minimum_cut_solves,
    )


def solve_heuristic(instance, method="auto", **kwargs):
    """Dispatch to A_H or A_SH without resetting the method's time budget."""
    budget = kwargs.pop("_budget", None)
    if budget is None:
        budget = TimeBudget(kwargs.get("time_limit"))
    method = method.lower()
    if method == "auto":
        # Resolve the method even for an expired budget; this atomic preparation
        # is charged to the same budget passed to the selected implementation.
        instance = _prepared_data(instance)
        method = "ash" if len(instance["intruders"]) == 1 else "ah"
    if method == "ah":
        return solve_standard_heuristic(instance, _budget=budget, **kwargs)
    if method == "ash":
        return solve_single_intruder_heuristic(instance, _budget=budget, **kwargs)
    raise ValueError("method must be auto, ah, or ash")
