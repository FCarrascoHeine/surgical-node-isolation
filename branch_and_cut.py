import heapq
import itertools
import math
import time

from gurobipy import GRB, Model, quicksum

from formulations import build_formulation_4
from instances import prepare_instance
from utils import collect_model_result, variable_values


def _prepared_data(instance):
    if isinstance(instance, dict) and "tau" in instance and "balance" in instance:
        return instance

    return prepare_instance(instance)


def shortest_path(nodes, edges, lengths, source, target):
    outgoing = {v: [] for v in nodes}

    for edge in edges:
        outgoing[edge[0]].append(edge)

    distance = {v: math.inf for v in nodes}
    previous = {}
    counter = itertools.count()
    distance[source] = 0.0
    pending = [(0.0, next(counter), source)]

    while pending:
        current_distance, _, v = heapq.heappop(pending)

        if current_distance > distance[v] + 1e-12:
            continue
        if v == target:
            break

        for edge in outgoing[v]:
            v_prime = edge[1]
            new_distance = current_distance + lengths[edge]

            if new_distance < distance[v_prime] - 1e-12:
                distance[v_prime] = new_distance
                previous[v_prime] = (v, edge)
                heapq.heappush(
                    pending,
                    (new_distance, next(counter), v_prime),
                )

    if not math.isfinite(distance[target]):
        return math.inf, []

    path = []
    v = target
    while v != source:
        v, edge = previous[v]
        path.append(edge)
    path.reverse()

    return distance[target], path


def directed_min_cut(nodes, edges, capacities, source, target, tolerance=1e-9):
    adjacent = {v: set() for v in nodes}
    residual = {v: {} for v in nodes}

    for v, v_prime in edges:
        capacity = max(0.0, float(capacities[v, v_prime]))
        adjacent[v].add(v_prime)
        adjacent[v_prime].add(v)
        residual[v][v_prime] = residual[v].get(v_prime, 0.0) + capacity
        residual[v_prime].setdefault(v, 0.0)

    flow_value = 0.0

    while True:
        previous = {source: None}
        pending = [source]
        position = 0

        while position < len(pending) and target not in previous:
            v = pending[position]
            position += 1

            for v_prime in sorted(adjacent[v], key=str):
                if v_prime in previous:
                    continue
                if residual[v].get(v_prime, 0.0) <= tolerance:
                    continue

                previous[v_prime] = v
                pending.append(v_prime)

        if target not in previous:
            break

        increment = math.inf
        v = target
        while v != source:
            v_previous = previous[v]
            increment = min(increment, residual[v_previous][v])
            v = v_previous

        v = target
        while v != source:
            v_previous = previous[v]
            residual[v_previous][v] -= increment
            residual[v][v_previous] = residual[v].get(v_previous, 0.0) + increment
            v = v_previous

        flow_value += increment

    reachable = {source}
    pending = [source]

    while pending:
        v = pending.pop()

        for v_prime in sorted(adjacent[v], key=str):
            if v_prime in reachable:
                continue
            if residual[v].get(v_prime, 0.0) <= tolerance:
                continue

            reachable.add(v_prime)
            pending.append(v_prime)

    cut_edges = [
        edge
        for edge in edges
        if edge[0] in reachable and edge[1] not in reachable
    ]
    cut_value = sum(capacities[e] for e in cut_edges)

    return float(cut_value), reachable, cut_edges


def solve_journeyer_dual(
    instance,
    journeyer,
    capacities,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    data = _prepared_data(instance)
    V = data["nodes"]
    E = data["edges"]
    tau = data["tau"]
    s_j = data["journeyer_source"]
    t_j = data["journeyer_target"]

    dual = Model(f"journeyer_dual_{journeyer}", env=env)

    #Variables of DP^j
    rho = {v: dual.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"rho[{v}]") for v in V}
    delta = {e: dual.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0, name=f"delta[{e[0]},{e[1]}]") for e in E}
    rho[s_j[journeyer]].LB = 0
    rho[s_j[journeyer]].UB = 0
    dual.update()

    #Objective function of DP^j
    objective = rho[t_j[journeyer]] - rho[s_j[journeyer]] + quicksum(delta[e] * capacities[e] for e in E)
    dual.setObjective(objective, GRB.MAXIMIZE)

    # Constraints (33)-(35)
    for e in E:
        v, v_prime = e
        dual.addConstr(rho[v_prime] - rho[v] + delta[e] <= tau[e], name=f"dual_arc_33[{v},{v_prime}]")

    dual.Params.OutputFlag = output_flag
    if threads is not None:
        dual.Params.Threads = threads
    if solver_seed is not None:
        dual.Params.Seed = solver_seed
    dual.Params.Method = 1
    dual.Params.DualReductions = 0
    dual.Params.FeasibilityTol = 1e-9
    dual.optimize()

    if dual.Status != GRB.OPTIMAL:
        status = dual.Status
        dual.dispose()
        raise RuntimeError(
            f"Journeyer dual {journeyer} ended with status {status}"
        )

    rho_value = {
        v: float(rho[v].X - rho[s_j[journeyer]].X)
        for v in V
    }
    delta_value = {
        e: min(
            0.0,
            tau[e] - rho_value[e[1]] + rho_value[e[0]],
        )
        for e in E
    }
    max_dual_violation = max(
        [
            abs(rho_value[s_j[journeyer]]),
            max(delta_value.values(), default=0.0),
        ]
        + [
            rho_value[e[1]] - rho_value[e[0]] + delta_value[e] - tau[e]
            for e in E
        ]
    )

    if max_dual_violation > 1e-7:
        dual.dispose()
        raise RuntimeError(
            f"Numerically infeasible dual solution for journeyer {journeyer}"
        )

    result = {
        "objective_value": float(
            rho_value[t_j[journeyer]]
            - rho_value[s_j[journeyer]]
            + sum(delta_value[e] * capacities[e] for e in E)
        ),
        "rho": rho_value,
        "delta": delta_value,
        "max_dual_violation": float(max(0.0, max_dual_violation)),
    }
    dual.dispose()

    return result


def separate_solution(
    instance,
    x_values,
    alpha_values,
    phi_values,
    tolerance=1e-6,
    solver_seed=0,
    threads=1,
    env=None,
):
    data = _prepared_data(instance)
    V = data["nodes"]
    E = data["edges"]
    I = data["intruders"]
    J = data["journeyers"]
    s_i = data["intruder_source"]
    t_i = data["intruder_target"]
    s_j = data["journeyer_source"]
    t_j = data["journeyer_target"]

    x_raw = {
        e: float(x_values[e])
        for e in E
    }
    alpha_raw = {
        (j, e): float(alpha_values[j, e])
        for j in J
        for e in E
    }
    phi_raw = {
        j: float(phi_values[j])
        for j in J
    }
    x_algorithm = {
        e: min(1.0, max(0.0, x_raw[e]))
        for e in E
    }
    alpha_algorithm = {
        (j, e): min(1.0, max(0.0, alpha_raw[j, e]))
        for j in J
        for e in E
    }
    cuts = []

    # Constraint (24): intruder path cuts
    for i in I:
        _, path = shortest_path(
            V,
            E,
            x_algorithm,
            s_i[i],
            t_i[i],
        )
        path_value = sum(x_raw[e] for e in path)

        if path and path_value < 1 - tolerance:
            cuts.append(
                {
                    "family": "intruder",
                    "agent": i,
                    "path": tuple(path),
                    "value": float(path_value),
                    "violation": float(1 - path_value),
                }
            )

    # Constraints (25)-(26): journeyer feasibility and optimality cuts
    for j in J:
        capacities_raw = {
            e: alpha_raw[j, e] + 1 - x_raw[e]
            for e in E
        }
        capacities_algorithm = {
            e: alpha_algorithm[j, e] + 1 - x_algorithm[e]
            for e in E
        }
        cut_value_algorithm, cut_set, cut_edges = directed_min_cut(
            V,
            E,
            capacities_algorithm,
            s_j[j],
            t_j[j],
            tolerance=tolerance * 0.1,
        )
        cut_value = sum(capacities_raw[e] for e in cut_edges)

        if cut_value < 1 - tolerance:
            cuts.append(
                {
                    "family": "feasibility",
                    "agent": j,
                    "cut_set": tuple(sorted(cut_set, key=str)),
                    "cut_edges": tuple(cut_edges),
                    "value": float(cut_value),
                    "violation": float(1 - cut_value),
                }
            )
            continue
        if cut_value_algorithm < 1 - 1e-9:
            # The violation is below the requested separation tolerance, but
            # SP^j is still mathematically infeasible, so DP^j is not solved.
            continue

        dual_result = solve_journeyer_dual(
            data,
            j,
            capacities_algorithm,
            solver_seed=solver_seed,
            threads=threads,
            env=env,
        )
        rho_difference = (
            dual_result["rho"][t_j[j]]
            - dual_result["rho"][s_j[j]]
        )
        right_hand_side = rho_difference + sum(
            dual_result["delta"][e] * capacities_raw[e]
            for e in E
        )
        optimality_tolerance = tolerance * max(
            1.0,
            abs(right_hand_side),
            abs(phi_raw[j]),
        )

        if right_hand_side > phi_raw[j] + optimality_tolerance:
            cuts.append(
                {
                    "family": "optimality",
                    "agent": j,
                    "rho_difference": float(rho_difference),
                    "rho": dual_result["rho"],
                    "delta": dual_result["delta"],
                    "max_dual_violation": dual_result["max_dual_violation"],
                    "value": float(right_hand_side),
                    "violation": float(right_hand_side - phi_raw[j]),
                }
            )

    return cuts


def _cut_key(cut, edges):
    if cut["family"] == "intruder":
        return ("intruder", cut["agent"], cut["path"])

    if cut["family"] == "feasibility":
        return ("feasibility", cut["agent"], cut["cut_edges"])

    coefficients = tuple(float(cut["delta"][e]) for e in edges)
    return (
        "optimality",
        cut["agent"],
        float(cut["rho_difference"]),
        coefficients,
    )


def _lazy_constraint(mod, cut):
    if cut["family"] == "intruder":
        mod.cbLazy(quicksum(mod._x[e] for e in cut["path"]) >= 1)
        return

    j = cut["agent"]
    if cut["family"] == "feasibility":
        mod.cbLazy(quicksum(mod._alpha[j, e] + 1 - mod._x[e] for e in cut["cut_edges"]) >= 1)
        return

    right_hand_side = cut["rho_difference"] + quicksum(cut["delta"][e] * (mod._alpha[j, e] + 1 - mod._x[e]) for e in mod._data["edges"])
    mod.cbLazy(mod._phi[j] >= right_hand_side)


def _model_constraint(mod, variables, cut, cut_number):
    x = variables["x"]
    alpha = variables["alpha"]
    phi = variables["phi"]

    if cut["family"] == "intruder":
        mod.addConstr(quicksum(x[e] for e in cut["path"]) >= 1, name=f"intruder_cut_24[{cut_number}]")
        return

    j = cut["agent"]
    if cut["family"] == "feasibility":
        mod.addConstr(quicksum(alpha[j, e] + 1 - x[e] for e in cut["cut_edges"]) >= 1, name=f"feasibility_cut_25[{cut_number}]")
        return

    right_hand_side = cut["rho_difference"] + quicksum(cut["delta"][e] * (alpha[j, e] + 1 - x[e]) for e in mod._data["edges"])
    mod.addConstr(phi[j] >= right_hand_side, name=f"optimality_cut_26[{cut_number}]")


def _branch_and_cut_callback(mod, where):
    if where != GRB.Callback.MIPSOL:
        return

    separation_start = time.perf_counter()
    mod._callback_calls += 1

    try:
        x_values = {
            e: mod.cbGetSolution(mod._x[e])
            for e in mod._data["edges"]
        }
        alpha_values = {
            (j, e): mod.cbGetSolution(mod._alpha[j, e])
            for j in mod._data["journeyers"]
            for e in mod._data["edges"]
        }
        phi_values = {
            j: mod.cbGetSolution(mod._phi[j])
            for j in mod._data["journeyers"]
        }
        cuts = separate_solution(
            mod._data,
            x_values,
            alpha_values,
            phi_values,
            tolerance=mod._tolerance,
            solver_seed=mod._solver_seed,
            threads=mod._threads,
            env=mod._solve_env,
        )

        for cut in cuts:
            key = _cut_key(cut, mod._data["edges"])
            # Gurobi may present the same incumbent more than once. A violated
            # lazy constraint must be submitted every time it is encountered.
            _lazy_constraint(mod, cut)
            mod._lazy_additions += 1

            if key not in mod._cut_keys:
                mod._cut_keys.add(key)
                mod._cut_count += 1
                mod._cuts_by_family[cut["family"]] += 1
    except Exception as error: # noqa: BLE001
        mod._callback_error = error
        mod.terminate()
    finally:
        mod._separation_time += time.perf_counter() - separation_start


def _solve_integer(
    instance,
    time_limit=None,
    output_flag=0,
    tolerance=1e-6,
    solver_seed=0,
    threads=1,
    env=None,
):
    mod, variables = build_formulation_4(
        instance,
        relax=False,
        time_limit=time_limit,
        output_flag=output_flag,
        solver_seed=solver_seed,
        threads=threads,
        env=env,
    )
    base_constraints = mod.NumConstrs

    mod._x = variables["x"]
    mod._alpha = variables["alpha"]
    mod._phi = variables["phi"]
    mod._tolerance = tolerance
    mod._solver_seed = solver_seed
    mod._threads = threads
    mod._cut_keys = set()
    mod._cut_count = 0
    mod._lazy_additions = 0
    mod._cuts_by_family = {
        "intruder": 0,
        "feasibility": 0,
        "optimality": 0,
    }
    mod._callback_calls = 0
    mod._separation_time = 0.0
    mod._callback_error = None

    start = time.perf_counter()
    mod.optimize(_branch_and_cut_callback)

    if mod._callback_error is not None:
        raise RuntimeError("Error in branch-and-cut callback") from mod._callback_error

    solution_values = {}
    remaining_cuts = []
    if mod.SolCount > 0:
        solution_values = variable_values(variables)
        verification_start = time.perf_counter()
        remaining_cuts = separate_solution(
            mod._data,
            solution_values["x"],
            solution_values["alpha"],
            solution_values["phi"],
            tolerance=tolerance,
            solver_seed=solver_seed,
            threads=threads,
            env=env,
        )
        mod._separation_time += time.perf_counter() - verification_start

    wall_time = time.perf_counter() - start
    result = collect_model_result(mod, formulation=4, relax=False)
    result["runtime"] = wall_time
    result["solver_runtime"] = float(mod.Runtime)
    result["num_constraints"] = int(base_constraints + mod._cut_count)
    result["num_linear_constraints"] = result["num_constraints"]
    result["cuts"] = mod._cut_count
    result["lazy_additions"] = mod._lazy_additions
    result["cuts_by_family"] = dict(mod._cuts_by_family)
    result["cut_iterations"] = mod._callback_calls
    result["callback_calls"] = mod._callback_calls
    result["separation_time"] = mod._separation_time
    result["remaining_violated_cuts"] = len(remaining_cuts)
    result["separation_complete"] = (
        mod.Status == GRB.OPTIMAL
        and not remaining_cuts
    )

    if solution_values:
        result["variables"] = solution_values

    mod.dispose()
    return result


def _solve_relaxation(
    instance,
    time_limit=None,
    output_flag=0,
    tolerance=1e-6,
    max_iterations=100,
    max_cuts=None,
    solver_seed=0,
    threads=1,
    env=None,
):
    mod, variables = build_formulation_4(
        instance,
        relax=True,
        time_limit=time_limit,
        output_flag=output_flag,
        solver_seed=solver_seed,
        threads=threads,
        env=env,
    )
    base_constraints = mod.NumConstrs
    data = mod._data
    cut_keys = set()
    cuts_by_family = {
        "intruder": 0,
        "feasibility": 0,
        "optimality": 0,
    }
    cut_count = 0
    cut_iterations = 0
    master_solves = 0
    solver_runtime = 0.0
    separation_time = 0.0
    separation_complete = False
    limit_reached = False
    objective_history = []
    cut_history = []
    start = time.perf_counter()

    while True:
        if time_limit is not None:
            remaining_time = time_limit - (time.perf_counter() - start)
            mod.Params.TimeLimit = max(0.0, remaining_time)

        mod.optimize()
        master_solves += 1
        solver_runtime += mod.Runtime

        if mod.Status != GRB.OPTIMAL or mod.SolCount == 0:
            limit_reached = mod.Status != GRB.OPTIMAL
            break

        objective_history.append(float(mod.ObjVal))

        x_values = {
            e: variables["x"][e].X
            for e in data["edges"]
        }
        alpha_values = {
            (j, e): variables["alpha"][j, e].X
            for j in data["journeyers"]
            for e in data["edges"]
        }
        phi_values = {
            j: variables["phi"][j].X
            for j in data["journeyers"]
        }

        separation_start = time.perf_counter()
        cuts = separate_solution(
            data,
            x_values,
            alpha_values,
            phi_values,
            tolerance=tolerance,
            solver_seed=solver_seed,
            threads=threads,
            env=env,
        )
        separation_time += time.perf_counter() - separation_start

        new_cuts = []
        for cut in cuts:
            key = _cut_key(cut, data["edges"])
            if key not in cut_keys:
                new_cuts.append((key, cut))

        if not new_cuts:
            separation_complete = True
            break
        if cut_iterations >= max_iterations:
            limit_reached = True
            break

        if max_cuts is not None:
            remaining_cuts = max_cuts - cut_count
            if remaining_cuts <= 0:
                limit_reached = True
                break
            new_cuts = new_cuts[:remaining_cuts]

        for key, cut in new_cuts:
            _model_constraint(mod, variables, cut, cut_count)
            cut_keys.add(key)
            cut_count += 1
            cuts_by_family[cut["family"]] += 1

        cut_iterations += 1
        cut_history.append(len(new_cuts))
        mod.update()

    wall_time = time.perf_counter() - start
    result = collect_model_result(mod, formulation=4, relax=True)
    result["runtime"] = wall_time
    result["solver_runtime"] = float(solver_runtime)
    result["num_constraints"] = int(base_constraints + cut_count)
    result["num_linear_constraints"] = result["num_constraints"]
    result["cuts"] = cut_count
    result["cuts_by_family"] = cuts_by_family
    result["cut_iterations"] = cut_iterations
    result["master_solves"] = master_solves
    result["separation_time"] = separation_time
    result["separation_complete"] = separation_complete
    result["limit_reached"] = limit_reached
    result["objective_history"] = objective_history
    result["cut_history"] = cut_history

    if mod.SolCount > 0:
        result["variables"] = variable_values(variables)

    mod.dispose()
    return result


def solve_instance(
    instance,
    relax=False,
    time_limit=None,
    output_flag=0,
    tolerance=1e-6,
    max_iterations=100,
    max_cuts=None,
    solver_seed=0,
    threads=1,
    env=None,
):
    if relax:
        return _solve_relaxation(
            instance,
            time_limit=time_limit,
            output_flag=output_flag,
            tolerance=tolerance,
            max_iterations=max_iterations,
            max_cuts=max_cuts,
            solver_seed=solver_seed,
            threads=threads,
            env=env,
        )

    return _solve_integer(
        instance,
        time_limit=time_limit,
        output_flag=output_flag,
        tolerance=tolerance,
        solver_seed=solver_seed,
        threads=threads,
        env=env,
    )
