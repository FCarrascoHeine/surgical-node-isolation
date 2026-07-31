import heapq
import itertools
import math

from instances import prepare_instance


def _path_exists(nodes, edges, source, target, blocked_edges=None):
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


def _shortest_distance(nodes, edges, costs, source, target):
    outgoing = {v: [] for v in nodes}

    for edge in edges:
        outgoing[edge[0]].append(edge)

    distance = {v: math.inf for v in nodes}
    counter = itertools.count()
    distance[source] = 0.0
    pending = [(0.0, next(counter), source)]

    while pending:
        current_distance, _, v = heapq.heappop(pending)

        if current_distance > distance[v] + 1e-12:
            continue
        if v == target:
            return current_distance

        for edge in outgoing[v]:
            v_prime = edge[1]
            new_distance = current_distance + costs[edge]

            if new_distance < distance[v_prime] - 1e-12:
                distance[v_prime] = new_distance
                heapq.heappush(
                    pending,
                    (new_distance, next(counter), v_prime),
                )

    return math.inf


def evaluate_allocation(instance, x_values, tolerance=1e-6):
    data = prepare_instance(instance)
    V = data["nodes"]
    E = data["edges"]
    I = data["intruders"]
    J = data["journeyers"]
    tau = data["tau"]
    t = data["inspection_time"]
    s_i = data["intruder_source"]
    t_i = data["intruder_target"]
    s_j = data["journeyer_source"]
    t_j = data["journeyer_target"]
    errors = []

    if any(e not in x_values for e in E):
        errors.append("The solution does not contain every x variable")
        return {
            "valid": False,
            "errors": errors,
            "objective_value": None,
            "selected_edges": [],
            "journeyer_distances": {},
        }

    for e in E:
        value = float(x_values[e])
        if value < -tolerance or value > 1 + tolerance:
            errors.append(f"x{e} is outside [0,1]")
        if abs(value - round(value)) > tolerance:
            errors.append(f"x{e} is not integral")

    selected_edges = {
        e
        for e in E
        if float(x_values[e]) >= 0.5
    }

    if len(selected_edges) > data["budget"]:
        errors.append("The checkpoint allocation exceeds the budget")

    for i in I:
        if _path_exists(V, E, s_i[i], t_i[i], blocked_edges=selected_edges):
            errors.append(f"Intruder {i} is not deterred")

    costs = {
        e: tau[e] + (t[e] if e in selected_edges else 0.0)
        for e in E
    }
    journeyer_distances = {
        j: _shortest_distance(V, E, costs, s_j[j], t_j[j])
        for j in J
    }

    if any(not math.isfinite(value) for value in journeyer_distances.values()):
        errors.append("A journeyer has no path under the allocation")

    objective_value = sum(journeyer_distances.values())

    return {
        "valid": not errors,
        "errors": errors,
        "objective_value": float(objective_value),
        "selected_edges": sorted(selected_edges),
        "journeyer_distances": journeyer_distances,
    }


def enumerate_original_problem(instance, max_edges=18):
    data = prepare_instance(instance)
    E = data["edges"]

    if len(E) > max_edges:
        raise ValueError(
            f"Enumeration is limited to {max_edges} edges"
        )

    feasible_allocations = []

    for size in range(data["budget"] + 1):
        for selected_tuple in itertools.combinations(E, size):
            selected_edges = set(selected_tuple)
            x_values = {
                e: float(e in selected_edges)
                for e in E
            }
            evaluation = evaluate_allocation(data["instance"], x_values)

            if evaluation["valid"]:
                feasible_allocations.append(
                    {
                        "objective_value": evaluation["objective_value"],
                        "selected_edges": tuple(sorted(selected_edges)),
                    }
                )

    if not feasible_allocations:
        return {
            "objective_value": None,
            "optimal_allocations": [],
            "feasible_allocations": [],
        }

    objective_value = min(
        allocation["objective_value"]
        for allocation in feasible_allocations
    )
    optimal_allocations = [
        allocation["selected_edges"]
        for allocation in feasible_allocations
        if abs(allocation["objective_value"] - objective_value) <= 1e-9
    ]

    return {
        "objective_value": float(objective_value),
        "optimal_allocations": optimal_allocations,
        "feasible_allocations": feasible_allocations,
    }


def _validate_common_variables(data, result, tolerance, errors):
    V = data["nodes"]
    E = data["edges"]
    I = data["intruders"]
    J = data["journeyers"]
    b = data["balance"]
    variables = result["variables"]

    if "z" in variables:
        z = variables["z"]

        for j in J:
            for e in E:
                value = float(z[j, e])
                if value < -tolerance or value > 1 + tolerance:
                    errors.append(f"z[{j},{e}] is outside [0,1]")
                if abs(value - round(value)) > tolerance:
                    errors.append(f"z[{j},{e}] is not integral")

            for v in V:
                inflow = sum(z[j, e] for e in E if e[1] == v)
                outflow = sum(z[j, e] for e in E if e[0] == v)

                if abs(inflow - outflow - b[j, v]) > tolerance:
                    errors.append(
                        f"Journeyer balance is violated for ({j},{v})"
                    )

    if "y" in variables:
        y = variables["y"]
        s_i = data["intruder_source"]
        t_i = data["intruder_target"]
        x = variables["x"]

        for i in I:
            if abs(y[i, s_i[i]]) > tolerance:
                errors.append("Intruder source potential is not zero")
            if y[i, t_i[i]] < 1 - tolerance:
                errors.append("Intruder target potential is smaller than one")

            for e in E:
                if y[i, e[1]] - y[i, e[0]] > x[e] + tolerance:
                    errors.append("An intruder potential constraint is violated")


def validate_integer_result(instance, result, tolerance=1e-6):
    data = prepare_instance(instance)
    E = data["edges"]
    J = data["journeyers"]
    tau = data["tau"]
    t = data["inspection_time"]
    errors = []

    if result.get("objective_value") is None:
        return {
            "valid": False,
            "errors": ["The formulation did not return an incumbent"],
            "original_objective": None,
        }
    if not result.get("variables") or "x" not in result["variables"]:
        return {
            "valid": False,
            "errors": ["The formulation did not return x variables"],
            "original_objective": None,
        }

    variables = result["variables"]
    x = variables["x"]
    allocation = evaluate_allocation(data["instance"], x, tolerance=tolerance)
    errors.extend(allocation["errors"])
    _validate_common_variables(data, result, tolerance, errors)

    formulation = result["formulation"]
    modeled_objective = None

    if formulation == 1:
        z = variables["z"]
        modeled_objective = sum(
            tau[e] * z[j, e] + t[e] * x[e] * z[j, e]
            for j in J
            for e in E
        )

    if formulation == 2:
        z = variables["z"]
        alpha = variables["alpha"]
        modeled_objective = sum(
            tau[e] * z[j, e] + t[e] * alpha[j, e]
            for j in J
            for e in E
        )

        for j in J:
            for e in E:
                if alpha[j, e] < x[e] + z[j, e] - 1 - tolerance:
                    errors.append("Constraint (7) is violated")
                if abs(alpha[j, e] - round(alpha[j, e])) > tolerance:
                    errors.append("An alpha variable is not integral")

    if formulation == 3:
        z = variables["z"]
        beta = variables["beta"]
        modeled_objective = sum(
            tau[e] * sum(z[j, e] for j in J) + t[e] * beta[e]
            for e in E
        )

        for e in E:
            flow = sum(z[j, e] for j in J)
            if beta[e] > flow + tolerance:
                errors.append("Constraint (17) is violated")
            if beta[e] < flow - len(J) * (1 - x[e]) - tolerance:
                errors.append("Constraint (18) is violated")
            if beta[e] > len(J) * x[e] + tolerance:
                errors.append("Constraint (19) is violated")

    if formulation == 4:
        alpha = variables["alpha"]
        phi = variables["phi"]
        modeled_objective = sum(
            sum(t[e] * alpha[j, e] for e in E) + phi[j]
            for j in J
        )

        for j in J:
            if phi[j] < -tolerance:
                errors.append("A phi variable is negative")

            for e in E:
                if alpha[j, e] > x[e] + tolerance:
                    errors.append("Constraint (27) is violated")
                if abs(alpha[j, e] - round(alpha[j, e])) > tolerance:
                    errors.append("An alpha variable is not integral")

    if modeled_objective is None:
        errors.append("Unknown formulation")
    elif abs(modeled_objective - result["objective_value"]) > tolerance * max(
        1.0,
        abs(result["objective_value"]),
    ):
        errors.append("The returned variables do not reproduce the model objective")

    if allocation["objective_value"] is not None and abs(
        allocation["objective_value"] - result["objective_value"]
    ) > tolerance * max(1.0, abs(result["objective_value"])):
        errors.append("The model objective differs from the original SNI objective")

    if result.get("dual_bound") is not None and (
        result["dual_bound"] > result["objective_value"] + tolerance
    ):
        errors.append("The dual bound is larger than the incumbent")

    return {
        "valid": not errors,
        "errors": errors,
        "original_objective": allocation["objective_value"],
        "selected_edges": allocation["selected_edges"],
        "journeyer_distances": allocation["journeyer_distances"],
    }


def validate_relaxation_bound(result, integer_optimum, tolerance=1e-6):
    if result.get("status_name") == "OPTIMAL":
        lower_bound = result.get("objective_value")
    else:
        lower_bound = result.get("dual_bound")

    if lower_bound is None:
        return {
            "valid": False,
            "error": "No valid lower bound was returned",
            "lower_bound": None,
        }

    valid = lower_bound <= integer_optimum + tolerance * max(
        1.0,
        abs(integer_optimum),
    )

    return {
        "valid": valid,
        "error": None if valid else "The relaxation is not a valid lower bound",
        "lower_bound": lower_bound,
    }


def validate_cut(
    instance,
    cut,
    x_values=None,
    alpha_values=None,
    phi_values=None,
    tolerance=1e-6,
):
    data = prepare_instance(instance)
    E = data["edges"]
    tau = data["tau"]
    errors = []
    family = cut.get("family")

    if family == "intruder":
        i = cut["agent"]
        path = list(cut["path"])
        source = data["intruder_source"][i]
        target = data["intruder_target"][i]

        if not path or path[0][0] != source or path[-1][1] != target:
            errors.append("The intruder cut does not contain an s_i-t_i path")
        for k in range(len(path) - 1):
            if path[k][1] != path[k + 1][0]:
                errors.append("The intruder path is not contiguous")
        if any(edge not in E for edge in path):
            errors.append("The intruder path contains an unknown edge")

        if x_values is not None:
            value = sum(x_values[e] for e in path)
            if value >= 1 - tolerance:
                errors.append("The intruder cut is not violated")

    elif family == "feasibility":
        j = cut["agent"]
        cut_set = set(cut["cut_set"])
        source = data["journeyer_source"][j]
        target = data["journeyer_target"][j]
        expected_edges = tuple(
            e
            for e in E
            if e[0] in cut_set and e[1] not in cut_set
        )

        if source not in cut_set or target in cut_set:
            errors.append("The feasibility cut set has the wrong terminals")
        if tuple(cut["cut_edges"]) != expected_edges:
            errors.append("The feasibility cut does not equal delta+(S)")

        if x_values is not None and alpha_values is not None:
            value = sum(
                alpha_values[j, e] + 1 - x_values[e]
                for e in expected_edges
            )
            if value >= 1 - tolerance:
                errors.append("The feasibility cut is not violated")

    elif family == "optimality":
        j = cut["agent"]
        rho = cut.get("rho")
        delta = cut["delta"]

        if rho is None:
            errors.append("The optimality cut does not include rho")
        else:
            source = data["journeyer_source"][j]
            if abs(rho[source]) > tolerance:
                errors.append("rho at the journeyer source is not zero")

            for e in E:
                if delta[e] > tolerance:
                    errors.append("A dual delta variable is positive")
                if rho[e[1]] - rho[e[0]] + delta[e] > tau[e] + tolerance:
                    errors.append("A dual arc constraint is violated")

        if (
            x_values is not None
            and alpha_values is not None
            and phi_values is not None
        ):
            right_hand_side = cut["rho_difference"] + sum(
                delta[e] * (alpha_values[j, e] + 1 - x_values[e])
                for e in E
            )
            if right_hand_side <= phi_values[j] + tolerance:
                errors.append("The optimality cut is not violated")

    else:
        errors.append("Unknown cut family")

    return {
        "valid": not errors,
        "errors": errors,
    }
