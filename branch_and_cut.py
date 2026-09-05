import math
import time

from gurobipy import GRB, Model, quicksum

from formulations import build_formulation_4
from graph_algorithms import directed_min_cut, shortest_path
from instances import prepare_instance
from time_budget import BudgetExpired, TimeBudget
from utils import (
    STATUS_NAMES,
    MemoryLimitReached,
    collect_model_result,
    empty_model_result,
    variable_values,
)


def _prepared_data(instance):
    if isinstance(instance, dict) and "tau" in instance and "balance" in instance:
        return instance

    return prepare_instance(instance)


def solve_journeyer_dual(
    instance,
    journeyer,
    capacities,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
    budget=None,
):
    budget = budget if budget is not None else TimeBudget()
    budget.check()
    data = _prepared_data(instance)
    V = data["nodes"]
    E = data["edges"]
    tau = data["tau"]
    s_j = data["journeyer_source"]
    t_j = data["journeyer_target"]

    budget.check()
    dual = Model(f"journeyer_dual_{journeyer}", env=env)

    try:
        # Variables of DP^j. Model construction is one cooperative operation.
        rho = {v: dual.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"rho[{v}]") for v in V}
        delta = {e: dual.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0, name=f"delta[{e[0]},{e[1]}]") for e in E}
        rho[s_j[journeyer]].LB = 0
        rho[s_j[journeyer]].UB = 0
        dual.update()

        objective = rho[t_j[journeyer]] - rho[s_j[journeyer]] + quicksum(delta[e] * capacities[e] for e in E)
        dual.setObjective(objective, GRB.MAXIMIZE)
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
        budget.apply_to(dual)
        dual.optimize()

        if dual.Status == GRB.TIME_LIMIT:
            raise BudgetExpired()
        if dual.Status == GRB.MEM_LIMIT:
            raise MemoryLimitReached("Journeyer dual reached the memory limit")
        if dual.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Journeyer dual {journeyer} ended with status {dual.Status}"
            )
        rho_value = {
            v: float(rho[v].X - rho[s_j[journeyer]].X)
            for v in V
        }
        delta_value = {
            e: min(0.0, tau[e] - rho_value[e[1]] + rho_value[e[0]])
            for e in E
        }
        max_dual_violation = max(
            [abs(rho_value[s_j[journeyer]]), max(delta_value.values(), default=0.0)]
            + [rho_value[e[1]] - rho_value[e[0]] + delta_value[e] - tau[e] for e in E]
        )
        if max_dual_violation > 1e-7:
            raise RuntimeError(
                f"Numerically infeasible dual solution for journeyer {journeyer}"
            )

        return {
            "objective_value": float(
                rho_value[t_j[journeyer]]
                - rho_value[s_j[journeyer]]
                + sum(delta_value[e] * capacities[e] for e in E)
            ),
            "rho": rho_value,
            "delta": delta_value,
            "max_dual_violation": float(max(0.0, max_dual_violation)),
        }
    finally:
        dual.dispose()


def separate_solution(
    instance,
    x_values,
    alpha_values,
    phi_values,
    tolerance=1e-6,
    solver_seed=0,
    threads=1,
    env=None,
    budget=None,
):
    budget = budget if budget is not None else TimeBudget()
    budget.check()
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
        budget.check()
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
        budget.check()
        capacities_raw = {
            e: alpha_raw[j, e] + 1 - x_raw[e]
            for e in E
        }
        capacities_algorithm = {
            e: alpha_algorithm[j, e] + 1 - x_algorithm[e]
            for e in E
        }
        budget.check()
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
            budget=budget,
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
    if where == GRB.Callback.MESSAGE:
        return
    if mod._memory_limit:
        mod.terminate()
        return
    if mod._budget.expired():
        mod._budget_expired = True
        mod.terminate()
        return
    if where != GRB.Callback.MIPSOL:
        return

    separation_start = time.perf_counter()
    mod._callback_calls += 1

    try:
        mod._budget.check()
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
            budget=mod._budget,
        )

        if not cuts:
            # A terminated lazy callback can leave an unchecked incumbent in
            # Gurobi. Only candidates whose complete separation passed are safe
            # to expose when optimization stops.
            objective = float(mod.cbGet(GRB.Callback.MIPSOL_OBJ))
            if mod._best_solution is None or objective < mod._best_solution["objective_value"]:
                mod._best_solution = {
                    "objective_value": objective,
                    "variables": {"x": x_values, "alpha": alpha_values, "phi": phi_values},
                }

        for cut in cuts:
            mod._budget.check()
            key = _cut_key(cut, mod._data["edges"])
            # Gurobi may present the same incumbent more than once. A violated
            # lazy constraint must be submitted every time it is encountered.
            _lazy_constraint(mod, cut)
            mod._lazy_additions += 1

            if key not in mod._cut_keys:
                mod._cut_keys.add(key)
                mod._cut_count += 1
                mod._cuts_by_family[cut["family"]] += 1
    except BudgetExpired:
        mod._budget_expired = True
        mod.terminate()
    except MemoryLimitReached:
        mod._memory_limit = True
        mod.terminate()
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
    budget = TimeBudget(time_limit)
    mod = None
    optimized = False
    timed_out = False
    memory_limited = False
    try:
        try:
            budget.check()
            mod, variables = build_formulation_4(
                instance,
                relax=False,
                time_limit=budget.remaining(),
                output_flag=output_flag,
                solver_seed=solver_seed,
                threads=threads,
                env=env,
            )
            mod._x = variables["x"]
            mod._alpha = variables["alpha"]
            mod._phi = variables["phi"]
            mod._tolerance = tolerance
            mod._solver_seed = solver_seed
            mod._threads = threads
            mod._budget = budget
            mod._budget_expired = False
            mod._memory_limit = False
            mod._best_solution = None
            mod._cut_keys = set()
            mod._cut_count = 0
            mod._lazy_additions = 0
            mod._cuts_by_family = {"intruder": 0, "feasibility": 0, "optimality": 0}
            mod._callback_calls = 0
            mod._separation_time = 0.0
            mod._callback_error = None
            base_constraints = mod.NumConstrs

            budget.apply_to(mod)
            optimized = True
            mod.optimize(_branch_and_cut_callback)
            if mod._callback_error is not None:
                error = mod._callback_error
                mod._callback_error = None
                raise error
            timed_out = mod._budget_expired or mod.Status == GRB.TIME_LIMIT
            memory_limited = mod._memory_limit or mod.Status == GRB.MEM_LIMIT
        except BudgetExpired:
            timed_out = True

        runtime = budget.elapsed()
        if optimized:
            result = collect_model_result(mod, formulation=4, relax=False)
            snapshot = mod._best_solution
            result["objective_value"] = snapshot["objective_value"] if snapshot else None
            result["variables"] = snapshot["variables"] if snapshot else {}
            result["has_solution"] = snapshot is not None
            result["solution_type"] = "integer" if snapshot else "none"
            result["gap"] = _relative_gap(result["objective_value"], result["dual_bound"])
            result["num_constraints"] = int(base_constraints + mod._cut_count)
            result["num_linear_constraints"] = result["num_constraints"]
            result["cuts"] = mod._cut_count
            result["lazy_additions"] = mod._lazy_additions
            result["cuts_by_family"] = dict(mod._cuts_by_family)
            result["cut_iterations"] = mod._callback_calls
            result["callback_calls"] = mod._callback_calls
            result["separation_time"] = mod._separation_time
            result["remaining_violated_cuts"] = 0 if snapshot else None
            result["separation_complete"] = (
                mod.Status == GRB.OPTIMAL and snapshot is not None
                and not timed_out and not memory_limited
            )
            if snapshot is None and result["status"] == GRB.OPTIMAL:
                _set_status(result, GRB.INTERRUPTED)
        else:
            result = empty_model_result(4, False, mod)

        if timed_out:
            _set_status(result, GRB.TIME_LIMIT)
        if memory_limited:
            _set_status(result, GRB.MEM_LIMIT)
            result["convergence_reason"] = "memory_limit"
        result["limit_reached"] = timed_out or memory_limited
        result["runtime"] = runtime
        return result
    finally:
        if mod is not None:
            mod.dispose()


def _set_status(result, status):
    result["status"] = int(status)
    result["status_name"] = STATUS_NAMES.get(status, f"STATUS_{status}")


def _relative_gap(objective, bound):
    if objective is None or bound is None:
        return None
    if abs(objective) <= 1e-10:
        return 0.0 if abs(objective - bound) <= 1e-10 else None
    return abs(objective - bound) / abs(objective)


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
    budget = TimeBudget(time_limit)
    mod = None
    base_constraints = 0
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
    last_result = None
    snapshot = None
    best_bound = None
    stop_status = None
    convergence_reason = None
    try:
        try:
            budget.check()
            mod, variables = build_formulation_4(
                instance,
                relax=True,
                time_limit=budget.remaining(),
                output_flag=output_flag,
                solver_seed=solver_seed,
                threads=threads,
                env=env,
            )
            base_constraints = mod.NumConstrs
            data = mod._data

            while True:
                budget.apply_to(mod)
                mod.optimize()
                master_solves += 1
                solver_runtime += mod.Runtime
                last_result = collect_model_result(mod, formulation=4, relax=True)
                bound = last_result["dual_bound"]
                if bound is not None and math.isfinite(bound):
                    best_bound = bound if best_bound is None else max(best_bound, bound)

                # Updating a master or interrupting its next solve can destroy
                # its incumbent attributes. Keep the last available primal and
                # the strongest lower bound separately before doing either.
                if last_result["objective_value"] is not None:
                    snapshot = {
                        "objective_value": last_result["objective_value"],
                        "variables": variable_values(variables),
                    }
                if mod.Status != GRB.OPTIMAL or mod.SolCount == 0:
                    stop_status = mod.Status if mod.Status != GRB.OPTIMAL else GRB.INTERRUPTED
                    limit_reached = True
                    convergence_reason = {
                        GRB.TIME_LIMIT: "time_limit", GRB.MEM_LIMIT: "memory_limit",
                    }.get(stop_status, "master_stopped")
                    break

                objective_history.append(float(mod.ObjVal))
                budget.check()
                separation_start = time.perf_counter()
                try:
                    cuts = separate_solution(
                        data,
                        snapshot["variables"]["x"],
                        snapshot["variables"]["alpha"],
                        snapshot["variables"]["phi"],
                        tolerance=tolerance,
                        solver_seed=solver_seed,
                        threads=threads,
                        env=env,
                        budget=budget,
                    )
                finally:
                    separation_time += time.perf_counter() - separation_start
                if not cuts:
                    separation_complete = True
                    convergence_reason = "no_violated_cuts"
                    break

                budget.check()
                new_cuts = []
                for cut in cuts:
                    budget.check()
                    key = _cut_key(cut, data["edges"])
                    if key not in cut_keys:
                        new_cuts.append((key, cut))
                if not new_cuts:
                    stop_status = GRB.NUMERIC
                    convergence_reason = "repeated_violated_cuts"
                    break
                if cut_iterations >= max_iterations:
                    limit_reached = True
                    stop_status = GRB.ITERATION_LIMIT
                    convergence_reason = "iteration_limit"
                    break
                if max_cuts is not None:
                    remaining_cuts = max_cuts - cut_count
                    if remaining_cuts <= 0:
                        limit_reached = True
                        stop_status = GRB.INTERRUPTED
                        convergence_reason = "cut_limit"
                        break
                    new_cuts = new_cuts[:remaining_cuts]

                for key, cut in new_cuts:
                    budget.check()
                    _model_constraint(mod, variables, cut, cut_count)
                    cut_keys.add(key)
                    cut_count += 1
                    cuts_by_family[cut["family"]] += 1
                cut_iterations += 1
                cut_history.append(len(new_cuts))
                budget.check()
                mod.update()
        except BudgetExpired:
            limit_reached = True
            stop_status = GRB.TIME_LIMIT
            convergence_reason = "time_limit"
        except MemoryLimitReached:
            limit_reached = True
            stop_status = GRB.MEM_LIMIT
            convergence_reason = "separation_memory_limit"

        runtime = budget.elapsed()
        result = dict(last_result) if last_result is not None else empty_model_result(4, True, mod)
        if stop_status is not None:
            _set_status(result, stop_status)
        result["objective_value"] = snapshot["objective_value"] if snapshot else None
        result["variables"] = snapshot["variables"] if snapshot else {}
        result["has_solution"] = snapshot is not None
        result["solution_type"] = (
            ("relaxation" if separation_complete else "restricted_master")
            if snapshot else "none"
        )
        result["dual_bound"] = best_bound
        result["gap"] = _relative_gap(result["objective_value"], best_bound) if separation_complete else None
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
        result["convergence_reason"] = convergence_reason
        result["objective_history"] = objective_history
        result["cut_history"] = cut_history
        result["runtime"] = runtime
        return result
    finally:
        if mod is not None:
            mod.dispose()


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
