"""Selected subset inequalities for the aggregated formulation, at the root only."""

import math
import time
from numbers import Integral

from gurobipy import GRB, quicksum

from formulations import build_formulation_3_VI
from time_budget import BudgetExpired, TimeBudget
from utils import STATUS_NAMES, collect_model_result, empty_model_result, variable_values

DEFAULT_MAX_CUTS = 100
DEFAULT_MAX_CUTS_PER_ROUND = 20
DEFAULT_MAX_ITERATIONS = 100


def validate_cut_limits(max_cuts, max_cuts_per_round, max_iterations, tolerance):
    for name, value in (
        ("max_cuts", max_cuts),
        ("max_cuts_per_round", max_cuts_per_round),
        ("max_iterations", max_iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")


def separate_valid_inequalities(
    edges, journeyers, x, z, beta, *, tolerance=1e-6, budget=None,
):
    """Select K = {j: z[j,e] + x[e] - 1 > 0}, then rank violated cuts.

    Empty and full subsets already occur in the aggregated model.
    Instance order breaks ties deterministically, without enumerating subsets.
    """
    cuts = []
    for edge in edges:
        if budget is not None:
            budget.check()
        subset = tuple(j for j in journeyers if z[j, edge] + x[edge] - 1 > 0)
        if not subset or len(subset) == len(journeyers):
            continue
        violation = sum(z[j, edge] + x[edge] - 1 for j in subset) - beta[edge]
        if violation > tolerance:
            cuts.append({"edge": edge, "journeyers": subset, "violation": violation})
    cuts.sort(key=lambda cut: -cut["violation"])
    return cuts


def _constraint(variables, cut):
    edge, subset = cut["edge"], cut["journeyers"]
    return variables["beta"][edge] >= (
        quicksum(variables["z"][j, edge] for j in subset)
        - len(subset) * (1 - variables["x"][edge])
    )


def _cut_key(cut):
    return cut["edge"], cut["journeyers"]


class RootSeparator:
    """Shared cut selection and counters for the LP loop and root MIP callback."""

    def __init__(self, variables, budget, max_cuts, max_cuts_per_round,
                 max_iterations, tolerance):
        self.variables = variables
        self.edges = tuple(variables["x"])
        self.journeyers = tuple(dict.fromkeys(j for j, _ in variables["z"]))
        self.budget = budget
        self.max_cuts = max_cuts
        self.max_cuts_per_round = max_cuts_per_round
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.keys = set()
        self.cut_history = []
        self.separation_time = 0.0
        self.complete = False
        self.reason = None
        self.error = None

    def limit_reason(self):
        if len(self.keys) >= self.max_cuts or self.max_cuts_per_round == 0:
            return "cut_limit"
        if len(self.cut_history) >= self.max_iterations:
            return "iteration_limit"
        return None

    def select(self, values):
        self.complete = False
        start = time.perf_counter()
        try:
            cuts = separate_valid_inequalities(
                self.edges, self.journeyers, values["x"], values["z"], values["beta"],
                tolerance=self.tolerance, budget=self.budget,
            )
        finally:
            self.separation_time += time.perf_counter() - start
        if not cuts:
            self.complete = True
            self.reason = "no_violated_cuts"
            return []
        self.reason = self.limit_reason()
        if self.reason is not None:
            return []
        new_cuts = [cut for cut in cuts if _cut_key(cut) not in self.keys]
        if not new_cuts:
            self.reason = "repeated_violated_cuts"
        return new_cuts[:min(self.max_cuts_per_round, self.max_cuts - len(self.keys))]

    def record(self, cuts):
        self.keys.update(_cut_key(cut) for cut in cuts)
        self.cut_history.append(len(cuts))

    def callback(self, model, where):
        # Node count stays zero throughout the root cut passes. Never separate
        # MIPSOL or non-root MIPNODE events; these are optional user cuts.
        if where != GRB.Callback.MIPNODE or self.error is not None:
            return
        try:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0:
                return
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            if self.limit_reason() is not None:
                self.reason = self.limit_reason()
                return
            self.budget.check()
            values = {
                name: dict(zip(mapping, model.cbGetNodeRel(list(mapping.values()))))
                for name, mapping in self.variables.items() if name in ("x", "z", "beta")
            }
            cuts = self.select(values)
            self.budget.check()
            for cut in cuts:
                model.cbCut(_constraint(self.variables, cut))
            if cuts:
                self.record(cuts)
        except BudgetExpired:
            self.reason = "time_limit"
            model.terminate()
        except Exception as error:
            # Gurobi otherwise logs and swallows Python callback exceptions.
            self.error = error
            model.terminate()

    def statistics(self):
        return {
            "cuts": len(self.keys),
            "cut_iterations": len(self.cut_history),
            "cut_history": list(self.cut_history),
            "separation_time": self.separation_time,
            "separation_complete": self.complete,
            "convergence_reason": self.reason,
            "max_cuts": self.max_cuts,
            "max_cuts_per_round": self.max_cuts_per_round,
            "max_iterations": self.max_iterations,
        }


def _solve_relaxation(model, variables, separator, budget):
    result = empty_model_result("3_VI", True, model)
    snapshot = None
    best_bound = None
    pending_cuts = False
    solver_runtime = 0.0
    master_solves = 0
    simplex_iterations = 0.0
    objective_history = []
    try:
        while True:
            budget.apply_to(model)
            model.optimize()
            master_solves += 1
            solver_runtime += model.Runtime
            simplex_iterations += model.IterCount
            result = collect_model_result(model, "3_VI", True)
            bound = result["dual_bound"]
            if bound is not None:
                best_bound = bound if best_bound is None else max(best_bound, bound)
            if result["has_solution"]:
                snapshot = (result["objective_value"], variable_values(variables))
                pending_cuts = False
            if model.Status != GRB.OPTIMAL:
                separator.reason = STATUS_NAMES.get(model.Status, "solver_stopped").lower()
                break
            objective_history.append(result["objective_value"])
            cuts = separator.select(snapshot[1])
            if not cuts:
                break
            budget.check()
            for position, cut in enumerate(cuts, start=len(separator.keys)):
                model.addConstr(_constraint(variables, cut), name=f"subset_VI[{position}]")
            separator.record(cuts)
            pending_cuts = True
            model.update()
    except BudgetExpired:
        result.update(status=GRB.TIME_LIMIT, status_name="TIME_LIMIT")
        separator.reason = "time_limit"
    if snapshot is not None:
        result.update(
            objective_value=snapshot[0], variables=snapshot[1], has_solution=True,
            solution_type="restricted_master" if pending_cuts else "relaxation",
        )
    result.update(
        dual_bound=best_bound, solver_runtime=solver_runtime, master_solves=master_solves,
        simplex_iterations=simplex_iterations, objective_history=objective_history,
        num_constraints=model.NumConstrs, num_linear_constraints=model.NumConstrs,
    )
    if result["status"] != GRB.OPTIMAL or pending_cuts:
        result["gap"] = None
    return result


def solve_instance(
    instance, relax=False, time_limit=None, output_flag=0, solver_seed=0,
    threads=1, env=None, max_cuts=DEFAULT_MAX_CUTS,
    max_cuts_per_round=DEFAULT_MAX_CUTS_PER_ROUND,
    max_iterations=DEFAULT_MAX_ITERATIONS, tolerance=1e-6,
):
    """Solve formulation 3_VI with a hard cap on generated inequalities.

    Integer models use cbCut only at root MIPNODE events. Continuous models
    solve/separate/reoptimize, including a final solve after the last cut batch.
    OPTIMAL for a capped LP describes the selected-cut model; check
    separation_complete to determine whether the full family was separated.
    """
    validate_cut_limits(max_cuts, max_cuts_per_round, max_iterations, tolerance)
    budget = TimeBudget(time_limit)
    model = None
    separator = None
    try:
        try:
            budget.check()
            model, variables = build_formulation_3_VI(
                instance, relax=relax, time_limit=budget.remaining(),
                output_flag=output_flag, solver_seed=solver_seed, threads=threads, env=env,
            )
            separator = RootSeparator(
                variables, budget, max_cuts, max_cuts_per_round, max_iterations, tolerance,
            )
            if relax:
                result = _solve_relaxation(model, variables, separator, budget)
            else:
                budget.apply_to(model)
                model.optimize(separator.callback)
                if separator.error is not None:
                    raise separator.error
                result = collect_model_result(model, "3_VI", False)
                if model.SolCount:
                    result["variables"] = variable_values(variables)
                if separator.reason == "time_limit" and model.Status == GRB.INTERRUPTED:
                    result.update(status=GRB.TIME_LIMIT, status_name="TIME_LIMIT")
                if separator.reason is None:
                    separator.reason = "no_root_separation"
        except BudgetExpired:
            result = empty_model_result("3_VI", relax, model)
            if separator is not None:
                separator.reason = "time_limit"
        result.update(
            max_cuts=max_cuts, max_cuts_per_round=max_cuts_per_round,
            max_iterations=max_iterations, cut_tolerance=tolerance,
        )
        if separator is not None:
            result.update(separator.statistics())
        else:
            result["convergence_reason"] = "time_limit"
        result["runtime"] = budget.elapsed()
        return result
    finally:
        if model is not None:
            model.dispose()
