"""Subset cuts for formulation 3: an LP cutting-plane loop or a root MIP callback."""

import math
import time
from numbers import Integral

from gurobipy import GRB, quicksum

from formulations import build_formulation_3_VI
from time_budget import BudgetExpired, TimeBudget
from utils import (
    STATUS_NAMES,
    collect_model_result,
    empty_model_result,
    variable_values,
)

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

    The disaggregated lower bounds give beta[e] >= sum_j max(0, x[e]+z[j,e]-1).
    Selecting every positive term produces the most violated subset inequality.
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


class CutPool:
    """Select new cuts, enforce limits, and keep their history for either solver."""

    def __init__(self, variables, budget, max_cuts, max_cuts_per_round,
                 max_iterations, tolerance):
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

    def limit_reason(self):
        if len(self.keys) >= self.max_cuts or self.max_cuts_per_round == 0:
            return "cut_limit"
        if len(self.cut_history) >= self.max_iterations:
            return "iteration_limit"
        return None

    def select(self, values):
        """Return (cuts, stop_reason); an empty list need not mean convergence."""
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
            return [], "no_violated_cuts"
        # Check convergence before limits: the final allowed batch may suffice.
        reason = self.limit_reason()
        if reason is not None:
            return [], reason
        new_cuts = [cut for cut in cuts if _cut_key(cut) not in self.keys]
        if not new_cuts:
            return [], "repeated_violated_cuts"
        batch_size = min(self.max_cuts_per_round, self.max_cuts - len(self.keys))
        return new_cuts[:batch_size], None

    def record(self, cuts):
        self.keys.update(_cut_key(cut) for cut in cuts)
        self.cut_history.append(len(cuts))

    def statistics(self):
        return {
            "cuts": len(self.keys),
            "cut_iterations": len(self.cut_history),
            "cut_history": list(self.cut_history),
            "separation_time": self.separation_time,
            "separation_complete": self.complete,
        }


def _solve_integer(model, variables, pool, budget):
    """Submit user cuts only at optimal root relaxations during the MIP solve."""
    reason = None
    callback_error = None

    def callback(model, where):
        nonlocal reason, callback_error
        # Node count stays zero throughout the root cut passes. Never separate
        # MIPSOL or non-root MIPNODE events; these are optional user cuts.
        if where != GRB.Callback.MIPNODE or callback_error is not None:
            return
        try:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0:
                return
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            reason = pool.limit_reason()
            if reason is not None:
                return
            budget.check()
            values = {
                name: model.cbGetNodeRel(variables[name])
                for name in ("x", "z", "beta")
            }
            cuts, reason = pool.select(values)
            budget.check()
            for cut in cuts:
                model.cbCut(_constraint(variables, cut))
            if cuts:
                pool.record(cuts)
        except BudgetExpired:
            reason = "time_limit"
            model.terminate()
        except Exception as error:  # noqa: BLE001
            # Gurobi otherwise logs and swallows Python callback exceptions.
            callback_error = error
            model.terminate()

    budget.apply_to(model)
    model.optimize(callback)
    if callback_error is not None:
        raise callback_error
    result = collect_model_result(model, "3_VI", False)
    if model.SolCount:
        result["variables"] = variable_values(variables)
    if reason == "time_limit" and model.Status == GRB.INTERRUPTED:
        result.update(status=GRB.TIME_LIMIT, status_name="TIME_LIMIT")
    result["convergence_reason"] = reason or "no_root_separation"
    return result


def _collect_lp_solve(model, variables, previous):
    """Accumulate work and retain the last solution and strongest known bound."""
    result = collect_model_result(model, "3_VI", True)
    result["master_solves"] = previous["master_solves"] + 1
    result["solver_runtime"] = previous["solver_runtime"] + model.Runtime
    result["simplex_iterations"] = previous["simplex_iterations"] + model.IterCount
    result["objective_history"] = previous["objective_history"]
    if model.Status == GRB.OPTIMAL:
        result["objective_history"].append(result["objective_value"])
    if previous["dual_bound"] is not None:
        bound = result["dual_bound"]
        result["dual_bound"] = (previous["dual_bound"] if bound is None
                                else max(previous["dual_bound"], bound))
    if result["has_solution"]:
        result["variables"] = variable_values(variables)
    elif previous["has_solution"]:
        for name in ("objective_value", "variables", "has_solution", "solution_type"):
            result[name] = previous[name]
    return result


def _solve_relaxation(model, variables, pool, budget):
    """Solve, separate, and add cuts; always solve again after adding a batch."""
    result = empty_model_result("3_VI", True, model)
    result["objective_history"] = []
    try:
        while True:
            budget.apply_to(model)
            model.optimize()
            result = _collect_lp_solve(model, variables, result)
            if model.Status != GRB.OPTIMAL:
                reason = STATUS_NAMES.get(model.Status, "solver_stopped").lower()
                break
            cuts, reason = pool.select(result["variables"])
            if not cuts:
                break
            budget.check()
            for position, cut in enumerate(cuts, start=len(pool.keys)):
                model.addConstr(_constraint(variables, cut), name=f"subset_VI[{position}]")
            pool.record(cuts)
            # This saved solution predates the new cuts until a solve replaces it.
            result["solution_type"] = "restricted_master"
            model.update()
    except BudgetExpired:
        result.update(status=GRB.TIME_LIMIT, status_name="TIME_LIMIT")
        reason = "time_limit"
    result.update(
        convergence_reason=reason,
        num_constraints=model.NumConstrs, num_linear_constraints=model.NumConstrs,
    )
    if result["status"] != GRB.OPTIMAL or result["solution_type"] == "restricted_master":
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
    pool = None
    try:
        budget.check()
        model, variables = build_formulation_3_VI(
            instance, relax=relax, time_limit=budget.remaining(),
            output_flag=output_flag, solver_seed=solver_seed, threads=threads, env=env,
        )
        pool = CutPool(
            variables, budget, max_cuts, max_cuts_per_round, max_iterations, tolerance,
        )
        if relax:
            result = _solve_relaxation(model, variables, pool, budget)
        else:
            result = _solve_integer(model, variables, pool, budget)
    except BudgetExpired:
        result = empty_model_result("3_VI", relax, model)
        result["convergence_reason"] = "time_limit"
    finally:
        elapsed = budget.elapsed()
        if model is not None:
            model.dispose()
    result.update(
        max_cuts=max_cuts, max_cuts_per_round=max_cuts_per_round,
        max_iterations=max_iterations, cut_tolerance=tolerance, runtime=elapsed,
    )
    if pool is not None:
        result.update(pool.statistics())
    return result
