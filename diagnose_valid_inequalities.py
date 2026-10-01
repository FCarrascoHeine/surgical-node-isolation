"""Explain absent VI gains using checkpoint integrality and budget experiments."""

import argparse
import csv
import gc
from pathlib import Path

from gurobipy import GRB, quicksum

from create_complex_instances import build_minimum_checkpoint_model
from formulation_3_VI import separate_valid_inequalities
from formulations import solve_instance
from instances import load_instance, prepare_instance
from run import resolve_instances
from utils import load_gurobi_env
from validation import evaluate_allocation


def save_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_diagnostics(instance, env):
    """Maximize distance from a binary certificate over the budget-feasible LP.

    For binary reference r, sum(r*(1-x) + (1-r)*x) is linear and nonnegative.
    An optimal maximum of zero proves that every feasible checkpoint x equals r.
    This model has only checkpoint/potential variables, without journeyer flows.
    """
    chosen = {tuple(edge) for edge in instance["known_feasible_checkpoints"]}
    model, x, costs = build_minimum_checkpoint_model(
        instance, budget_model="disaggregated", time_limit=300, env=env,
    )
    try:
        for variable in x.values():
            variable.VType = GRB.CONTINUOUS
        model.Params.FeasibilityTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        model.optimize()
        result = {"minimum_cost_status": model.Status,
                  "minimum_cost_lp": model.ObjVal if model.Status == GRB.OPTIMAL else None}
        model.addConstr(quicksum(costs[e] * x[e] for e in x) <= instance["budget"])
        model.setObjective(quicksum(1 - x[e] if e in chosen else x[e] for e in x), GRB.MAXIMIZE)
        model.optimize()
        distance = model.ObjVal if model.Status == GRB.OPTIMAL else None
        result.update(max_distance_status=model.Status, max_certificate_distance=distance,
                      unique_checkpoint_vector=(distance <= 1e-6) if distance is not None else None)
        return result
    finally:
        model.dispose()


def solution_diagnostics(instance, result):
    if result["status"] != GRB.OPTIMAL:
        return {"fractional_x": None, "fractional_z": None, "violated_edges": None,
                "max_subset_violation": None, "inspection_objective": None}
    data, values = prepare_instance(instance), result["variables"]
    x, z, beta = (values[name] for name in ("x", "z", "beta"))
    violation = max(0.0, max(sum(max(0.0, z[j, e] + x[e] - 1)
                                   for j in data["journeyers"]) - beta[e] for e in data["edges"]))
    return {
        "fractional_x": sum(abs(v - round(v)) > 1e-6 for v in x.values()),
        "fractional_z": sum(abs(v - round(v)) > 1e-6 for v in z.values()),
        "violated_edges": len(separate_valid_inequalities(data["edges"], data["journeyers"], x, z, beta)),
        "max_subset_violation": violation,
        "inspection_objective": sum(data["inspection_time"][e] * beta[e] for e in data["edges"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", help="c-instance paths or quoted patterns")
    parser.add_argument("--output-dir", default="results/lp_larger_20260925")
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="Check uniqueness without building journeyer-flow LPs")
    parser.add_argument("--budget-controls", action="store_true",
                        help="Also compare all three LPs at +20%%/+50%% budgets on 100-node inputs")
    args = parser.parse_args(argv)
    if args.checkpoint_only and args.budget_controls:
        parser.error("--checkpoint-only cannot be combined with --budget-controls")
    paths = resolve_instances(args.instances)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics, controls = [], []
    with load_gurobi_env() as env:
        env.setParam("SoftMemLimit", 2)
        for path in paths:
            instance = load_instance(path)
            print(f"Diagnosing {instance['name']}", flush=True)
            checkpoint = checkpoint_diagnostics(instance, env)
            if args.checkpoint_only:
                diagnostics.append({"instance": instance["name"], "budget": instance["budget"], **checkpoint})
                save_rows(output / "c_diagnostics.csv", diagnostics)
                print(checkpoint, flush=True)
                continue
            baseline = solve_instance(instance, 3, relax=True, time_limit=300, env=env)
            data = prepare_instance(instance)
            chosen = {tuple(e) for e in instance["known_feasible_checkpoints"]}
            certificate = evaluate_allocation(instance, {e: float(e in chosen) for e in data["edges"]})
            assert certificate["valid"], certificate["errors"]
            row = {"instance": instance["name"], "budget": instance["budget"],
                   "aggregated_status": baseline["status_name"],
                   "aggregated_lp": baseline["objective_value"],
                   "certificate_objective": certificate["objective_value"],
                   **checkpoint, **solution_diagnostics(instance, baseline)}
            diagnostics.append(row)
            save_rows(output / "c_diagnostics.csv", diagnostics)
            del baseline
            gc.collect()
            if not args.budget_controls or len(instance["nodes"]) != 100:
                continue
            for factor in (1.2, 1.5):
                changed = dict(instance, budget=int(factor * instance["budget"]))
                record = {"instance": instance["name"], "budget_factor": factor,
                          "original_budget": instance["budget"], "budget": changed["budget"]}
                for formulation in (3, 2, "3_VI"):
                    result = solve_instance(changed, formulation, relax=True, time_limit=300, env=env)
                    record[f"status_{formulation}"] = result["status_name"]
                    record[f"lp_{formulation}"] = result["objective_value"]
                    if formulation == 3:
                        record.update(solution_diagnostics(changed, result))
                    if formulation == "3_VI":
                        record.update(cuts=result["cuts"], separation_complete=result["separation_complete"])
                    del result
                    gc.collect()
                if all(record[f"status_{f}"] == "OPTIMAL" for f in (3, 2, "3_VI")):
                    record["vi_improvement_percent"] = 100 * (record["lp_3_VI"] - record["lp_3"]) / abs(record["lp_3"])
                    record["disaggregated_improvement_percent"] = 100 * (record["lp_2"] - record["lp_3"]) / abs(record["lp_3"])
                else:
                    record.update(vi_improvement_percent=None, disaggregated_improvement_percent=None)
                controls.append(record)
                save_rows(output / "c_budget_controls.csv", controls)
                print(f"  Budget x{factor}: VI improvement {record['vi_improvement_percent']}%", flush=True)


if __name__ == "__main__":
    main()
