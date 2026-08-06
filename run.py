import argparse
import gc
import glob
from pathlib import Path

from branch_and_cut import separate_solution
from formulations import solve_instance
from instances import prepare_instance
from utils import load_gurobi_env, save_rows, software_metadata
from validation import (
    enumerate_original_problem,
    validate_integer_result,
    validate_relaxation_bound,
)

DEFAULT_FORMULATIONS = (1, 2, 3, 4)


def _same_value(values, tolerance=1e-6):
    if not values:
        return False
    reference = values[0]
    return all(
        abs(value - reference) <= tolerance * max(1.0, abs(reference))
        for value in values[1:]
    )


def _print_solve_start(instance, repetition, formulation, solve_mode):
    name = instance.get("name", "unnamed")
    print(
        f"Solving instance '{name}' | formulation {formulation} | "
        f"{solve_mode} | repetition {repetition}",
        flush=True,
    )


def _row_from_result(result, instance, repetition, solver_seed, threads):
    cuts_by_family = result.get("cuts_by_family", {})
    metadata = software_metadata()

    return {
        "instance": instance.get("name", "unnamed"),
        "instance_seed": instance.get("seed"),
        "repetition": repetition,
        "formulation": result["formulation"],
        "mode": "relaxation" if result["relax"] else "integer",
        "solver_seed": solver_seed,
        "threads": threads,
        "status": result["status_name"],
        "objective_value": result["objective_value"],
        "dual_bound": result["dual_bound"],
        "gap": result["gap"],
        "runtime": result["runtime"],
        "solver_runtime": result.get("solver_runtime", result["runtime"]),
        "num_variables": result["num_variables"],
        "num_constraints": result["num_constraints"],
        "nodes_explored": result["nodes_explored"],
        "simplex_iterations": result["simplex_iterations"],
        "cuts": result.get("cuts", 0),
        "intruder_cuts": cuts_by_family.get("intruder", 0),
        "feasibility_cuts": cuts_by_family.get("feasibility", 0),
        "optimality_cuts": cuts_by_family.get("optimality", 0),
        "cut_iterations": result.get("cut_iterations", 0),
        "master_solves": result.get("master_solves", 1),
        "lazy_additions": result.get("lazy_additions", 0),
        "separation_time": result.get("separation_time", 0.0),
        "separation_complete": result.get("separation_complete", True),
        "validation_passed": None,
        "original_objective": None,
        **metadata,
    }


def run_comparison(
    instance,
    mode="both",
    formulations=DEFAULT_FORMULATIONS,
    repetition=1,
    time_limit=None,
    output_flag=0,
    max_iterations=100,
    max_cuts=None,
    tolerance=1e-6,
    strict_validation=True,
    solver_seed=0,
    threads=1,
    env=None,
    row_callback=None,
    retain_variables=True,
):
    data = prepare_instance(instance)
    selected_formulations = tuple(dict.fromkeys(int(f) for f in formulations))
    if not selected_formulations or any(f not in DEFAULT_FORMULATIONS for f in selected_formulations):
        raise ValueError("formulations must contain values from 1, 2, 3, and 4")
    if mode not in ("integer", "relaxation", "both"):
        raise ValueError("mode must be integer, relaxation, or both")

    solve_integer = mode in ("integer", "both")
    solve_relaxation = mode in ("relaxation", "both")
    results = {}
    rows = []

    def complete_row(row, result):
        rows.append(row)
        if row_callback is not None:
            row_callback(row)
        if not retain_variables:
            variables = result.get("variables")
            if variables is not None:
                variables.clear()

    for formulation in selected_formulations:
        common_arguments = {
            "time_limit": time_limit,
            "output_flag": output_flag,
            "solver_seed": solver_seed,
            "threads": threads,
            "env": env,
        }

        if solve_integer:
            _print_solve_start(
                data["instance"], repetition, formulation, "integer"
            )
            result = solve_instance(
                data["instance"],
                formulation=formulation,
                relax=False,
                **common_arguments,
            )
            results[formulation, "integer"] = result
            row = _row_from_result(
                result,
                data["instance"],
                repetition,
                solver_seed,
                threads,
            )
            validation = validate_integer_result(
                data["instance"], result, tolerance=tolerance
            )
            row["validation_passed"] = validation["valid"]
            row["original_objective"] = validation["original_objective"]

            if formulation == 4 and result.get("variables"):
                variables = result["variables"]
                remaining_cuts = separate_solution(
                    data,
                    variables["x"],
                    variables["alpha"],
                    variables["phi"],
                    tolerance=tolerance,
                    solver_seed=solver_seed,
                    threads=threads,
                    env=env,
                )
                if remaining_cuts:
                    validation["valid"] = False
                    validation["errors"].append(
                        "The final branch-and-cut solution has violated cuts"
                    )
                    row["validation_passed"] = False

            complete_row(row, result)

            if strict_validation and not validation["valid"]:
                raise AssertionError(
                    "Formulation {} failed validation: {}".format(
                        formulation, validation["errors"]
                    )
                )

        if solve_relaxation:
            _print_solve_start(
                data["instance"], repetition, formulation, "relaxation"
            )
            extra_arguments = {}
            if formulation == 4:
                extra_arguments = {
                    "max_iterations": max_iterations,
                    "max_cuts": max_cuts,
                }
            result = solve_instance(
                data["instance"],
                formulation=formulation,
                relax=True,
                **common_arguments,
                **extra_arguments,
            )
            results[formulation, "relaxation"] = result
            row = _row_from_result(
                result,
                data["instance"],
                repetition,
                solver_seed,
                threads,
            )
            complete_row(row, result)

    oracle = None
    if len(data["edges"]) <= 18:
        oracle = enumerate_original_problem(data["instance"])

    if solve_integer:
        optimal_results = [
            results[formulation, "integer"]
            for formulation in selected_formulations
            if results[formulation, "integer"]["status_name"] == "OPTIMAL"
        ]
        if len(optimal_results) == len(selected_formulations):
            objective_values = [result["objective_value"] for result in optimal_results]
            if strict_validation and not _same_value(objective_values, tolerance):
                raise AssertionError(
                    "The selected integer formulations have different objective values"
                )
            if oracle is not None and oracle["objective_value"] is not None:
                for result in optimal_results:
                    if abs(result["objective_value"] - oracle["objective_value"]) > (
                        tolerance * max(1.0, abs(oracle["objective_value"]))
                    ):
                        raise AssertionError(
                            "Formulation {} differs from the enumeration oracle".format(
                                result["formulation"]
                            )
                        )

    integer_optimum = None
    if oracle is not None:
        integer_optimum = oracle["objective_value"]
    elif solve_integer:
        optimal_values = [
            results[formulation, "integer"]["objective_value"]
            for formulation in selected_formulations
            if results[formulation, "integer"]["status_name"] == "OPTIMAL"
        ]
        if optimal_values and _same_value(optimal_values, tolerance):
            integer_optimum = optimal_values[0]

    if solve_relaxation and integer_optimum is not None:
        for row in rows:
            if row["mode"] != "relaxation":
                continue
            result = results[row["formulation"], "relaxation"]
            validation = validate_relaxation_bound(
                result, integer_optimum, tolerance=tolerance
            )
            row["validation_passed"] = validation["valid"]
            if row_callback is not None:
                row_callback(row)
            if strict_validation and not validation["valid"]:
                raise AssertionError(
                    "Formulation {} returned an invalid relaxation bound".format(
                        row["formulation"]
                    )
                )

    return {"rows": rows, "results": results, "oracle": oracle}


def resolve_instances(specifications):
    resolved = []
    for specification in specifications:
        path = Path(specification)
        if path.is_file():
            matches = [path]
        elif path.is_dir():
            matches = sorted(path.glob("*.json"))
        else:
            matches = sorted(Path(match) for match in glob.glob(specification))

        for match in matches:
            absolute = match.resolve()
            if absolute.is_file() and absolute not in resolved:
                resolved.append(absolute)

    if not resolved:
        raise FileNotFoundError("No JSON instance matched the supplied path(s)")
    return resolved


def run_experiments(
    instances,
    repetitions=1,
    csv_filename=None,
    env=None,
    **comparison_arguments,
):
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")

    rows = []
    comparisons = []
    for instance in instances:
        for repetition in range(1, repetitions + 1):
            recorded_row_ids = set()

            def checkpoint_row(row, recorded_row_ids=recorded_row_ids):
                row_id = id(row)
                if row_id not in recorded_row_ids:
                    rows.append(row)
                    recorded_row_ids.add(row_id)
                if csv_filename is not None:
                    save_rows(rows, csv_filename)

            comparison = run_comparison(
                instance,
                repetition=repetition,
                env=env,
                row_callback=checkpoint_row,
                retain_variables=False,
                **comparison_arguments,
            )

            # Keep this fallback for custom or monkeypatched comparison runners
            # that return rows without invoking the callback.
            for row in comparison["rows"]:
                if id(row) not in recorded_row_ids:
                    checkpoint_row(row)

            # Custom comparison runners may ignore retain_variables.
            for result in comparison["results"].values():
                variables = result.get("variables")
                if variables is not None:
                    variables.clear()
            comparisons.append(comparison)

            # Models are explicitly disposed by the solvers. Collect here as
            # well so unreachable Python-side model and callback cycles do not
            # linger between repetitions or instances.
            gc.collect()

    return {"rows": rows, "comparisons": comparisons}


def print_results(rows):
    header = (
        "{:<18} {:>3} {:>4} {:<10} {:<12} {:>11} "
        "{:>11} {:>9} {:>6}"
    ).format(
        "Instance", "Rep", "Form", "Mode", "Status", "Objective",
        "Dual bound", "Time", "Cuts"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        objective = "-" if row["objective_value"] is None else "{:.6f}".format(row["objective_value"])
        dual_bound = "-" if row["dual_bound"] is None else "{:.6f}".format(row["dual_bound"])
        print(
            "{:<18.18} {:>3} {:>4} {:<10} {:<12} {:>11} "
            "{:>11} {:>9.4f} {:>6}".format(
                row["instance"], row["repetition"], row["formulation"],
                row["mode"], row["status"], objective, dual_bound,
                row["runtime"], row["cuts"]
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare SNI formulations on one or more JSON instances"
    )
    parser.add_argument(
        "instances",
        nargs="+",
        help="JSON file, directory, or quoted glob such as instances/*.json",
    )
    parser.add_argument(
        "--formulations",
        nargs="+",
        type=int,
        choices=DEFAULT_FORMULATIONS,
        default=list(DEFAULT_FORMULATIONS),
    )
    parser.add_argument(
        "--mode",
        choices=["integer", "relaxation", "both"],
        default="both",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--csv", default="results/results.csv")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-cuts", type=int, default=None)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--allow-validation-failures", action="store_true")
    args = parser.parse_args()

    instance_paths = resolve_instances(args.instances)
    with load_gurobi_env() as env:
        experiment = run_experiments(
            instance_paths,
            repetitions=args.repetitions,
            csv_filename=args.csv,
            mode=args.mode,
            formulations=args.formulations,
            time_limit=args.time_limit,
            output_flag=int(args.output),
            max_iterations=args.max_iterations,
            max_cuts=args.max_cuts,
            strict_validation=not args.allow_validation_failures,
            solver_seed=args.solver_seed,
            threads=args.threads,
            env=env,
        )
    print_results(experiment["rows"])
    print(f"Results saved in {args.csv}")


if __name__ == "__main__":
    main()
