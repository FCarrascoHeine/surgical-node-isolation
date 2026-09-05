import argparse
import gc
import glob
from pathlib import Path

from gurobipy import GRB

from branch_and_cut import separate_solution
from formulations import solve_instance
from heuristics import (
    HEURISTIC_NAMES,
    solve_single_intruder_heuristic,
    solve_standard_heuristic,
)
from instances import prepare_instance
from time_budget import TimeBudget
from utils import MemoryLimitReached, save_rows, software_metadata
from validation import (
    enumerate_original_problem,
    evaluate_allocation,
    validate_integer_result,
    validate_relaxation_bound,
)

DEFAULT_FORMULATIONS = (1, 2, 3, 4)
DEFAULT_HEURISTICS = ()


def _same_value(values, tolerance=1e-6):
    if not values:
        return False
    reference = values[0]
    return all(
        abs(value - reference) <= tolerance * max(1.0, abs(reference))
        for value in values[1:]
    )


def _print_solve_start(instance, repetition, method, solve_mode):
    name = instance.get("name", "unnamed")
    print(
        f"Solving instance '{name}' | method {method} | "
        f"{solve_mode} | repetition {repetition}",
        flush=True,
    )


def _row_from_result(result, instance, repetition, solver_seed, threads):
    cuts_by_family = result.get("cuts_by_family", {})
    metadata = software_metadata()
    formulation = result.get("formulation")
    method = result.get("method")
    if method is None and formulation is not None:
        method = f"f{formulation}"
    heuristic = method in HEURISTIC_NAMES

    return {
        "instance": instance.get("name", "unnamed"),
        "instance_seed": instance.get("seed"),
        "repetition": repetition,
        "method": method,
        "method_type": "heuristic" if heuristic else "formulation",
        "formulation": formulation,
        "mode": (
            "heuristic"
            if heuristic
            else ("relaxation" if result.get("relax") else "integer")
        ),
        "solver_seed": solver_seed,
        "threads": threads,
        "status": result["status_name"],
        "memory_limit_gb": None,
        "memory_limit_source": None,
        "error_type": None,
        "error_message": None,
        "error_code": None,
        "error_phase": None,
        "worker_exit_code": None,
        "has_solution": result.get("has_solution", result.get("objective_value") is not None),
        "solution_type": result.get(
            "solution_type",
            ("relaxation" if result.get("relax") else "integer")
            if result.get("objective_value") is not None else "none",
        ),
        "objective_value": result.get("objective_value"),
        "dual_bound": result.get("dual_bound"),
        "gap": result.get("gap"),
        "reference_objective": None,
        "reference_gap": None,
        "runtime": result["runtime"],
        "solver_runtime": result.get("solver_runtime", result["runtime"]),
        "num_variables": result.get("num_variables", 0),
        "num_constraints": result.get("num_constraints", 0),
        "nodes_explored": result.get("nodes_explored", 0.0),
        "simplex_iterations": result.get("simplex_iterations", 0.0),
        "cuts": result.get("cuts", 0),
        "intruder_cuts": cuts_by_family.get("intruder", 0),
        "feasibility_cuts": cuts_by_family.get("feasibility", 0),
        "optimality_cuts": cuts_by_family.get("optimality", 0),
        "cut_iterations": result.get("cut_iterations", 0),
        "master_solves": result.get("master_solves", 1),
        "lazy_additions": result.get("lazy_additions", 0),
        "separation_time": result.get("separation_time", 0.0),
        "separation_complete": result.get("separation_complete", True),
        "checkpoint_cost": result.get("checkpoint_cost"),
        "heuristic_iterations": result.get("heuristic_iterations"),
        "convergence_reason": result.get("convergence_reason"),
        "auxiliary_solves": result.get("auxiliary_solves"),
        "minimum_cut_solves": result.get("minimum_cut_solves"),
        "best_iteration": result.get("best_iteration"),
        "terminal_objective": result.get("terminal_objective"),
        "subproblems_optimal": result.get("subproblems_optimal"),
        "validation_passed": None,
        "original_objective": None,
        **metadata,
    }


def _not_applicable_heuristic_result(method):
    return {
        "method": method,
        "formulation": None,
        "relax": False,
        "status_name": "NOT_APPLICABLE",
        "has_solution": False,
        "solution_type": "none",
        "objective_value": None,
        "dual_bound": None,
        "gap": None,
        "runtime": 0.0,
        "solver_runtime": 0.0,
        "num_variables": 0,
        "num_constraints": 0,
        "nodes_explored": 0.0,
        "simplex_iterations": 0.0,
        "variables": {},
        "checkpoint_cost": None,
        "heuristic_iterations": 0,
        "convergence_reason": "requires_exactly_one_intruder",
        "auxiliary_solves": 0,
        "minimum_cut_solves": 0,
        "best_iteration": None,
        "terminal_objective": None,
        "subproblems_optimal": True,
    }


def run_comparison(
    instance,
    mode="both",
    formulations=DEFAULT_FORMULATIONS,
    heuristics=DEFAULT_HEURISTICS,
    repetition=1,
    time_limit=None,
    output_flag=0,
    max_iterations=100,
    max_cuts=None,
    heuristic_max_iterations=100,
    binary_search_tolerance=1e-4,
    heuristic_return_best=True,
    tolerance=1e-6,
    strict_validation=True,
    solver_seed=0,
    threads=1,
    env=None,
    row_callback=None,
    retain_variables=True,
    phase_callback=None,
):
    # Reject invalid budgets even if a selected method is not applicable.
    TimeBudget(time_limit)
    phase = phase_callback or (lambda _name: None)
    phase("preparation")
    data = prepare_instance(instance)
    selected_formulations = tuple(dict.fromkeys(int(f) for f in formulations))
    selected_heuristics = tuple(dict.fromkeys(str(h).lower() for h in heuristics))
    if any(f not in DEFAULT_FORMULATIONS for f in selected_formulations):
        raise ValueError("formulations must contain values from 1, 2, 3, and 4")
    if any(h not in HEURISTIC_NAMES for h in selected_heuristics):
        raise ValueError("heuristics must contain ah and/or ash")
    if not selected_formulations and not selected_heuristics:
        raise ValueError("At least one formulation or heuristic must be selected")
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
                data["instance"], repetition, f"f{formulation}", "integer"
            )
            phase("build_solve")
            result = solve_instance(
                data["instance"],
                formulation=formulation,
                relax=False,
                **common_arguments,
            )
            results[formulation, "integer"] = result
            phase("validation")
            row = _row_from_result(
                result,
                data["instance"],
                repetition,
                solver_seed,
                threads,
            )
            validation = None
            if row["has_solution"]:
                validation = validate_integer_result(
                    data["instance"], result, tolerance=tolerance
                )
                row["validation_passed"] = validation["valid"]
                row["original_objective"] = validation["original_objective"]

            if formulation == 4 and validation is not None and result.get("variables"):
                variables = result["variables"]
                try:
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
                except MemoryLimitReached:
                    # The returned integer snapshot already passed full lazy
                    # separation. An interrupted redundant check proves nothing new.
                    remaining_cuts = []
                    result.update(status=GRB.MEM_LIMIT, status_name="MEM_LIMIT",
                                  separation_complete=False,
                                  convergence_reason="validation_memory_limit")
                    row.update(status="MEM_LIMIT", separation_complete=False,
                               convergence_reason="validation_memory_limit",
                               validation_passed=None)
                if remaining_cuts:
                    validation["valid"] = False
                    validation["errors"].append(
                        "The final branch-and-cut solution has violated cuts"
                    )
                    row["validation_passed"] = False

            complete_row(row, result)

            if strict_validation and validation is not None and not validation["valid"]:
                raise AssertionError(
                    "Formulation {} failed validation: {}".format(
                        formulation, validation["errors"]
                    )
                )

        if solve_relaxation:
            _print_solve_start(
                data["instance"], repetition, f"f{formulation}", "relaxation"
            )
            extra_arguments = {}
            if formulation == 4:
                extra_arguments = {
                    "max_iterations": max_iterations,
                    "max_cuts": max_cuts,
                }
            phase("build_solve")
            result = solve_instance(
                data["instance"],
                formulation=formulation,
                relax=True,
                **common_arguments,
                **extra_arguments,
            )
            results[formulation, "relaxation"] = result
            phase("validation")
            row = _row_from_result(
                result,
                data["instance"],
                repetition,
                solver_seed,
                threads,
            )
            complete_row(row, result)

    for heuristic in selected_heuristics:
        phase("build_solve")
        _print_solve_start(data["instance"], repetition, heuristic, "heuristic")
        if heuristic == "ash" and len(data["intruders"]) != 1:
            result = _not_applicable_heuristic_result(heuristic)
        elif heuristic == "ah":
            result = solve_standard_heuristic(
                data["instance"],
                max_iterations=heuristic_max_iterations,
                time_limit=time_limit,
                tolerance=tolerance,
                output_flag=output_flag,
                solver_seed=solver_seed,
                threads=threads,
                env=env,
                return_best=heuristic_return_best,
            )
        else:
            result = solve_single_intruder_heuristic(
                data["instance"],
                max_iterations=heuristic_max_iterations,
                binary_search_tolerance=binary_search_tolerance,
                time_limit=time_limit,
                tolerance=tolerance,
                return_best=heuristic_return_best,
            )

        phase("validation")
        results[heuristic, "heuristic"] = result
        row = _row_from_result(
            result,
            data["instance"],
            repetition,
            solver_seed,
            threads,
        )
        validation = None
        if result.get("variables", {}).get("x"):
            validation = evaluate_allocation(
                data["instance"],
                result["variables"]["x"],
                tolerance=tolerance,
            )
            row["validation_passed"] = validation["valid"]
            row["original_objective"] = validation["objective_value"]

        complete_row(row, result)
        if strict_validation and validation is not None and not validation["valid"]:
            raise AssertionError(
                "Heuristic {} failed validation: {}".format(
                    heuristic, validation["errors"]
                )
            )

    phase("comparison")
    oracle = None
    if len(data["edges"]) <= 18:
        oracle = enumerate_original_problem(data["instance"])

    return finalize_comparison(
        rows, results, oracle, selected_formulations, solve_integer,
        solve_relaxation, strict_validation, tolerance, row_callback,
    )


def finalize_comparison(
    rows, results, oracle, selected_formulations, solve_integer, solve_relaxation,
    strict_validation=True, tolerance=1e-6, row_callback=None,
):
    """Compare compact, already validated summaries; no model/instance required."""
    if solve_integer and selected_formulations:
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
    elif solve_integer and selected_formulations:
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
            if result.get("dual_bound") is None and result["status_name"] != "OPTIMAL":
                # An interrupted solve without a bound is not a validation failure.
                continue
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

    if integer_optimum is not None:
        for row in rows:
            if row["mode"] not in ("integer", "heuristic"):
                continue
            if row["objective_value"] is None:
                continue
            row["reference_objective"] = integer_optimum
            if abs(integer_optimum) <= tolerance:
                row["reference_gap"] = (
                    0.0
                    if abs(row["objective_value"] - integer_optimum) <= tolerance
                    else None
                )
            else:
                row["reference_gap"] = (
                    row["objective_value"] - integer_optimum
                ) / abs(integer_optimum)
            if row_callback is not None:
                row_callback(row)

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
        "{:<18} {:>3} {:>6} {:<10} {:<15} {:<17} {:>11} "
        "{:>11} {:>9} {:>6}"
    ).format(
        "Instance", "Rep", "Method", "Mode", "Status", "Solution", "Objective",
        "Dual bound", "Time", "Cuts"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        objective = (
            "-"
            if row["objective_value"] is None
            else "{:.6f}".format(row["objective_value"])
        )
        dual_bound = "-" if row["dual_bound"] is None else "{:.6f}".format(row["dual_bound"])
        cuts = "-" if row["cuts"] is None else row["cuts"]
        print(
            "{:<18.18} {:>3} {:>6} {:<10} {:<15} {:<17} {:>11} "
            "{:>11} {:>9.4f} {:>6}".format(
                row["instance"], row["repetition"], row["method"],
                row["mode"], row["status"], row.get("solution_type", "none"), objective, dual_bound,
                row["runtime"], cuts
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SNI formulations and heuristics on one or more JSON instances"
        )
    )
    parser.add_argument(
        "instances",
        nargs="+",
        help="JSON file, directory, or quoted glob such as instances/*.json",
    )
    parser.add_argument(
        "--formulations",
        nargs="*",
        type=int,
        choices=DEFAULT_FORMULATIONS,
        default=list(DEFAULT_FORMULATIONS),
    )
    parser.add_argument(
        "--heuristics",
        nargs="+",
        choices=HEURISTIC_NAMES,
        default=list(DEFAULT_HEURISTICS),
        help="Run the general heuristic (ah), single-intruder heuristic (ash), or both",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Run all four formulations and both heuristics",
    )
    parser.add_argument(
        "--mode",
        choices=["integer", "relaxation", "both"],
        default="both",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--csv", default="results/results.csv")
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help=("Elapsed seconds per method invocation, including preparation and "
              "model construction; checked between operations, so overruns are possible"),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum cut rounds for the formulation 4 relaxation",
    )
    parser.add_argument(
        "--memory-limit-gb", default="auto", metavar="GB|auto|none",
        help=("Gurobi soft memory limit in decimal GB; auto reserves headroom "
              "for Python and the OS, none disables the soft limit"),
    )
    parser.add_argument("--max-cuts", type=int, default=None)
    parser.add_argument("--heuristic-max-iterations", type=int, default=100)
    parser.add_argument("--binary-search-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--return-terminal-heuristic",
        action="store_true",
        help="Return the terminal heuristic candidate; timeouts always return the best available",
    )
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--allow-validation-failures", action="store_true")
    args = parser.parse_args()

    instance_paths = resolve_instances(args.instances)
    formulations = (
        list(DEFAULT_FORMULATIONS) if args.all_methods else args.formulations
    )
    heuristics = list(HEURISTIC_NAMES) if args.all_methods else args.heuristics
    from experiment_supervisor import run_supervised_experiments
    from memory_limits import resolve_memory_limit

    try:
        memory_policy = resolve_memory_limit(args.memory_limit_gb)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    experiment = run_supervised_experiments(
        instance_paths,
        repetitions=args.repetitions,
        csv_filename=args.csv,
        mode=args.mode,
        formulations=formulations,
        heuristics=heuristics,
        time_limit=args.time_limit,
        output_flag=int(args.output),
        max_iterations=args.max_iterations,
        max_cuts=args.max_cuts,
        heuristic_max_iterations=args.heuristic_max_iterations,
        binary_search_tolerance=args.binary_search_tolerance,
        heuristic_return_best=not args.return_terminal_heuristic,
        strict_validation=not args.allow_validation_failures,
        solver_seed=args.solver_seed,
        threads=args.threads,
        memory_policy=memory_policy,
    )
    print_results(experiment["rows"])
    print(f"Results saved in {args.csv}")


if __name__ == "__main__":
    main()
