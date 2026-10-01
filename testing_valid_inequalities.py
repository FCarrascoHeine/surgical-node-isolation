"""Compare LP relaxations: aggregated (3), disaggregated (2), and aggregated + VI."""

import argparse
from pathlib import Path

from experiment_supervisor import run_supervised_experiments
from formulation_3_VI import (
    DEFAULT_MAX_CUTS, DEFAULT_MAX_CUTS_PER_ROUND, DEFAULT_MAX_ITERATIONS,
    validate_cut_limits,
)
from memory_limits import resolve_memory_limit
from run import print_results, resolve_instances
from time_budget import TimeBudget

LP_FORMULATIONS = (3, 2, "3_VI")
DEFAULT_INSTANCE = Path(__file__).resolve().parent / "instances/small_instance.json"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("instances", nargs="*", default=[str(DEFAULT_INSTANCE)],
                        help="JSON files, directories, or quoted globs")
    parser.add_argument("--csv", default="results/valid_inequalities.csv")
    parser.add_argument("--max-cuts", type=int, default=DEFAULT_MAX_CUTS,
                        help="Total VI cut limit; zero disables cuts")
    parser.add_argument("--max-cuts-per-round", type=int, default=DEFAULT_MAX_CUTS_PER_ROUND)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Elapsed seconds per formulation, including build and cut rounds")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit-gb", default="none")
    parser.add_argument("--output", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_cut_limits(args.max_cuts, args.max_cuts_per_round,
                            args.max_iterations, args.tolerance)
        TimeBudget(args.time_limit)
        memory_policy = resolve_memory_limit(args.memory_limit_gb)
        paths = resolve_instances(args.instances)
        if args.repetitions < 1:
            raise ValueError("repetitions must be at least one")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    experiment = run_supervised_experiments(
        paths, formulations=LP_FORMULATIONS, mode="relaxation", heuristics=(),
        csv_filename=args.csv, repetitions=args.repetitions,
        max_cuts=args.max_cuts, max_cuts_per_round=args.max_cuts_per_round,
        max_iterations=args.max_iterations, tolerance=args.tolerance,
        time_limit=args.time_limit, solver_seed=args.solver_seed, threads=args.threads,
        memory_policy=memory_policy, output_flag=int(args.output),
    )
    print_results(experiment["rows"])
    print(f"Results saved in {args.csv}")
    return experiment


if __name__ == "__main__":
    main()
