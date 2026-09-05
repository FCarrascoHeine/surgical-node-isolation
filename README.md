# Surgical Node Isolation formulations

This repository compares four Gurobi formulations and the paper's general and
single-intruder heuristics for the safety-check allocation problem. It is organized
as a small computational-research project with one experiment runner, reproducible
JSON instances, and a compact correctness suite.

## Setup

Python 3.11 or newer and a valid Gurobi license are required.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The code first looks for `gurobi_secrets.toml` in the repository root. For a WLS
license, copy `gurobi_secrets.example.toml` and fill in the credentials. If that
file is absent, Gurobi's normal local or network-license discovery is used.

## Run an experiment

Compare all four integer formulations and their relaxations on one instance:

```bash
python run.py instances/small_instance.json
```

Run every JSON instance in the directory three times:

```bash
python run.py instances --repetitions 3 --csv results/experiment.csv
```

Run selected formulations or modes:

```bash
python run.py "instances/*.json" --formulations 2 3 4 --mode integer
```

Compare every exact formulation and both heuristics in one experiment:

```bash
python run.py instances/small_instance.json --all-methods
```

Use `python run.py --help` for time limits, branch-and-cut limits, solver output,
seed, thread count, and validation options. The default seed is 0 and the default
thread count is 1 so repeated runs are comparable.

## Detailed command reference

The general command is:

```bash
python run.py INSTANCE [INSTANCE ...] [OPTIONS]
```

An `INSTANCE` argument can be a JSON file, a directory, or a quoted file pattern.
When a directory is supplied, every `.json` file directly inside that directory
is included. Multiple inputs can be supplied in the same command; duplicate files
are solved only once.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `INSTANCE [INSTANCE ...]` | Required | One or more JSON files, directories, or quoted patterns such as `"instances/*.json"`. Directory searches are not recursive. |
| `--csv PATH` | `results/results.csv` | Destination of the combined result table. Parent directories are created automatically. The file is atomically checkpointed after every completed result row, so completed formulations survive a later failure in the same instance. An existing file at this path is overwritten, not appended to. |
| `--formulations {1,2,3,4} [...]` | `1 2 3 4` | Formulations to solve. For example, `--formulations 2 4` runs only formulations 2 and 4. |
| `--heuristics {ah,ash} [...]` | None | Heuristics to run. `ah` is the general heuristic and `ash` is the single-intruder minimum-cut heuristic. |
| `--all-methods` | Disabled | Runs formulations 1--4 and both heuristics. `ash` is recorded as `NOT_APPLICABLE` on instances that do not contain exactly one intruder. |
| `--mode {integer,relaxation,both}` | `both` | Runs the integer models, the continuous relaxations, or both. |
| `--repetitions N` | `1` | Solves every selected instance/formulation/mode combination `N` times. Repetition numbers are recorded in the CSV. |
| `--time-limit SECONDS` | No limit | Shared elapsed-time budget for each instance/method/mode/repetition, including method preparation, model construction, optimization and separation. Checked between operations; an operation may finish after the deadline. Zero skips the search; negative values and NaN are rejected. |
| `--memory-limit-gb GB\|auto\|none` | `auto` | Gurobi soft memory allowance in decimal GB. Auto uses the smaller of 50% of total physical RAM and 60% of available physical RAM, measured once at batch startup. A positive number overrides it; `none` disables it. The effective allowance and its source are recorded in every CSV row. |
| `--solver-seed N` | `0` | Sets Gurobi's random seed. Keep this fixed when comparing formulations; vary it deliberately when studying solver variability. |
| `--threads N` | `1` | Sets the number of Gurobi threads. Using one thread favors repeatability; larger values may reduce runtime. Gurobi interprets `0` as its automatic setting. |
| `--max-iterations N` | `100` | Maximum number of cut-addition rounds for the formulation 4 relaxation. It has no effect on formulations 1--3 or on the formulation 4 integer callback. |
| `--max-cuts N` | No limit | Maximum total number of cuts added while solving the formulation 4 relaxation. It has no effect on the other solves. |
| `--heuristic-max-iterations N` | `100` | Maximum number of outer iterations for either heuristic. |
| `--binary-search-tolerance VALUE` | `1e-4` | Precision used by the single-intruder heuristic's capacity-weight binary search. |
| `--return-terminal-heuristic` | Disabled | Returns the terminal heuristic candidate instead of the best evaluated true-objective candidate. Timeouts always return the best available candidate. Iteration history is retained either way. |
| `--output` | Disabled | Displays Gurobi's solver log while the experiment runs. |
| `--allow-validation-failures` | Disabled | Keeps numerical results with a failed-validation flag. By default the supervised CLI records `VALIDATION_FAILED` and continues; direct Python comparison calls retain their strict exception behavior. |

Without heuristics, the number of result rows is:

```text
instances * formulations * modes * repetitions
```

Here, `both` counts as two modes. For example, three instances with all four
formulations, both modes, and five repetitions produce `3 * 4 * 2 * 5 = 120`
rows in one CSV file.

Each selected heuristic contributes one additional row per instance and repetition.

### Experiment examples

Solve every JSON file in `instances/` and save one combined comparison:

```bash
python run.py instances --csv results/all_instances.csv
```

Run five repetitions for all instances, formulations, and modes:

```bash
python run.py instances --repetitions 5 --csv results/all_instances_5_repetitions.csv
```

Compare only formulations 2 and 4 as integer models with a one-hour limit per
solve:

```bash
python run.py instances --formulations 2 4 --mode integer --time-limit 3600 --csv results/integer_f2_f4.csv
```

Compare formulation 2 and both heuristics without running relaxations:

```bash
python run.py instances --formulations 2 --heuristics ah ash --mode integer --csv results/f2_heuristics.csv
```

Study only the formulation 4 relaxation and stop after at most 50 cut rounds or
500 generated cuts:

```bash
python run.py instances --formulations 4 --mode relaxation --max-iterations 50 --max-cuts 500 --csv results/f4_relaxation.csv
```

Enable the Gurobi log for a single diagnostic run without changing the default
result location:

```bash
python run.py instances/small_instance.json --formulations 4 --mode integer --output
```

Use descriptive `--csv` filenames for paper experiments. If `--csv` is omitted,
the next run will overwrite the default `results/results.csv` file.

Each CSV row records the instance, method, mode, repetition, solver settings,
objective, bound, solver gap, runtimes, model size, cut or heuristic statistics,
validation status, and Python/Gurobi versions. `reference_gap` compares a feasible
integer or heuristic objective to a proven small-instance oracle or common optimal
formulation result; it is distinct from Gurobi's `gap`. `runtime` measures the
method's elapsed time from before its own input preparation and model construction
until its algorithm stops. Independent runner validation, shared batch setup,
result formatting and CSV writing are excluded. `solver_runtime` is a separate
diagnostic: Gurobi time for the compact solve, AH's auxiliary solves, or formulation
4's master solves. For integer formulation 4 this already includes callback time;
do not add `separation_time` to it. ASH uses no Gurobi solves.

The existing Python `time_limit` argument to the solve functions has the same
meaning as `--time-limit`. A single budget is shared across all iterations and
auxiliary solves, with each Gurobi allowance refreshed immediately before
optimization. Checks happen at safe boundaries, including after model building
and between separation operations. There is no forced interruption of Python
graph algorithms or model construction. Consequently an uninterrupted operation
can overrun; if that operation also completes the method, its normal completion
status is retained. Omitting the limit (or using positive infinity) imposes no
method deadline. Iteration and cut limits still apply independently.

`TIME_LIMIT` means the budget stopped unfinished work. `has_solution=False` and
`solution_type=none` mean that no candidate was returned; objective and gap are
empty, and unavailable bounds stay empty. This is not an infeasibility result
or a validation failure. If a solution is available, the method retains its best
completed candidate, even when terminal heuristic output was requested. ASH can
also return its feasible minimum-cost fallback or the current feasible
binary-search cut. Integer formulation 4 returns only candidates whose lazy-cut
separation completed successfully, never an unchecked master incumbent.

`solution_type=integer` identifies a feasible integer candidate, and `relaxation`
a candidate of the full continuous relaxation. An unfinished formulation 4
relaxation may instead return `restricted_master`: a snapshot from a master with
only some cuts present, not a feasible integer solution or a certified feasible
point of the full relaxation. Its `separation_complete` is false and `gap` is
empty; `dual_bound` retains the strongest established lower bound even if a later
master solve is interrupted. The console's `Solution` column shows this scope.
An interrupted LP's primal objective alone is never treated as a lower bound.

A feasible nonoptimal integer incumbent can use journeyer routes that are longer
than the shortest routes for its allocation. In that case `objective_value`
retains the model objective and `original_objective` reports the independently
evaluated allocation cost; the latter may be smaller. They must agree at optimality.

During a batch, detailed per-variable solution mappings are used for validation and
then released after each solve. The returned experiment object and CSV retain the
compact solve summaries and result rows.

### Memory limits and recovery

The command-line runner uses one persistent worker subprocess. It reuses a single
Gurobi environment for sequential methods, modes, instances, and repetitions. The
parent owns the schedule and atomically checkpoints the CSV after each result.
Model construction, optimization, graph algorithms, and individual solution
validation run in the worker. Only compact rows and an oracle objective are sent
back; variable mappings, heuristic histories, and enumeration allocations are not
retained by the supervisor. Cross-method objective comparisons and reference gaps
are still computed for each instance/repetition.

Every instance is attempted, with no model-size cutoff. For example:

```bash
python run.py instances --formulations 2 --mode integer --time-limit 300
python run.py instances --all-methods --memory-limit-gb 8 --csv results/memory8.csv
```

`SoftMemLimit` is inherited by every Gurobi model in the worker, including AH's
auxiliary models and formulation 4's separation duals. It counts Gurobi allocations
across models in that environment and can overshoot between safe checkpoints.
The allowance does **not** cap total process memory: Python dictionaries, graph
algorithms (including ASH), other programs, and container/OS resource restrictions
are outside that measurement. Automatic detection currently supports Windows and
Linux, with a `sysconf` fallback on other platforms; specify an explicit allowance
if detection fails. The automatic formula is a conservative heuristic, not a
guarantee against system-wide memory exhaustion.

- `MEM_LIMIT`: Gurobi or a separation subproblem stopped gracefully. Available
  justified solutions/bounds are retained. AH returns its best validated candidate
  even with `--return-terminal-heuristic`. An interrupted F4 callback never accepts
  an unchecked incumbent, and an unfinished F4 relaxation retains its
  `restricted_master` scope without claiming full relaxation feasibility.
- `OUT_OF_MEMORY`: a Python or Gurobi OOM exception was reported. The affected
  solve gets an empty solution row and the worker is replaced.
- `ERROR` / `VALIDATION_FAILED`: the solve or validation raised an exception.
  The failure is recorded and subsequent scheduled solves continue.
- `PROCESS_FAILED`: the worker exited without reporting a result. Its exit code
  and last reported phase are saved. OOM is not assumed without evidence.

Failure rows include `error_type`, `error_message`, `error_code`, `error_phase`, and
`worker_exit_code`. Unavailable counts, objective, bound, and gap are empty rather
than invented zeros. A worker crash loses any incumbent it had not reported.
For exception/crash rows, `runtime` is the parent's observed duration of the failed
attempt, including startup/validation if applicable; license retry waiting is
excluded. Successful-result runtime keeps the existing method-time definition.
These failure statuses do not establish infeasibility.

After an exception or crash, the old worker exits before a replacement is used,
releasing its Python/native allocations. Successful tasks explicitly dispose their
models and run garbage collection while reusing the environment. A graceful
`MEM_LIMIT` does not require a restart. WLS session-limit errors during environment
startup retry the **same** pending solve twice, waiting 310 seconds between attempts
(with progress messages). Persistent license failures are recorded as
`LICENSE_ERROR`; they do not stop later methods, including graph-only ASH. Other
license errors are recorded without repeated retries. Actual WLS token expiry or
usage by other applications may require longer than one retry interval. Ctrl+C and
CSV writing failures stop the runner and close its worker; they are not silently
converted into instance failures.

Python callers can opt into the same behavior with
`experiment_supervisor.run_supervised_experiments(...)` under an
`if __name__ == "__main__":` guard (required for spawned processes). It accepts
`memory_limit_gb="auto"` or a positive GB value, and returns `{"rows": ...}`.
The existing `run_comparison`, `run_experiments`, and direct solver APIs remain
in-process with their existing return structures and exception behavior. Pass an
environment with `SoftMemLimit` configured to apply a memory allowance there.

See Gurobi's [memory-limit parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameter:SoftMemLimit)
and [WLS session guidance](https://support.gurobi.com/hc/en-us/articles/34567582787345-How-do-I-resolve-the-error-Too-many-sessions).

## Generate instances

Generate and validate the complete 45-instance rectangular-grid collection
without writing files:

```bash
python generate_grid_instances.py --dry-run
```

Write the collection to its default directory:

```bash
python generate_grid_instances.py
```

Each of the five grid sizes has one reproducible master graph, one ordered list
of 10 intruders, and one ordered list of 100 journeyers. The nine instances per
grid take nested prefixes of 1, 5, or 10 intruders and 10, 50, or 100 journeyers,
so agent data and arc parameters remain fixed within a grid-size comparison.
Use `--output PATH`, `--seed N`, or `--overwrite` when a nonstandard collection
is required.

The older general-purpose random generator remains available:

```bash
python instances.py --nodes 20 --edges 60 --intruders 3 --journeyers 5 \
    --seed 1 --output instances/generated_seed_1.json
```

Generation is deterministic for a fixed seed. Generated instances contain a
feasible checkpoint certificate and are validated before being saved.

## Verify correctness

```bash
python -m pytest -q
```

The regression tests compare all formulations against a small independently
enumerated instance and check the known LP relaxation values. The branch-and-cut
tests verify min-cut behavior, generated cut validity, convergence, and limit
reporting. Heuristic tests independently verify the scalarized minimum cuts,
weighted budgets, auxiliary objectives, convergence, and unified-runner output.
Solver-dependent tests skip with a clear message when no Gurobi license is available.

## Structure

- `run.py`: single- and multi-instance experiment runner.
- `formulations.py`: model builders for formulations 1--4 and common solve logic.
- `branch_and_cut.py`: separation and solve procedure for formulation 4.
- `heuristics.py`: implementations of the general and single-intruder heuristics.
- `graph_algorithms.py`: shared directed shortest-path and minimum-cut routines.
- `instances.py`: instance validation, JSON I/O, preparation, and generation.
- `generate_grid_instances.py`: deterministic rectangular-grid collection generator.
- `validation.py`: independent allocation, result, relaxation, and cut checks.
- `utils.py`: Gurobi environment, result normalization, metadata, and CSV output.
- `instances/`: JSON experiment instances.
- `results/`: generated CSV outputs; CSV files are ignored by Git.
- `tests/`: compact mathematical regression suite.
- `docs/`: local reference PDFs and legacy example code; ignored by Git.

Constraint numbers in the source correspond to the formulation document kept
locally under `docs/`.
