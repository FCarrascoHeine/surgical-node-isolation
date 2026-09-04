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
| `--time-limit SECONDS` | No limit | Sets the Gurobi time limit for each solve. Formulation 4 also reports the wall time spent in its separation procedure. |
| `--solver-seed N` | `0` | Sets Gurobi's random seed. Keep this fixed when comparing formulations; vary it deliberately when studying solver variability. |
| `--threads N` | `1` | Sets the number of Gurobi threads. Using one thread favors repeatability; larger values may reduce runtime. Gurobi interprets `0` as its automatic setting. |
| `--max-iterations N` | `100` | Maximum number of cut-addition rounds for the formulation 4 relaxation. It has no effect on formulations 1--3 or on the formulation 4 integer callback. |
| `--max-cuts N` | No limit | Maximum total number of cuts added while solving the formulation 4 relaxation. It has no effect on the other solves. |
| `--heuristic-max-iterations N` | `100` | Maximum number of outer iterations for either heuristic. |
| `--binary-search-tolerance VALUE` | `1e-4` | Precision used by the single-intruder heuristic's capacity-weight binary search. |
| `--return-terminal-heuristic` | Disabled | Returns the terminal heuristic candidate instead of the best true-objective candidate encountered. Iteration history is retained either way. |
| `--output` | Disabled | Displays Gurobi's solver log while the experiment runs. |
| `--allow-validation-failures` | Disabled | Records failed validation instead of stopping with an error. The default behavior is deliberately strict so invalid results do not silently enter an experiment. |

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
formulation result; it is distinct from Gurobi's `gap`. `runtime` is total wall
time, while `solver_runtime` contains time reported by Gurobi.

During a batch, detailed per-variable solution mappings are used for validation and
then released after each solve. The returned experiment object and CSV retain the
compact solve summaries and result rows.

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
