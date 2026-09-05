"""Sequential experiments with a disposable worker and parent-owned CSV.

The worker reuses one Gurobi environment across successful tasks. All model
construction, solves, and allocation validation happen there; only summaries
cross the pipe. The parent never opens a Gurobi environment.
"""

import gc
import multiprocessing as mp
import time
from contextlib import suppress
from pathlib import Path

from gurobipy import GRB, GurobiError

from memory_limits import resolve_memory_limit
from run import (
    DEFAULT_FORMULATIONS, HEURISTIC_NAMES, _row_from_result,
    finalize_comparison, run_comparison,
)
from time_budget import TimeBudget
from utils import MemoryLimitReached, empty_model_result, load_gurobi_env, save_rows


def _exception_info(error):
    """Serialize diagnostics, never retain traceback frames or model references."""
    root = error
    seen = set()
    while root.__cause__ is not None and id(root) not in seen:
        seen.add(id(root))
        root = root.__cause__
    code = root.errno if isinstance(root, GurobiError) else None
    status = "ERROR"
    if isinstance(root, MemoryError) or code == GRB.Error.OUT_OF_MEMORY:
        status = "OUT_OF_MEMORY"
    elif isinstance(root, MemoryLimitReached):
        status = "MEM_LIMIT"
    elif isinstance(root, AssertionError):
        status = "VALIDATION_FAILED"
    elif code == GRB.Error.NO_LICENSE:
        status = "LICENSE_ERROR"
    message = str(root)[:2000]
    retry_license = isinstance(root, GurobiError) and any(
        text in message.lower() for text in ("too many sessions", "overage for too long")
    )
    if retry_license:
        status = "LICENSE_ERROR"
    return {
        "status": status, "error_type": type(root).__name__,
        "error_message": message, "error_code": code,
        "retry_license": retry_license,
    }


def _worker(connection, memory_policy):
    env = None
    try:
        while True:
            task = connection.recv()
            if task is None:
                return
            phase_name = "preparation"

            def phase(name):
                nonlocal phase_name
                phase_name = name
                connection.send({"kind": "phase", "phase": name})

            try:
                if task["method"] != "ash" and env is None:
                    phase("license")
                    env = load_gurobi_env()
                    limit = memory_policy["memory_limit_gb"]
                    if limit is not None:
                        env.setParam("SoftMemLimit", limit)
                comparison = run_comparison(
                    task["instance"], env=env, retain_variables=False,
                    phase_callback=phase, **task["arguments"],
                )
                # Discard histories, enumeration allocations, and other large
                # diagnostics too. The CSV fields are the retained contract.
                row = comparison["rows"][0]
                oracle = comparison["oracle"]
                oracle = None if oracle is None else {
                    "objective_value": oracle["objective_value"]
                }
                del comparison
                gc.collect()
                connection.send({"kind": "result", "row": row, "oracle": oracle})
                del row, oracle, task
            except Exception as error:
                info = _exception_info(error)
                connection.send({"kind": "error", "phase": phase_name, **info})
                # Every exceptional task retires the process. Even partial
                # builds and native allocator state are then released by the OS.
                return
    finally:
        if env is not None:
            with suppress(Exception):
                env.dispose()
        connection.close()


class WorkerClient:
    """Own only the process created here; never terminate unrelated solvers."""

    def __init__(self, memory_policy, *, target=_worker):
        self.memory_policy = memory_policy
        self.target = target
        self.process = None
        self.connection = None

    def _start(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=self.target, args=(child, self.memory_policy))
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self.process, self.connection = process, parent

    def solve(self, task):
        if self.process is None:
            try:
                self._start()
            except OSError as error:
                return {"kind": "error", "phase": "startup",
                        **_exception_info(error)}
        phase = "startup"
        try:
            self.connection.send(task)
            while True:
                # Pipe EOF also detects native crashes that raise no Python error.
                if self.connection.poll(0.25):
                    message = self.connection.recv()
                    if message["kind"] == "phase":
                        phase = message["phase"]
                        continue
                    if message["kind"] == "error":
                        self.close()
                    return message
                if not self.process.is_alive():
                    # Drain a final message sent immediately before process exit.
                    if self.connection.poll():
                        continue
                    break
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
            pass
        self.process.join(timeout=1)
        code = self.process.exitcode
        self.close()
        return {
            "kind": "error", "status": "PROCESS_FAILED", "phase": phase,
            "error_type": "WorkerExit", "error_code": None,
            "error_message": "Worker exited without a result; the cause is unknown",
            "worker_exit_code": code, "retry_license": False,
        }

    def close(self):
        if self.process is None:
            return
        with suppress(OSError):
            self.connection.send(None)
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
        self.connection.close()
        self.process.close()
        self.process = self.connection = None


def _result_summary(row):
    return {
        "formulation": row["formulation"], "relax": row["mode"] == "relaxation",
        "status_name": row["status"], "objective_value": row["objective_value"],
        "dual_bound": row["dual_bound"], "gap": row["gap"],
        "separation_complete": row["separation_complete"],
        "solution_type": row["solution_type"], "has_solution": row["has_solution"],
        "variables": {},
    }


def _failure_row(task, message, elapsed):
    method = task["method"]
    formulation = int(method[1:]) if method.startswith("f") else None
    arguments = task["arguments"]
    result = empty_model_result(formulation, arguments["mode"] == "relaxation")
    result.update(method=method, status_name=message["status"], runtime=elapsed)
    instance = task["instance"]
    metadata = instance if isinstance(instance, dict) else {"name": Path(instance).stem}
    row = _row_from_result(
        result, metadata, arguments["repetition"],
        arguments.get("solver_seed", 0), arguments.get("threads", 1),
    )
    for name in ("error_type", "error_message", "error_code", "worker_exit_code"):
        row[name] = message.get(name)
    row["error_phase"] = message["phase"]
    if message["status"] == "VALIDATION_FAILED":
        row["validation_passed"] = False
    # A failed build has unknown counts; zero would incorrectly look measured.
    for name in (
        "solver_runtime", "num_variables", "num_constraints", "nodes_explored",
        "simplex_iterations", "cuts", "intruder_cuts", "feasibility_cuts",
        "optimality_cuts", "cut_iterations", "master_solves", "lazy_additions",
        "separation_time",
    ):
        row[name] = None
    return row


def _wait_for_license(seconds):
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        print(f"Waiting for WLS session availability ({remaining:.0f}s remaining)", flush=True)
        time.sleep(min(30, remaining))


def run_supervised_experiments(
    instances, repetitions=1, csv_filename=None, *, memory_limit_gb="auto",
    memory_policy=None, license_retry_seconds=310, license_retries=2,
    _client_factory=WorkerClient, _sleep=_wait_for_license, **arguments,
):
    """Continue after per-method failures; retry transient WLS startup failures.

    License waiting is outside method runtime. CSV failures and user interrupts
    remain fatal: the supervisor must not silently lose results or ignore Ctrl+C.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if license_retries < 0 or license_retry_seconds < 0:
        raise ValueError("License retry count and wait must be nonnegative")
    TimeBudget(arguments.get("time_limit"))
    mode = arguments.get("mode", "both")
    formulations = tuple(dict.fromkeys(arguments.get("formulations", DEFAULT_FORMULATIONS)))
    heuristics = tuple(dict.fromkeys(arguments.get("heuristics", ())))
    if mode not in ("integer", "relaxation", "both"):
        raise ValueError("mode must be integer, relaxation, or both")
    if any(f not in DEFAULT_FORMULATIONS for f in formulations):
        raise ValueError("Invalid formulation")
    if any(h not in HEURISTIC_NAMES for h in heuristics):
        raise ValueError("Invalid heuristic")
    if not formulations and not heuristics:
        raise ValueError("At least one formulation or heuristic must be selected")
    if "env" in arguments:
        raise ValueError("A supervised worker creates its own environment; do not pass env")
    policy = memory_policy if memory_policy is not None else resolve_memory_limit(memory_limit_gb)
    limit = policy["memory_limit_gb"]
    display_limit = f"{limit:.6g}" if limit is not None else "disabled"
    print(f"Gurobi soft memory limit: {display_limit} GB "
          f"({policy['memory_limit_source']})", flush=True)
    modes = ("integer", "relaxation") if mode == "both" else (mode,)
    methods = [(f"f{f}", m) for f in formulations for m in modes]
    methods += [(h, "heuristic") for h in heuristics]
    rows = []
    client = _client_factory(policy)

    def checkpoint(_row=None):
        if csv_filename is not None:
            save_rows(rows, csv_filename)

    try:
        for instance in instances:
            for repetition in range(1, repetitions + 1):
                group_rows, results, oracle = [], {}, None
                for method, solve_mode in methods:
                    options = dict(arguments)
                    options.update(
                        repetition=repetition,
                        formulations=(int(method[1:]),) if method.startswith("f") else (),
                        heuristics=(method,) if solve_mode == "heuristic" else (),
                        mode="integer" if solve_mode == "heuristic" else solve_mode,
                    )
                    task = {"instance": instance, "method": method, "arguments": options}
                    for attempt in range(license_retries + 1):
                        start = time.perf_counter()
                        message = client.solve(task)
                        elapsed = time.perf_counter() - start
                        if (message.get("retry_license") and message.get("phase") == "license"
                                and attempt < license_retries):
                            _sleep(license_retry_seconds)
                            continue
                        break
                    if message["kind"] == "result":
                        row = message["row"]
                        if message["oracle"] is not None:
                            oracle = message["oracle"]
                    else:
                        row = _failure_row(task, message, elapsed)
                        print(f"{row['instance']} {method} {solve_mode}: {row['status']} "
                              f"({row['error_phase']}) {row['error_message']}", flush=True)
                    row.update(policy)
                    group_rows.append(row)
                    rows.append(row)
                    key = (row["formulation"], solve_mode) if method.startswith("f") else (method, solve_mode)
                    results[key] = _result_summary(row)
                    checkpoint()
                try:
                    finalize_comparison(
                        group_rows, results, oracle, formulations,
                        mode in ("integer", "both"), mode in ("relaxation", "both"),
                        arguments.get("strict_validation", True),
                        arguments.get("tolerance", 1e-6), checkpoint,
                    )
                except AssertionError as error:
                    # Disagreement is a failed validation, never silent success.
                    for row in group_rows:
                        if row["method_type"] == "formulation" and (
                            row["has_solution"] or row["dual_bound"] is not None
                        ):
                            row.update(status="VALIDATION_FAILED", validation_passed=False,
                                       error_type="AssertionError", error_message=str(error),
                                       error_phase="comparison",
                                       reference_objective=None, reference_gap=None)
                checkpoint()
    finally:
        client.close()
    return {"rows": rows}
