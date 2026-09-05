import csv
import math
import platform
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import tomllib
from gurobipy import GRB, Env, gurobi

STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    GRB.WORK_LIMIT: "WORK_LIMIT",
    GRB.MEM_LIMIT: "MEM_LIMIT",
}


class MemoryLimitReached(Exception):
    """A subproblem stopped gracefully before completing its required work."""


@contextmanager
def dispose_on_error(model):
    """Transfer a built model to its caller only if construction succeeds."""
    try:
        yield
    except BaseException:
        model.dispose()
        raise


def load_gurobi_env(secrets_file=None):
    """Create the Gurobi environment used by runners and tests."""
    if secrets_file is None:
        secrets_file = Path(__file__).resolve().parent / "gurobi_secrets.toml"
    else:
        secrets_file = Path(secrets_file)

    if not secrets_file.exists():
        # This permits normal local/network license discovery when WLS
        # credentials are not stored in the repository directory.
        return Env()

    with secrets_file.open("rb") as file:
        data = tomllib.load(file)

    try:
        params = data["gurobi"]
        credentials = {
            "WLSACCESSID": params["WLSACCESSID"],
            "WLSSECRET": params["WLSSECRET"],
            "LICENSEID": int(params["LICENSEID"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "gurobi_secrets.toml must define WLSACCESSID, WLSSECRET, "
            "and an integer LICENSEID in a [gurobi] section."
        ) from error

    return Env(params=credentials)


def configure_model(
    model,
    *,
    output_flag=0,
    time_limit=None,
    solver_seed=0,
    threads=1,
):
    """Apply the common solver settings used in every formulation."""
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if solver_seed is not None:
        model.Params.Seed = solver_seed
    if threads is not None:
        model.Params.Threads = threads


def _finite_value(value):
    value = float(value)
    return value if math.isfinite(value) else None


def empty_model_result(formulation, relax, model=None, status=GRB.TIME_LIMIT):
    """A result with no incumbent, including when optimize() was never called."""
    linear = int(model.NumConstrs) if model is not None else 0
    quadratic = int(model.NumQConstrs) if model is not None else 0
    return {
        "formulation": formulation,
        "relax": relax,
        "objective_value": None,
        "has_solution": False,
        "solution_type": "none",
        "runtime": 0.0,
        "solver_runtime": 0.0,
        "status": int(status),
        "status_name": STATUS_NAMES.get(status, f"STATUS_{status}"),
        "dual_bound": None,
        "gap": None,
        "num_variables": int(model.NumVars) if model is not None else 0,
        "num_constraints": linear + quadratic,
        "num_linear_constraints": linear,
        "num_quadratic_constraints": quadratic,
        "nodes_explored": 0.0,
        "simplex_iterations": 0.0,
        "cuts": 0,
        "cut_iterations": 0,
        "master_solves": 0,
        "separation_time": 0.0,
        "separation_complete": False,
        "variables": {},
    }


def collect_model_result(model, formulation, relax):
    if model.Status == GRB.LOADED:
        return empty_model_result(formulation, relax, model, status=GRB.LOADED)
    objective_value = None
    dual_bound = None

    if model.SolCount > 0:
        objective_value = _finite_value(model.ObjVal)

    try:
        dual_bound = _finite_value(model.ObjBound)
    except AttributeError:
        # A feasible interrupted LP objective is not a lower bound.
        dual_bound = objective_value if model.Status == GRB.OPTIMAL else None

    if objective_value is None or dual_bound is None:
        gap = None
    elif abs(objective_value) <= 1e-10:
        gap = 0.0 if abs(objective_value - dual_bound) <= 1e-10 else None
    else:
        gap = abs(objective_value - dual_bound) / abs(objective_value)

    return {
        "formulation": formulation,
        "relax": relax,
        "objective_value": objective_value,
        "has_solution": objective_value is not None,
        "solution_type": (
            ("relaxation" if relax else "integer")
            if objective_value is not None else "none"
        ),
        "runtime": float(model.Runtime),
        "solver_runtime": float(model.Runtime),
        "status": int(model.Status),
        "status_name": STATUS_NAMES.get(
            model.Status, f"STATUS_{model.Status}"
        ),
        "dual_bound": dual_bound,
        "gap": gap,
        "num_variables": int(model.NumVars),
        "num_constraints": int(model.NumConstrs + model.NumQConstrs),
        "num_linear_constraints": int(model.NumConstrs),
        "num_quadratic_constraints": int(model.NumQConstrs),
        "nodes_explored": float(model.NodeCount),
        "simplex_iterations": float(model.IterCount),
        "cuts": 0,
        "cut_iterations": 0,
        "master_solves": 1,
        "separation_time": 0.0,
        "separation_complete": True,
        "variables": {},
    }


def variable_values(variables):
    return {
        name: {index: variable.X for index, variable in values.items()}
        for name, values in variables.items()
    }


def software_metadata():
    return {
        "python_version": platform.python_version(),
        "gurobi_version": ".".join(str(value) for value in gurobi.version()),
        "platform": sys.platform,
    }


def save_rows(rows, filename):
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()

        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

