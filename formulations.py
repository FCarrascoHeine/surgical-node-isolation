from gurobipy import GRB, Model, quicksum

from instances import prepare_instance
from time_budget import BudgetExpired, TimeBudget
from utils import (
    collect_model_result, configure_model, empty_model_result, variable_values,
)


def _add_common_variables(model, data, binary_type):
    nodes = data["nodes"]
    edges = data["edges"]
    intruders = data["intruders"]
    journeyers = data["journeyers"]

    x = {
        edge: model.addVar(
            vtype=binary_type,
            lb=0,
            ub=1,
            name="x[{},{}]".format(*edge),
        )
        for edge in edges
    }
    y = {
        (intruder, node): model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=1,
            name=f"y[{intruder},{node}]",
        )
        for intruder in intruders
        for node in nodes
    }
    z = {
        (journeyer, edge): model.addVar(
            vtype=binary_type,
            lb=0,
            ub=1,
            name="z[{},{},{}]".format(journeyer, *edge),
        )
        for journeyer in journeyers
        for edge in edges
    }
    model.update()
    return x, y, z


def _add_common_constraints(model, data, x, y, z, number_offset=0):
    nodes = data["nodes"]
    edges = data["edges"]

    model.addConstr(
        quicksum(data["checkpoint_cost"][edge] * x[edge] for edge in edges)
        <= data["budget"],
        name=f"budget_{1 + number_offset}",
    )

    for journeyer in data["journeyers"]:
        for node in nodes:
            inflow = quicksum(z[journeyer, edge] for edge in edges if edge[1] == node)
            outflow = quicksum(z[journeyer, edge] for edge in edges if edge[0] == node)
            model.addConstr(
                inflow - outflow == data["balance"][journeyer, node],
                name=f"journeyer_balance_{2 + number_offset}[{journeyer},{node}]",
            )

    for intruder in data["intruders"]:
        source = data["intruder_source"][intruder]
        target = data["intruder_target"][intruder]
        model.addConstr(
            y[intruder, source] == 0,
            name=f"intruder_source_{3 + number_offset}[{intruder}]",
        )
        for tail, head in edges:
            model.addConstr(
                y[intruder, head] - y[intruder, tail] <= x[tail, head],
                name=f"intruder_arc_{4 + number_offset}[{intruder},{tail},{head}]",
            )
        model.addConstr(
            y[intruder, target] >= 1,
            name=f"intruder_target_{5 + number_offset}[{intruder}]",
        )


def _build_common_model(
    instance,
    formulation,
    relax,
    time_limit,
    output_flag,
    solver_seed,
    threads,
    env,
):
    data = prepare_instance(instance)
    binary_type = GRB.CONTINUOUS if relax else GRB.BINARY
    model = Model(f"SNI_formulation_{formulation}", env=env)
    configure_model(
        model,
        output_flag=output_flag,
        time_limit=time_limit,
        solver_seed=solver_seed,
        threads=threads,
    )
    x, y, z = _add_common_variables(model, data, binary_type)
    return model, data, binary_type, x, y, z


def build_formulation_1(
    instance,
    relax=False,
    time_limit=None,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    model, data, _, x, y, z = _build_common_model(
        instance, 1, relax, time_limit, output_flag, solver_seed, threads, env
    )
    edges = data["edges"]
    journeyers = data["journeyers"]
    tau = data["tau"]
    inspection_time = data["inspection_time"]

    model.setObjective(
        quicksum(
            tau[edge] * z[journeyer, edge]
            + inspection_time[edge] * x[edge] * z[journeyer, edge]
            for journeyer in journeyers
            for edge in edges
        ),
        GRB.MINIMIZE,
    )
    _add_common_constraints(model, data, x, y, z)
    model.Params.NonConvex = 2
    model.update()
    return model, {"x": x, "y": y, "z": z}


def build_formulation_2(
    instance,
    relax=False,
    time_limit=None,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    model, data, binary_type, x, y, z = _build_common_model(
        instance, 2, relax, time_limit, output_flag, solver_seed, threads, env
    )
    edges = data["edges"]
    journeyers = data["journeyers"]
    tau = data["tau"]
    inspection_time = data["inspection_time"]
    alpha = {
        (journeyer, edge): model.addVar(
            vtype=binary_type,
            lb=0,
            ub=1,
            name="alpha[{},{},{}]".format(journeyer, *edge),
        )
        for journeyer in journeyers
        for edge in edges
    }
    model.update()

    model.setObjective(
        quicksum(
            tau[edge] * z[journeyer, edge]
            + inspection_time[edge] * alpha[journeyer, edge]
            for journeyer in journeyers
            for edge in edges
        ),
        GRB.MINIMIZE,
    )
    _add_common_constraints(model, data, x, y, z)
    for journeyer in journeyers:
        for edge in edges:
            model.addConstr(
                alpha[journeyer, edge]
                >= x[edge] + z[journeyer, edge] - 1,
                name="product_7[{},{},{}]".format(journeyer, *edge),
            )

    model.update()
    return model, {"x": x, "y": y, "z": z, "alpha": alpha}


def build_formulation_3(
    instance,
    relax=False,
    time_limit=None,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    model, data, _, x, y, z = _build_common_model(
        instance, 3, relax, time_limit, output_flag, solver_seed, threads, env
    )
    edges = data["edges"]
    journeyers = data["journeyers"]
    tau = data["tau"]
    inspection_time = data["inspection_time"]
    beta = {
        edge: model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=len(journeyers),
            name="beta[{},{}]".format(*edge),
        )
        for edge in edges
    }
    model.update()

    model.setObjective(
        quicksum(
            tau[edge]
            * quicksum(z[journeyer, edge] for journeyer in journeyers)
            + inspection_time[edge] * beta[edge]
            for edge in edges
        ),
        GRB.MINIMIZE,
    )
    _add_common_constraints(model, data, x, y, z, number_offset=10)
    for edge in edges:
        flow = quicksum(z[journeyer, edge] for journeyer in journeyers)
        model.addConstr(
            beta[edge] <= flow,
            name="aggregate_upper_flow_17[{},{}]".format(*edge),
        )
        model.addConstr(
            beta[edge] >= flow - len(journeyers) * (1 - x[edge]),
            name="aggregate_lower_18[{},{}]".format(*edge),
        )
        model.addConstr(
            beta[edge] <= len(journeyers) * x[edge],
            name="aggregate_upper_x_19[{},{}]".format(*edge),
        )

    model.update()
    return model, {"x": x, "y": y, "z": z, "beta": beta}


def build_formulation_4(
    instance,
    relax=False,
    time_limit=None,
    output_flag=0,
    solver_seed=0,
    threads=1,
    env=None,
):
    data = prepare_instance(instance)
    edges = data["edges"]
    journeyers = data["journeyers"]
    binary_type = GRB.CONTINUOUS if relax else GRB.BINARY
    model = Model("SNI_formulation_4", env=env)
    configure_model(
        model,
        output_flag=output_flag,
        time_limit=time_limit,
        solver_seed=solver_seed,
        threads=threads,
    )

    x = {
        edge: model.addVar(
            vtype=binary_type,
            lb=0,
            ub=1,
            name="x[{},{}]".format(*edge),
        )
        for edge in edges
    }
    alpha = {
        (journeyer, edge): model.addVar(
            vtype=binary_type,
            lb=0,
            ub=1,
            name="alpha[{},{},{}]".format(journeyer, *edge),
        )
        for journeyer in journeyers
        for edge in edges
    }
    phi = {
        journeyer: model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name=f"phi[{journeyer}]",
        )
        for journeyer in journeyers
    }
    model.update()

    model.setObjective(
        quicksum(
            quicksum(
                data["inspection_time"][edge] * alpha[journeyer, edge]
                for edge in edges
            )
            + phi[journeyer]
            for journeyer in journeyers
        ),
        GRB.MINIMIZE,
    )
    model.addConstr(
        quicksum(data["checkpoint_cost"][edge] * x[edge] for edge in edges)
        <= data["budget"],
        name="budget_23",
    )
    for journeyer in journeyers:
        for edge in edges:
            model.addConstr(
                alpha[journeyer, edge] <= x[edge],
                name="alpha_link_27[{},{},{}]".format(journeyer, *edge),
            )

    model.Params.FeasibilityTol = 1e-8
    if not relax:
        model.Params.LazyConstraints = 1
    model.update()
    model._data = data
    model._solve_env = env
    return model, {"x": x, "alpha": alpha, "phi": phi}


BUILDERS = {
    1: build_formulation_1,
    2: build_formulation_2,
    3: build_formulation_3,
    4: build_formulation_4,
}


def build_model(instance, formulation, **kwargs):
    try:
        builder = BUILDERS[int(formulation)]
    except (KeyError, ValueError) as error:
        raise ValueError("formulation must be one of 1, 2, 3, or 4") from error
    return builder(instance, **kwargs)


def _solve_compact(instance, formulation, **kwargs):
    budget = TimeBudget(kwargs.get("time_limit"))
    relax = bool(kwargs.get("relax", False))
    model = None
    try:
        budget.check()
        kwargs["time_limit"] = budget.remaining()
        model, variables = build_model(instance, formulation, **kwargs)
        budget.apply_to(model)
        model.optimize()
        wall_time = budget.elapsed()
        result = collect_model_result(model, formulation=formulation, relax=relax)
        result["runtime"] = wall_time
        if model.SolCount > 0:
            result["variables"] = variable_values(variables)
        return result
    except BudgetExpired:
        result = empty_model_result(formulation, relax, model)
        result["runtime"] = budget.elapsed()
        return result
    finally:
        if model is not None:
            model.dispose()


def solve_instance(instance, formulation, **kwargs):
    formulation = int(formulation)
    if formulation == 4:
        from branch_and_cut import solve_instance as solve_branch_and_cut

        return solve_branch_and_cut(instance, **kwargs)
    if formulation not in (1, 2, 3):
        raise ValueError("formulation must be one of 1, 2, 3, or 4")
    return _solve_compact(instance, formulation, **kwargs)
