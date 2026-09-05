"""Exercise real Windows-compatible spawn/pipe recovery without requiring WLS."""

import csv
import os

import pytest
from gurobipy import GRB, GurobiError

import experiment_supervisor as supervisor
import memory_limits
from run import _row_from_result, print_results
from utils import empty_model_result


POLICY = {"memory_limit_gb": 2.5, "memory_limit_source": "explicit"}


class FakeEnv:
    created = 0

    def __init__(self):
        type(self).created += 1
        assert self.created == 1, "Successful tasks must reuse the same environment"
        self.params = {}

    def setParam(self, name, value):
        self.params[name] = value

    def dispose(self):
        pass


def fake_comparison(instance, *, env, phase_callback, **arguments):
    phase_callback("build_solve")
    formulation = arguments["formulations"][0] if arguments["formulations"] else None
    heuristic = arguments["heuristics"][0] if arguments["heuristics"] else None
    if heuristic != "ash":
        assert env.params["SoftMemLimit"] == POLICY["memory_limit_gb"]
    if formulation == 2 and arguments["mode"] == "integer" and arguments["repetition"] == 1:
        failure = instance.get("failure")
        if failure == "python":
            raise MemoryError("simulated Python allocation failure")
        if failure == "gurobi":
            raise GurobiError(GRB.Error.OUT_OF_MEMORY, "simulated Gurobi allocation failure")
        if failure == "error":
            raise RuntimeError("simulated model build failure")
        if failure == "validation":
            phase_callback("validation")
            raise AssertionError("simulated invalid solution")
        if failure == "exit":
            os._exit(73)
    result = empty_model_result(formulation, arguments["mode"] == "relaxation")
    if heuristic:
        result["method"] = heuristic
    row = _row_from_result(result, instance, arguments["repetition"], 0, 1)
    # None of this deliberately unpicklable detail should cross the pipe.
    return {"rows": [row], "results": {"large": lambda: None}, "oracle": None}


def fake_worker(connection, policy):
    supervisor.load_gurobi_env = FakeEnv
    supervisor.run_comparison = fake_comparison
    supervisor._worker(connection, policy)


def fake_client(policy):
    return supervisor.WorkerClient(policy, target=fake_worker)


@pytest.mark.parametrize("failure,status", [
    ("python", "OUT_OF_MEMORY"), ("gurobi", "OUT_OF_MEMORY"),
    ("error", "ERROR"), ("exit", "PROCESS_FAILED"),
    ("validation", "VALIDATION_FAILED"),
])
def test_worker_failure_checkpoints_and_continues_every_scheduled_task(tmp_path, failure, status):
    filename = tmp_path / "experiment.csv"
    result = supervisor.run_supervised_experiments(
        [{"name": "first", "failure": failure}, {"name": "second"}],
        repetitions=2, csv_filename=filename, memory_policy=POLICY,
        formulations=(1, 2, 3), heuristics=("ah", "ash"),
        _client_factory=fake_client,
    )
    rows = result["rows"]
    assert len(rows) == 32
    failures = [row for row in rows if row["status"] != "TIME_LIMIT"]
    assert len(failures) == 1
    failed = failures[0]
    assert failed["status"] == status
    assert (failed["method"], failed["mode"], failed["repetition"]) == ("f2", "integer", 1)
    assert not failed["has_solution"]
    assert failed["objective_value"] is failed["dual_bound"] is failed["num_variables"] is None
    assert failed["error_phase"] == ("validation" if failure == "validation" else "build_solve")
    assert failed["worker_exit_code"] == (73 if failure == "exit" else None)
    if failure == "gurobi":
        assert failed["error_code"] == GRB.Error.OUT_OF_MEMORY
    assert all(row["memory_limit_gb"] == 2.5 for row in rows)
    assert rows[-1]["instance"] == "second"
    with filename.open(newline="") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == len(rows)
    assert saved[2]["status"] == status
    assert saved[2]["num_variables"] == ""
    print_results(rows)  # Unknown diagnostic counts must also print successfully.


def test_successful_tasks_reuse_process_and_environment():
    client = fake_client(POLICY)
    task = {"instance": {"name": "test"}, "method": "f1", "arguments": {
        "formulations": (1,), "heuristics": (), "mode": "integer", "repetition": 1,
    }}
    try:
        assert client.solve(task)["kind"] == "result"
        pid = client.process.pid
        assert client.solve(task)["kind"] == "result"
        assert client.process.pid == pid
    finally:
        client.close()
    assert client.process is None


class ScriptedClient:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.tasks = []
        self.closed = False

    def solve(self, task):
        self.tasks.append(task)
        message = next(self.messages)
        if isinstance(message, BaseException):
            raise message
        return message

    def close(self):
        self.closed = True


def success(formulation=1, mode="integer", objective=None, bound=None):
    result = empty_model_result(formulation, mode == "relaxation")
    if objective is not None:
        result.update(status_name="OPTIMAL", objective_value=objective, has_solution=True,
                      solution_type=mode, dual_bound=bound, separation_complete=True)
    row = _row_from_result(result, {"name": "test"}, 1, 0, 1)
    return {"kind": "result", "row": row, "oracle": None}


def test_license_start_failure_is_reported_by_real_worker():
    def unavailable():
        raise GurobiError(10009, "Too many sessions")

    class Connection:
        def __init__(self):
            self.messages = []

        def recv(self):
            return {"method": "f1"}

        def send(self, message):
            self.messages.append(message)

        def close(self):
            pass

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(supervisor, "load_gurobi_env", unavailable)
        connection = Connection()
        supervisor._worker(connection, POLICY)
    message = connection.messages[-1]
    assert message["kind"] == "error"
    assert message["phase"] == "license"
    assert message["retry_license"]


def license_error():
    return {"kind": "error", "phase": "license", "status": "LICENSE_ERROR",
            "retry_license": True, "error_message": "Too many sessions"}


def test_license_wait_retries_same_task_without_failed_experiment_row(tmp_path):
    client = ScriptedClient([license_error(), license_error(), success()])
    waits = []
    result = supervisor.run_supervised_experiments(
        [{"name": "test"}], formulations=(1,), mode="integer", memory_policy=POLICY,
        _client_factory=lambda policy: client, _sleep=waits.append,
        csv_filename=tmp_path / "retry.csv",
    )
    assert waits == [310, 310]
    assert client.tasks[0] == client.tasks[1] == client.tasks[2]
    assert len(result["rows"]) == 1
    assert result["rows"][0]["status"] == "TIME_LIMIT"
    assert client.closed


def test_persistent_license_failure_is_bounded_and_next_method_runs():
    client = ScriptedClient([license_error()] * 3 + [success(2)])
    result = supervisor.run_supervised_experiments(
        [{"name": "test"}], formulations=(1, 2), mode="integer", memory_policy=POLICY,
        _client_factory=lambda policy: client, _sleep=lambda seconds: None,
    )
    assert [row["status"] for row in result["rows"]] == ["LICENSE_ERROR", "TIME_LIMIT"]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), OSError("disk full")])
def test_parent_failures_close_worker_and_preserve_completed_csv(tmp_path, interruption):
    filename = tmp_path / "partial.csv"
    client = ScriptedClient([success(), interruption])
    with pytest.raises(type(interruption)):
        supervisor.run_supervised_experiments(
            [{"name": "test"}], formulations=(1, 2), mode="integer", memory_policy=POLICY,
            csv_filename=filename, _client_factory=lambda policy: client,
        )
    assert client.closed
    with filename.open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 1


def test_csv_write_failure_does_not_become_an_instance_error(tmp_path, monkeypatch):
    client = ScriptedClient([success()])

    def cannot_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr(supervisor, "save_rows", cannot_save)
    with pytest.raises(OSError, match="disk full"):
        supervisor.run_supervised_experiments(
            [{"name": "test"}], formulations=(1,), mode="integer", memory_policy=POLICY,
            csv_filename=tmp_path / "unwritable.csv", _client_factory=lambda policy: client,
        )
    assert client.closed


def test_invalid_cross_method_relaxation_bound_is_flagged():
    client = ScriptedClient([success(1, objective=10, bound=10), success(1, "relaxation", 12, 12)])
    rows = supervisor.run_supervised_experiments(
        [{"name": "test"}], formulations=(1,), memory_policy=POLICY,
        _client_factory=lambda policy: client,
    )["rows"]
    assert rows[1]["status"] == "VALIDATION_FAILED"
    assert rows[1]["validation_passed"] is False
    assert "invalid relaxation bound" in rows[1]["error_message"]


def test_worker_creation_failure_is_a_reportable_failure(monkeypatch):
    client = supervisor.WorkerClient(POLICY)

    def fail():
        raise OSError("cannot create process")

    monkeypatch.setattr(client, "_start", fail)
    result = client.solve({})
    assert result["kind"] == "error"
    assert result["phase"] == "startup"
    assert result["status"] == "ERROR"
    client.close()


def test_cross_method_reference_and_relaxation_validation_survive_isolation():
    client = ScriptedClient([
        success(1, objective=10, bound=10), success(1, "relaxation", 5, 5),
        success(2, objective=10, bound=10), success(2, "relaxation", 7, 7),
    ])
    rows = supervisor.run_supervised_experiments(
        [{"name": "test"}], formulations=(1, 2), memory_policy=POLICY,
        _client_factory=lambda policy: client,
    )["rows"]
    assert rows[0]["reference_objective"] == rows[2]["reference_objective"] == 10
    assert rows[0]["reference_gap"] == 0
    assert rows[1]["validation_passed"] and rows[3]["validation_passed"]


def test_cross_method_disagreement_is_recorded_and_next_instance_runs():
    client = ScriptedClient([
        success(1, objective=10), success(2, objective=20), success(), success(2),
    ])
    rows = supervisor.run_supervised_experiments(
        [{"name": "first"}, {"name": "second"}], formulations=(1, 2), mode="integer",
        memory_policy=POLICY, _client_factory=lambda policy: client,
    )["rows"]
    assert [row["status"] for row in rows] == [
        "VALIDATION_FAILED", "VALIDATION_FAILED", "TIME_LIMIT", "TIME_LIMIT",
    ]
    assert rows[0]["validation_passed"] is False
    assert rows[0]["error_phase"] == "comparison"


def test_automatic_memory_limit_reserves_python_and_os_headroom(monkeypatch):
    monkeypatch.setattr(memory_limits, "physical_memory", lambda: (32e9, 24e9))
    assert memory_limits.resolve_memory_limit()["memory_limit_gb"] == pytest.approx(14.4)
    monkeypatch.setattr(memory_limits, "physical_memory", lambda: (32e9, 32e9))
    assert memory_limits.resolve_memory_limit()["memory_limit_gb"] == 16
    assert memory_limits.resolve_memory_limit(8) == {
        "memory_limit_gb": 8, "memory_limit_source": "explicit",
    }
    assert memory_limits.resolve_memory_limit("none")["memory_limit_gb"] is None


@pytest.mark.parametrize("value", [0, -1, "wrong", "nan", "inf"])
def test_invalid_memory_limits_are_rejected(value):
    with pytest.raises(ValueError):
        memory_limits.resolve_memory_limit(value)


def test_system_memory_detection_is_sane():
    total, available = memory_limits.physical_memory()
    assert 0 < available <= total
