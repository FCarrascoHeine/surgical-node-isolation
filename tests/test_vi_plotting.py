import csv

import pytest

import plot_valid_inequalities as plotting


def test_plotting_cli_writes_comparison_tables_and_dispatches_renderer(monkeypatch, tmp_path):
    source = tmp_path / "results.csv"
    fields = [
        "instance", "repetition", "solver_seed", "threads", "time_limit_seconds",
        "formulation", "mode", "status", "validation_passed", "solution_type",
        "objective_value", "cuts", "max_cuts", "separation_complete",
    ]
    common = {
        "instance": "generated_dir", "repetition": "1", "solver_seed": "0",
        "threads": "1", "time_limit_seconds": "30", "mode": "relaxation",
        "status": "OPTIMAL", "validation_passed": "True",
        "solution_type": "relaxation", "cuts": "", "max_cuts": "",
        "separation_complete": "True",
    }
    rows = [
        {**common, "formulation": "3", "objective_value": "3.0"},
        {**common, "formulation": "2", "objective_value": "3.25"},
        {**common, "formulation": "3_VI", "objective_value": "3.2",
         "cuts": "5", "max_cuts": "100", "separation_complete": "False"},
    ]
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    rendered = []

    def fake_plot(compared, output, metric):
        rendered.append((compared, output, metric))

    monkeypatch.setattr(plotting, "plot_comparisons", fake_plot)
    output = tmp_path / "plots"
    compared = plotting.main([
        str(source), "--output-dir", str(output), "--metric", "absolute",
    ])

    assert len(compared) == 1
    assert compared[0]["vi_absolute"] == pytest.approx(0.2)
    assert compared[0]["gap_closed_percent"] == pytest.approx(80)
    assert rendered == [(compared, output, "absolute")]
    for name in ("lp_comparison.csv", "excluded.csv"):
        assert (output / name).is_file()
