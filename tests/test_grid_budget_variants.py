import copy
from pathlib import Path

import pytest

import create_grid_budget_variants as generator
from create_grid_budget_variants import (
    budget_variant_path,
    build_grid_budget_variant,
    create_grid_budget_variants,
)
from instances import load_instance, save_instance, validate_instance

GRID_DIRECTORY = Path(__file__).resolve().parents[1] / "instances" / "grid_collection"
SOURCE = GRID_DIRECTORY / "grid_5x10_i1_j10_seed0_c.json"


@pytest.mark.parametrize("budget,percentage,expected", [
    (85, 120, 102), (85, 150, 127), (116, 120, 139), (255, 150, 382),
])
def test_variant_rounds_down_and_only_changes_name_budget_and_provenance(
    budget, percentage, expected
):
    instance = load_instance(SOURCE)
    instance["budget"] = budget
    before = copy.deepcopy(instance)
    variant = build_grid_budget_variant(instance, SOURCE, percentage)
    assert instance == before
    assert variant["budget"] == expected
    assert variant["name"] == f"{SOURCE.stem}_b{percentage}"
    assert variant["budget_scaling"] == {
        "source": SOURCE.name, "original_budget": budget,
        "multiplier": percentage / 100, "rounding": "floor",
    }
    assert variant.keys() == instance.keys() | {"budget_scaling"}
    assert {k: v for k, v in variant.items() if k not in {"name", "budget", "budget_scaling"}} == {
        k: v for k, v in instance.items() if k not in {"name", "budget"}
    }
    assert validate_instance(variant)
    variant["edges"][0]["checkpoint_cost"] += 1
    assert instance == before


def test_materialization_uses_only_original_weighted_grids_and_preserves_sources(tmp_path):
    source = tmp_path / SOURCE.name
    save_instance(load_instance(SOURCE), source)
    original_bytes = source.read_bytes()
    # Non-source files deliberately contain no valid instance data.
    (tmp_path / "grid_5x10_i1_j10_seed0.json").write_text("{}", encoding="utf-8")
    (tmp_path / "grid_5x10_i1_j50_seed0_c_b120.json").write_text("{}", encoding="utf-8")
    destinations = create_grid_budget_variants(tmp_path)
    assert [p.name for p in destinations] == [
        f"{source.stem}_b120.json", f"{source.stem}_b150.json",
    ]
    assert all(p.parent == source.parent for p in destinations)
    assert [load_instance(p)["budget"] for p in destinations] == [102, 127]
    with pytest.raises(FileExistsError, match="--overwrite"):
        create_grid_budget_variants(tmp_path)
    assert create_grid_budget_variants(tmp_path, overwrite=True) == destinations
    assert source.read_bytes() == original_bytes


def test_overwrite_conflicts_are_detected_before_any_loading_or_writing(tmp_path, monkeypatch):
    source = tmp_path / SOURCE.name
    source.write_text("{}", encoding="utf-8")
    conflict = budget_variant_path(source, 150)
    conflict.write_text("already exists", encoding="utf-8")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Destination conflicts must be checked before loading sources")

    monkeypatch.setattr(generator, "load_instance", unexpected_load)
    with pytest.raises(FileExistsError, match="--overwrite"):
        create_grid_budget_variants(tmp_path)
    assert not budget_variant_path(source, 120).exists()
    assert conflict.read_text(encoding="utf-8") == "already exists"


def test_empty_collection_and_attempts_to_rescale_variants_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="No original weighted grids"):
        create_grid_budget_variants(tmp_path)
    with pytest.raises(ValueError, match="original weighted grid"):
        budget_variant_path(budget_variant_path(SOURCE, 120), 150)
    with pytest.raises(ValueError, match="percentage"):
        budget_variant_path(SOURCE, 120.0)


def test_materialized_grid_budget_collection():
    sources = sorted(GRID_DIRECTORY.glob("grid_*_c.json"))
    variants = sorted(GRID_DIRECTORY.glob("grid_*_c_b*.json"))
    assert len(sources) == 45
    assert len(variants) == 90
    expected_destinations = set()
    for source in sources:
        original = load_instance(source)
        for percentage, numerator, denominator in ((120, 6, 5), (150, 3, 2)):
            destination = source.with_name(f"{source.stem}_b{percentage}.json")
            expected_destinations.add(destination)
            # Loading independently validates each saved feasibility certificate.
            variant = load_instance(destination)
            expected = copy.deepcopy(original)
            expected["name"] = destination.stem
            expected["budget"] = original["budget"] * numerator // denominator
            expected["budget_scaling"] = {
                "source": source.name, "original_budget": original["budget"],
                "multiplier": numerator / denominator, "rounding": "floor",
            }
            assert variant == expected, destination.name
            assert variant["budget"] >= original["budget"]
    assert set(variants) == expected_destinations
