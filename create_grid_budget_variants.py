"""Copy weighted grids with 120% and 150% budgets, rounded down exactly."""

import argparse
import copy
import re
from pathlib import Path

from instances import load_instance, save_instance, validate_instance

BUDGET_PERCENTAGES = (120, 150)
WEIGHTED_GRID_STEM = re.compile(r"grid_\d+x\d+_i\d+_j\d+_seed-?\d+_c")


def budget_variant_path(source_path, percentage):
    source_path = Path(source_path)
    if source_path.suffix != ".json" or not WEIGHTED_GRID_STEM.fullmatch(source_path.stem):
        raise ValueError(f"Expected an original weighted grid ending in '_c.json': {source_path}")
    if type(percentage) is not int or percentage not in BUDGET_PERCENTAGES:
        raise ValueError("percentage must be 120 or 150")
    return source_path.with_name(f"{source_path.stem}_b{percentage}.json")


def build_grid_budget_variant(instance, source_path, percentage):
    """Preserve the source data and certificate, changing only name/budget/provenance."""
    destination = budget_variant_path(source_path, percentage)
    if "grid" not in instance:
        raise ValueError("The source instance must have grid metadata")
    variant = copy.deepcopy(instance)
    variant["name"] = destination.stem
    # Integer arithmetic avoids floating-point rounding at integer boundaries.
    variant["budget"] = instance["budget"] * percentage // 100
    variant["budget_scaling"] = {
        "source": Path(source_path).name,
        "original_budget": instance["budget"],
        "multiplier": percentage / 100,
        "rounding": "floor",
    }
    # complex_generation remains the record of the original minimum-budget
    # solve. No optimality claim is made about the enlarged budget.
    validate_instance(variant)
    return variant


def create_grid_budget_variants(
    instances_directory=Path("instances/grid_collection"), *, overwrite=False
):
    """Write two sisters beside each original weighted grid, never scaling variants."""
    directory = Path(instances_directory)
    sources = sorted(
        source for source in directory.glob("grid_*_c.json")
        if source.is_file() and WEIGHTED_GRID_STEM.fullmatch(source.stem)
    )
    if not sources:
        raise FileNotFoundError(f"No original weighted grids found in {directory}")
    destinations = [
        budget_variant_path(source, percentage)
        for source in sources for percentage in BUDGET_PERCENTAGES
    ]
    existing = [destination for destination in destinations if destination.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {existing[0]}; use --overwrite to replace files"
        )
    for source in sources:
        instance = load_instance(source)
        for percentage in BUDGET_PERCENTAGES:
            variant = build_grid_budget_variant(instance, source, percentage)
            destination = budget_variant_path(source, percentage)
            save_instance(variant, destination)
            print(f"Saved {destination.name}: budget={variant['budget']}", flush=True)
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-directory", type=Path, default=Path("instances/grid_collection"),
        help="Directory of weighted grid originals (default: instances/grid_collection)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destinations = create_grid_budget_variants(
        args.instances_directory, overwrite=args.overwrite
    )
    print(f"Created {len(destinations)} grid budget variants")


if __name__ == "__main__":
    main()
