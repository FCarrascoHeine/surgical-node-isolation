# Why the 100-node c instances do not improve

For all six 100-node c instances, the aggregated LP already attains the integer optimum. This follows from the checkpoint budget and is not a separation failure.

## What differs between c and dir

The paired JSON files have identical nodes, arcs, transit times, inspection times, intruders, and journeyers. The differences are checkpoint costs and budget (plus the stored feasible checkpoint certificate in c). The c costs are integers from 10 to 50; the dir costs are one.

For each of the six 100-node c instances, the stored budget equals the minimum checkpoint cost even in the continuous checkpoint/potential model. Furthermore, only one checkpoint vector is feasible at that budget.

## Checking uniqueness

Let r be the binary checkpoint vector stored in `known_feasible_checkpoints`. Over the continuous intruder-separation constraints, bounds, and the original budget constraint, solve

```text
maximize sum(1 - x[e] for e with r[e] = 1)
       + sum(x[e] for e with r[e] = 0).
```

This linear objective is the distance from r: every term is nonnegative. If its optimal maximum is zero, every feasible x must equal r. All six 100-node c instances gave maximum zero, within numerical tolerance. The auxiliary LPs use feasibility and optimality tolerances of 1e-8; distances up to 1e-6 are treated as zero.

The [diagnostic CSV](c_diagnostics.csv) records the minimum checkpoint costs, maximum distances, and solver statuses. Status 2 is Gurobi OPTIMAL. Two small regression tests verify that this calculation distinguishes a unique allocation from two alternative minimum-cost allocations.

## Why integral x makes the inequalities redundant

The aggregated model already includes

```text
0 <= beta[e] <= sum_j z[j,e]
beta[e] <= |J| * x[e]
beta[e] >= sum_j z[j,e] - |J| * (1 - x[e]).
```

When x[e] = 0, these give beta[e] = 0, while every subset lower bound is nonpositive because z[j,e] <= 1. When x[e] = 1, they give beta[e] = sum_j z[j,e], which is at least the sum over any subset. Therefore every subset inequality is already satisfied, even if z is fractional.

Consequently the uniquely fixed binary checkpoint vector makes formulations 3, 2, and 3_VI equivalent in objective value on these six instances. Increasing the cut cap cannot strengthen their LP bound.

## Independent integer-optimality check

For each stored checkpoint certificate, `evaluate_allocation` checks the budget and every intruder's separation, then computes shortest journeyer paths with the corresponding inspection delays. The resulting feasible integer objective matches the optimal aggregated LP within 1e-6 absolute tolerance on all six 100-node c instances. Since the LP is a lower bound and this feasible integer solution is an upper bound, equality certifies the integer optimum without another MIP solve.

The allocation costs and objectives are saved in [c_certificates.csv](c_certificates.csv), and the original LP values are in the [100-node experiment](../lp_100_node_20260925/README.md).

## Scope of the explanation

The suffix c does not make the inequalities redundant in general. The larger diagnostic already finds two 200-node c instances with alternative budget-feasible checkpoint vectors: `200_1172_500_10_1c` and `200_1172_50_10_1c`. For the first, the stored budget is 1350 and the continuous minimum checkpoint cost is 1329; for the second, both are 1314 but the optimal allocation is not unique. Their LP gains must be measured rather than inferred from the filename.

The larger experiment and controlled budget increases are reported in [README.md](README.md). The budget controls change only the budget in memory, preserving all source JSON files.

## Reproduce the checkpoint diagnostic

From the repository root:

```bash
python diagnose_valid_inequalities.py 'instances/100_*c.json' \
    'instances/200_*c.json' 'instances/500_*_80_*c.json' \
    --checkpoint-only --output-dir results/lp_larger_20260925
```
