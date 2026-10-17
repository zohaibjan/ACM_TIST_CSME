# Code review and validation scope

Candidate predictions are computed once and validated. Randomized selection samples only unselected candidates with a remaining retry budget, so exhausted candidates cannot be revisited indefinitely. The historical strict-accuracy acceptance rule is retained; its diversity-tie branch was unreachable, so `addedDiv` remains zero. All-tied above-worst selection retains valid models, and ANN labels use the training vocabulary. Standalone diversity helpers still have undefined degenerate cases. `Main.m` has incompatible legacy optimizer signatures; `clusteringPSO.m` depends on unprovided `fusionPSO`.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Still requires target-runtime validation

- MATLAB/toolbox compatibility and neural-network training/prediction APIs.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The review environment has no MATLAB installation. Static lint passed, but the MATLAB unit tests were not run. No reproduced accuracy or benchmark claim is made.
