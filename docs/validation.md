# Executed validation

Date: 17 October 2026. [Successful GitHub Actions run 39000469683](https://github.com/zohaibjan/ACM_TIST_CSME/actions/runs/39000469683) at commit `d09358b8cdf392ceb25b5e857847bd2745489781`.

- MATLAB R2024b on Ubuntu 22.04.
- 27 tests passed; zero failed or incomplete.
- 42 root function files loaded/compiled by MATLAB without executing the experiment entry scripts.
- Installed toolboxes: Statistics and Machine Learning, Deep Learning, Global Optimization, Optimization (dependency), and Parallel Computing.

## What was executed

Actual SVM/tree and ANN training, particleswarm, ga, gamultiobj, all eight binary PSO variants, feature selection with fitted decision trees, and core utility regressions.

Toolbox tests fit real models and/or call real solvers. Deterministic predict-compatible fixtures are also used to isolate selection and voting behavior. Smoke tests use small synthetic datasets and bounded solver/epoch budgets; default research budgets remain available.

## Changes and behavior

GA uses the same >0.5 nonempty mask during scoring and return. Multi-objective GA measures pairwise double fault on selected models, returns one score-aligned ensemble, and exposes all Pareto masks/scores separately. BPSO personal-best storage is a vector. Feature PSO uses supported option names, nonempty masks and reproducible learner initialization. Prediction/training failures propagate instead of manufacturing votes. Original third-party author notices remain.

## Reproduce

Use a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The workflow runs tests on MATLAB changes and pull requests, and supports manual dispatch. It has read-only repository permissions and performs no deployment.

## Limits

These tests do not execute every function branch or reproduce a paper's full datasets, published accuracy, runtime or memory results. Experiment scripts and unsupported auxiliary paths still need their documented inputs/dependencies. Root function loading is a compilation check, not full code coverage. Consult [code-review.md](code-review.md) for repository-specific limitations. Historical result CSVs were not regenerated or relabelled.
