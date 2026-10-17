# Classifier Selection and Ensemble Diversity Experiments

MATLAB experiments with clustered ensembles, classifier selection and diversity measures.

## Research reference and attribution

Jan, Z. M. and Verma, B. (2020). Multiple Strong and Balanced Clusters based Ensemble of Deep Learners. Pattern Recognition, article 107420.

This citation is retained from the original repository. Its publication label differs from the repository name; confirm the intended paper/version before citing results from this checkout.

Please cite the relevant publication and retain source author notices. Existing dataset and third-party terms still apply; this cleanup does not grant a new licence.

## Data and environment

Supply `P-Data/<name>.mat` for each name returned by `dataSetNames.m`. Each file must contain numeric `X` (samples × features) and label column `y`. Datasets are not bundled.

**MATLAB dependencies:** Statistics and Machine Learning Toolbox; Deep Learning Toolbox; Global Optimization Toolbox for optional PSO/GA paths.

No exact original MATLAB/toolbox version is recorded. Use MATLAB rather than assuming GNU Octave compatibility for the learner APIs.

## Start here

1. Open this repository as the MATLAB current folder. Use a separate session/path for each research repository: helper filenames overlap across projects.
2. Supply or extract the data described above. Check feature/label row counts and class labels.
3. Edit `Problem` in `mainProgram.m` (or `dataSetNames.m` where used), and start with one dataset and one repeat.
4. Run the synthetic utility checks, then inspect `mainProgram.m` before starting the full experiment:

```matlab
results = runtests('tests');
assertSuccess(results);
rng(42, 'twister');
mainProgram;
```

The seed is a starting point for recording runs; nested seed resets, parallel workers and toolbox versions can still affect reproducibility. `runMultiple.m` performs 50 full driver repetitions and is not the quick-start command.

## Workflow and output

`mainProgram.m` configures learners, diversity options and ten folds. `runTraining.m` forms clusters, trains classifiers once per eligible cluster, selects with `rBCS.m`, and exports accuracy and pool sizes. Diversity helpers include disagreement, double fault, correlation and Q statistics.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

Candidate predictions are computed once and validated. Randomized selection samples only unselected candidates with a remaining retry budget, so exhausted candidates cannot be revisited indefinitely. The historical strict-accuracy acceptance rule is retained; its diversity-tie branch was unreachable, so `addedDiv` remains zero. All-tied above-worst selection retains valid models, and ANN labels use the training vocabulary. Standalone diversity helpers still have undefined degenerate cases. `Main.m` has incompatible legacy optimizer signatures; `clusteringPSO.m` depends on unprovided `fusionPSO`.

## Runtime follow-up — October 2026

GA uses the same >0.5 nonempty mask during scoring and return. Multi-objective GA measures pairwise double fault on selected models, returns one score-aligned ensemble, and exposes all Pareto masks/scores separately. BPSO personal-best storage is a vector. Feature PSO uses supported option names, nonempty masks and reproducible learner initialization. Prediction/training failures propagate instead of manufacturing votes. Original third-party author notices remain.

## Correctness review — October 2026

This pass fixes implementation defects; it does not validate the associated research results. Changes can alter model pools, predictions and measured scores. Re-run the experiments before comparing corrected code with historical CSVs or published results.

- Label conversion uses numeric category names, not categorical indices.
- Voting rejects empty/nonfinite inputs; a prediction failure is reported rather than counted as a fabricated class-one vote.
- Neural models store their training labels in `net.userdata.classLabels`. Retrain old models or restore that exact training vocabulary before prediction; inferring it from a validation subset is unsafe.
- Synthetic checks cover normalization, CSV append behavior and the new prediction/selection boundaries. Additional cluster checks run where applicable.

See [the code review](docs/code-review.md) for repository-specific changes, tests and remaining limitations.

**Validation:** 27 regression/smoke tests passed in MATLAB R2024b on Linux through [GitHub Actions](https://github.com/zohaibjan/ACM_TIST_CSME/actions/workflows/matlab-tests.yml). Tests include real toolbox execution on synthetic data. Full research datasets and published accuracy results have not been reproduced. See [validation evidence and limits](docs/validation.md).

[Profile and project directory](https://github.com/zohaibjan)
