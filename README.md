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

Duplicate nested retraining was removed; this changes the generated model pool and requires new measurements. ANN output indices need consistent 1-based labels. Some diversity formulas have undefined denominators for degenerate predictions; review them and selection stopping behaviour before large experiments. `Main.m` is a separately attributed PSO example whose current call signatures do not match the modified ensemble cost function. Auxiliary `clusteringPSO.m` still depends on unprovided `fusionPSO`.

The cleanup aligns function/file names, restores visible warnings, makes CSV handles close on errors, and tests constant-feature normalisation plus CSV append behaviour. MATLAB experiments and the new utility tests have **not** been executed in the cleanup environment, which has no MATLAB installation. Static parsing is not a reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
