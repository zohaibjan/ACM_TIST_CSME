# Classifier Selection and Ensemble Diversity Experiments

MATLAB research experiments involving clustered ensembles, classifier selection and diversity measures.

## Start here

- `mainProgram.m` defines dataset selection, classifiers, cluster settings and diversity options.
- `runTraining.m` coordinates experiments.
- `ensembleSelection.m` and `fusion.m` select and combine predictions.
- The individual diversity files implement measures such as disagreement, double fault, correlation and Q statistics.

Inspect the dataset layout and required MATLAB toolboxes before running `mainProgram`. The entry point uses `parfor`.

## Citation retained from the original repository

Jan, Z. M. and Verma, B. (2020). Multiple Strong and Balanced Clusters based Ensemble of Deep Learners. Pattern Recognition, article 107420.

The repository name and this original citation refer to different publication labels. Confirm the intended paper/version before citing results from this checkout.

## Attribution and status

Historical research implementation. Retain third-party author notices in the optimisation source files. Experiments have not been rerun as part of the documentation cleanup.

[Profile and project directory](https://github.com/zohaibjan)
