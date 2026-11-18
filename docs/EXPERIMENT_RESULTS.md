# Experiment Results

The complete machine-readable results are in `reports/checkpoint_c_v2/`.

## Cross-validation comparison

| Candidate | Mean PR-AUC | PR-AUC SD | Mean ROC-AUC | Mean recall | Mean precision |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.671 | 0.059 | 0.868 | 0.652 | 0.693 |
| Logistic Regression + SMOTE | 0.668 | 0.067 | 0.863 | 0.732 | 0.622 |
| Logistic Regression, class-weighted | 0.667 | 0.063 | 0.866 | 0.773 | 0.630 |
| Logistic Regression | 0.667 | 0.067 | 0.864 | 0.586 | 0.678 |
| Extra Trees | 0.646 | 0.053 | 0.861 | 0.384 | 0.631 |
| Extra Trees, class-weighted | 0.631 | 0.049 | 0.857 | 0.712 | 0.628 |
| Random Forest | 0.610 | 0.036 | 0.853 | 0.162 | 0.551 |
| Random Forest, class-weighted | 0.603 | 0.056 | 0.847 | 0.738 | 0.622 |
| Decision Tree, class-weighted | 0.593 | 0.059 | 0.790 | 0.743 | 0.637 |
| Decision Tree | 0.526 | 0.061 | 0.730 | 0.642 | 0.626 |

SMOTE was applied inside an imbalanced-learn pipeline after the shared fitted
preprocessing steps and only within each training fold. It did not outperform
HistGradientBoosting on mean PR-AUC. It also did not outperform the
class-weighted logistic model on cross-validated PR-AUC, so it was not selected.

## Final choices

- Selected model: HistGradientBoostingClassifier.
- Calibration: isotonic, selected by internal validation Brier score.
- Validation Brier scores: uncalibrated 0.128, sigmoid 0.122, isotonic 0.114.
- Frozen threshold: 0.18, selected on internal validation by F1 with recall and
  precision tie-breakers.
- Artifact: `models/claim_risk_pipeline_v2.joblib`.

