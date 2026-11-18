# Experiment Comparison

All test results below use the same untouched 200-row test set. Model selection
was based on CV and internal validation, not on the test ranking.

| Model | CV PR-AUC +/- SD | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current calibrated production HGB | 0.671 +/- 0.059 | 0.815 | 0.583 | 0.857 | 0.694 | 0.835 | 0.535 | 0.123 |
| Tuned HGB | 0.704 +/- 0.049 | 0.825 | 0.603 | 0.837 | 0.701 | 0.824 | 0.536 | 0.149 |
| Default HGB | 0.671 +/- 0.059 | 0.820 | 0.594 | 0.837 | 0.695 | 0.847 | 0.558 | 0.151 |
| Logistic Regression balanced | 0.667 +/- 0.063 | 0.835 | 0.648 | 0.714 | 0.680 | 0.830 | 0.532 | 0.148 |
| XGBoost | 0.647 +/- 0.067 | 0.845 | 0.636 | 0.857 | 0.730 | 0.841 | 0.568 | 0.134 |
| Extra Trees balanced | 0.631 +/- 0.049 | 0.840 | 0.627 | 0.857 | 0.724 | 0.850 | 0.560 | 0.134 |
| Random Forest balanced | 0.603 +/- 0.056 | 0.820 | 0.592 | 0.857 | 0.700 | 0.850 | 0.551 | 0.145 |

XGBoost reached the experimental accuracy range and improved held-out F1 over
the current calibrated baseline, but its cross-validation PR-AUC was below
tuned HGB and below the current HGB CV result. With only 49 fraud cases in the
test partition, this is not sufficient evidence to replace production.

CatBoost and LightGBM were checked and were not installed. They were not added
just to increase the model count.

## Calibration comparison for the current model

Internal validation Brier scores were `0.128` uncalibrated, `0.122` with
sigmoid calibration, and `0.114` with isotonic calibration. Isotonic was kept in
the current artifact, with the documented warning that 160 validation rows are
small for a flexible calibration map.

## Class imbalance

The previous SMOTE experiment used fold-local SMOTE and achieved CV PR-AUC
`0.668`. It did not outperform the tuned HGB result. Class weighting was
therefore not treated as automatically superior or inferior; model-specific
results are reported above.
