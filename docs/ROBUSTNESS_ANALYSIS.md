# Robustness Analysis

## Scope

This is a read-only audit of the frozen production bundle [claim_risk_pipeline_v2.joblib](../models/claim_risk_pipeline_v2.joblib). The raw dataset, dashboard, threshold `0.18`, and production artifact were not changed.

The audit recreated the repository split: stratified 80/20 train/test split, `random_state=42`, with 800 training and 200 held-out records. The held-out set contains 151 non-fraud and 49 fraud claims.

## Reproducibility

- The saved bundle loaded in a fresh Python process and predicted all 200 held-out records without retraining.
- The loaded threshold was `0.18`; the loaded target was `fraud_reported`.
- Recalculated accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, specificity, and the confusion matrix matched the saved evaluation values.
- SHAP produced a `(1, 173)` output aligned with the saved transformed feature names.
- The shared preprocessing is inside the estimator pipeline. Cross-validation clones and fits that pipeline per training fold.
- Calibration was fit on the training-side fit partition using `CalibratedClassifierCV`; threshold selection was performed on the validation partition. The held-out test split was used only for final evaluation.

## Cross-validation stability

The following are base HistGradientBoosting fold results on the 800-row training partition at the frozen threshold. The final production test result additionally uses the saved isotonic calibration wrapper.

| Fold | PR-AUC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.658 | 0.603 | 0.897 | 0.722 |
| 2 | 0.743 | 0.711 | 0.821 | 0.762 |
| 3 | 0.716 | 0.708 | 0.850 | 0.773 |
| 4 | 0.670 | 0.630 | 0.725 | 0.674 |
| 5 | 0.570 | 0.564 | 0.775 | 0.653 |
| Mean ± SD | 0.671 ± 0.059 | 0.643 ± 0.065 | 0.814 ± 0.067 | 0.717 ± 0.053 |

Fold 5 is the lowest PR-AUC and F1 fold; fold 2 is highest PR-AUC. This spread is material for a 1,000-row dataset and supports reporting stability rather than relying on the best fold.

## XGBoost comparison

For a direct comparison, uncalibrated HGB and XGBoost were fit on the same 800 rows and evaluated on the same 200 rows at the common threshold `0.18`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---:|---:|---|
| HGB base | 0.845 | 0.645 | 0.816 | 0.721 | 0.847 | 0.558 | (129, 22, 9, 40) |
| XGBoost | 0.850 | 0.646 | 0.857 | 0.737 | 0.831 | 0.543 | (128, 23, 7, 42) |

The models disagreed on 5 of 200 records: 3 were actually non-fraud and 2 were fraud. XGBoost had two fewer false negatives but one more false positive in this split. Five-fold training-partition stability was HGB PR-AUC `0.671 ± 0.066` and XGBoost `0.647 ± 0.075`; HGB therefore retains the stronger mean PR-AUC and slightly lower dispersion. The five-record disagreement and one test split do not establish that XGBoost's accuracy advantage will persist.

## Feature robustness

The audited categories were present in both train and test partitions and are available in the source claim record:

| Feature/category | Full count | Train count | Test count | Train fraud rate | Test fraud rate |
|---|---:|---:|---:|---:|---:|
| `insured_hobbies_chess` | 46 | 38 | 8 | 4.75% | 4.00% |
| `insured_hobbies_cross-fit` | 35 | 28 | 7 | 3.50% | 3.50% |
| `incident_severity_Major Damage` | 276 | 219 | 57 | 27.38% | 28.50% |

Across five training folds, mean absolute SHAP magnitude and rank were:

| Feature | Mean absolute SHAP | SHAP SD | Mean rank | Rank range |
|---|---:|---:|---:|---:|
| `incident_severity_Major Damage` | 2.070 | 0.136 | 1.0 | 1–1 |
| `insured_hobbies_chess` | 0.569 | 0.077 | 2.0 | 2–2 |
| `insured_hobbies_cross-fit` | 0.331 | 0.039 | 4.8 | 3–7 |
| `policy_age_days` | 0.305 | 0.037 | 5.6 | 3–9 |
| `policy_annual_premium` | 0.276 | 0.066 | 6.4 | 4–9 |

These are associations learned from this dataset, not causal effects. The unusually strong hobby/severity signals may be sampling-specific or proxy effects. They should be monitored and validated on a larger, independently collected dataset before operational use.

## Reproducible outputs

Machine-readable outputs are in [reports/checkpoint_g/](../reports/checkpoint_g/), including fold metrics, bootstrap intervals, recalculated metrics, and the HGB/XGBoost disagreement table. The audit runner is [robustness_audit.py](../src/robustness_audit.py).
