# Statistical Validation

## Held-out metrics and uncertainty

The frozen production model was evaluated once on the original stratified 200-row test split (151 non-fraud, 49 fraud). The point estimates were accuracy `0.815`, precision `0.583`, recall `0.857`, F1 `0.694`, ROC-AUC `0.835`, PR-AUC `0.535`, Brier `0.123`, and specificity `0.801`.

The intervals below are percentile 95% intervals from 5,000 stratified bootstrap resamples. Non-fraud and fraud test rows were resampled separately, preserving the observed class counts in every resample. This quantifies sampling uncertainty conditional on this split; it is not an external validation interval.

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Accuracy | 0.815 | 0.760–0.865 |
| Precision | 0.583 | 0.506–0.672 |
| Recall | 0.857 | 0.755–0.939 |
| F1 | 0.694 | 0.621–0.769 |
| ROC-AUC | 0.835 | 0.767–0.897 |
| PR-AUC | 0.535 | 0.448–0.677 |
| Brier score | 0.123 | 0.099–0.148 |
| Specificity | 0.801 | 0.735–0.861 |

Because only 49 fraud cases are in the held-out set, precision, recall, F1, and PR-AUC remain sensitive to a small number of claims. The bootstrap treats the observed test records as the sampling population and does not correct for dataset shift, label noise, dependence between claims, or selection effects. No significance test was used, and no claim of statistically significant superiority is made.

## CV versus test PR-AUC

The five-fold training-partition PR-AUC for base HGB was `0.671 ± 0.059`, with fold values `0.658, 0.743, 0.716, 0.670, 0.570`. The calibrated production test PR-AUC was `0.535`, a difference of approximately `0.137`. The comparison is not strictly like-for-like: the CV values are uncalibrated base-estimator fold scores, while the final test score is from the isotonic-calibrated production wrapper. Isotonic calibration can change probability ties and ranking. In addition, the test set contains only 49 fraud examples and represents one split. The uncalibrated HGB test PR-AUC was `0.558`, still below the CV mean, consistent with split variability.

## Calibration limitations

The production calibration method is isotonic. Calibration selection used a validation partition of the training data, and the saved final wrapper was then refit on the complete 800-row training partition. The isotonic calibration sample available for selection was approximately 160 validation records, with about 40 fraud records; this is small for a flexible non-parametric calibrator. The Brier score is reported, but the results do not justify claiming reliable real-world probability calibration. Recalibration should be reconsidered with substantially more representative data.

## Reproducibility controls

The audit uses seed `42`, a stratified 80/20 split, and five-fold stratified CV with seed `42`. The test set was not used for threshold selection, calibration selection, or feature selection. All reported comparisons are observational experiments on the same fixed split; repeated external or temporal validation is still required.

Supporting files: [heldout_bootstrap_ci.csv](../reports/checkpoint_g/heldout_bootstrap_ci.csv), [hgb_cv_fold_metrics.csv](../reports/checkpoint_g/hgb_cv_fold_metrics.csv), and [audit_config.json](../reports/checkpoint_g/audit_config.json).
