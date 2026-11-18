# Advanced Model Optimization

Checkpoint F experiments were run without modifying the production artifact
`models/claim_risk_pipeline_v2.joblib`, its threshold (`0.18`), the raw dataset,
or the dashboard.

## Protocol

- Same stratified 80/20 split as Checkpoint C (`random_state=42`).
- Five-fold stratified CV on the 800-row training partition.
- All preprocessing and feature engineering remained inside estimator
  pipelines.
- HGB randomized search used 12 parameter combinations and average precision as
  the CV objective.
- Thresholds were selected on a 160-row internal validation partition only.
- The 200-row held-out test set was evaluated after choices were frozen and was
  not used to tune the search or threshold.

## HGB search

Best parameters:

```text
learning_rate=0.12
max_iter=150
max_leaf_nodes=63
max_depth=3
min_samples_leaf=10
l2_regularization=0.0
```

The tuned model's CV PR-AUC was `0.703637 ± 0.049364`, versus
`0.671259 ± 0.059392` for the current default HGB candidate.

## Result

The tuned model did not demonstrate a reliable production improvement. Its
held-out PR-AUC was `0.535823`, F1 was `0.700855`, and accuracy was `0.825`.
The held-out result is descriptive only because the tuning decision was based
on CV/validation evidence. The current calibrated model remains the production
artifact.

Full outputs are in [reports/checkpoint_f/](../reports/checkpoint_f/).

## Threshold analysis

For tuned HGB, the internal validation set selected threshold `0.15` by maximum
F1 with recall and precision tie-breakers. At that threshold, validation
accuracy was 0.850, precision 0.660, recall 0.825, F1 0.733, and specificity
0.858. At validation threshold 0.30, accuracy was 0.863 and precision 0.714,
but recall fell to 0.750. This illustrates the fraud-screening trade-off and
was not selected using the held-out test set.

