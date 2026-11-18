# Final Model Selection

The production model remains the Checkpoint C artifact:

```text
models/claim_risk_pipeline_v2.joblib
```

It is HistGradientBoosting with isotonic calibration and frozen threshold
`0.18`.

## Decision

No Checkpoint F candidate demonstrated a reliable improvement that justified
replacing the current artifact:

- Tuned HGB improved CV PR-AUC but not the held-out target range.
- XGBoost reached 84.5% held-out accuracy and 0.730 F1, but its CV PR-AUC was
  lower than tuned HGB and the current HGB candidate.
- Extra Trees reached 84.0% held-out accuracy but lower precision and F1 than
  the target, with lower CV PR-AUC.
- Feature ablations that removed hobby information substantially degraded CV
  PR-AUC.

The apparent XGBoost test improvement is therefore treated as a hypothesis for
future external/temporal validation, not as evidence for a production swap.
No new model artifact was created and the dashboard was not changed.

