# Dashboard Guide

Run from the repository root:

```powershell
streamlit run app/app.py
```

The dashboard loads `models/claim_risk_pipeline_v2.joblib` and never trains at
startup. It has six sections:

1. **Overview** — dataset counts, fraud rate, selected model, threshold, and
   held-out test metrics.
2. **Claim risk prediction** — an editable, fully populated claim form with
   validation, probability, risk band, and deterministic SHAP factors.
3. **Batch screening** — validated CSV upload and downloadable predictions.
4. **Explainability** — global SHAP importance and individual claim factors.
5. **Model performance** — confusion matrix, ROC, precision-recall, and
   calibration plots from the held-out test evaluation.
6. **Data quality** — missing-token counts, duplicates, invalid dates, and
   reconciliation findings.

Risk bands use the frozen threshold 0.18 for High and the documented secondary
boundary `max(0.30, threshold / 2)` for Medium. These bands prioritize review
and are not fraud determinations. Invalid claims are rejected with validation
messages; they are not silently repaired or scored.

