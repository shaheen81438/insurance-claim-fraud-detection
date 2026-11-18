# Explainability

The selected model is explained with SHAP using the actual transformed matrix
produced by the saved preprocessing pipeline. Feature names are obtained from
the fitted `ColumnTransformer`, so one-hot categorical columns remain aligned
with the model input.

Generated artifacts include:

- `reports/checkpoint_c_v2/shap_global_importance.csv`
- `reports/checkpoint_c_v2/shap_summary_bar.png`
- `reports/checkpoint_c_v2/shap_summary_beeswarm.png`
- `reports/checkpoint_c_v2/individual_explanation.json`

The largest global mean absolute SHAP features in the valid explanation set
were `incident_severity_Major Damage`, `insured_hobbies_chess`,
`insured_hobbies_cross-fit`, `policy_age_days`, and
`policy_annual_premium`.

These are associations learned by the model, not causal effects. A feature's
importance does not demonstrate that the feature caused fraud or that a person
acted improperly. Individual explanations show positive and negative
contributions relative to the model's learned baseline and are intended to
support human review.

The one source row with an impossible date order was excluded from the global
SHAP report because the validation policy rejects it. The raw dataset was not
changed.

