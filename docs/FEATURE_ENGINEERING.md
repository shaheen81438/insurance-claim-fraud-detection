# Feature Engineering

All features are created by `src/feature_engineering.py` from columns present in
the supplied 39-column dataset. The transformer is part of the shared
scikit-learn pipeline, so training and inference use the same formulas.

| Feature | Formula | Justification |
|---|---|---|
| `policy_age_days` | `incident_date - policy_bind_date` | Short policy tenure at incident time can be operationally relevant. |
| `customer_tenure_years` | `months_as_customer / 12` | Expresses tenure in an interpretable unit. |
| `claim_amount_per_tenure_year` | `total_claim_amount / customer_tenure_years` | Relates claim size to customer tenure; zero tenure yields missing and is imputed. |
| `is_high_value_claim` | Claim amount ≥ training-only 90th percentile | Flags unusually large claims without learning the threshold from validation/test data. |
| `vehicle_claim_ratio` | `vehicle_claim / total_claim_amount` | Captures claim composition. |
| `injury_claim_ratio` | `injury_claim / total_claim_amount` | Captures claim composition. |
| `property_claim_ratio` | `property_claim / total_claim_amount` | Captures claim composition. |
| `missing_doc_count` | Missing police report + missing property-damage response + zero witnesses | Summarizes absent supporting-evidence signals. |
| `is_odd_hour_incident` | Hour ≤ 5 or ≥ 22 | Flags incidents during unusual hours; it is an association, not proof of fraud. |
| `vehicle_age_at_incident` | Incident year − vehicle year | Approximates insured vehicle age at the incident. |
| `claim_amount_to_annual_premium` | `total_claim_amount / policy_annual_premium` | Relates loss amount to the annual premium. |
| `claim_component_gap` | Total claim − (injury + property + vehicle claim) | Exposes claim amount reconciliation inconsistencies. |

Ratios with zero denominators produce missing values and are handled by the
numeric imputer. The 90th percentile is learned in `fit`, not in `transform`.
This is essential for cross-validation: each fold learns its own training-only
threshold.

## Excluded fields

`fraud_reported` is the target. `policy_number`, `insured_zip`, and
`incident_location` are identifier/free-form fields that can encourage
memorization. `policy_bind_date` and `incident_date` are not passed raw because
their useful temporal information is represented by engineered durations and
vehicle age.

