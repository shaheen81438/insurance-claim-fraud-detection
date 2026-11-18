# Data Dictionary

Source: `data/insurance_claims.csv`  
Shape: 1,000 rows × 39 columns  
Target: `fraud_reported` (`Y` = fraud reported, `N` = not reported)

## Raw columns

| Column | Role / type | Availability and handling |
|---|---|---|
| `months_as_customer` | Numeric | Customer tenure input |
| `age` | Numeric | Policyholder age input |
| `policy_number` | Identifier | Dropped; not a generalizable predictive feature |
| `policy_bind_date` | Date | Used to derive `policy_age_days`, then dropped |
| `policy_state` | Categorical | One-hot encoded |
| `policy_csl` | Categorical | One-hot encoded |
| `policy_deductable` | Numeric | Median-imputed and scaled |
| `policy_annual_premium` | Numeric | Median-imputed and scaled |
| `umbrella_limit` | Numeric | Median-imputed and scaled; sentinel semantics require monitoring |
| `insured_zip` | Identifier/geography code | Dropped to avoid memorization and sparse geography effects |
| `insured_sex` | Categorical | One-hot encoded |
| `insured_education_level` | Categorical | One-hot encoded |
| `insured_occupation` | Categorical | One-hot encoded |
| `insured_hobbies` | Categorical | One-hot encoded; association is not causation |
| `insured_relationship` | Categorical | One-hot encoded |
| `capital-gains` | Numeric | Median-imputed and scaled |
| `capital-loss` | Numeric | Median-imputed and scaled |
| `incident_date` | Date | Used to derive temporal features, then dropped |
| `incident_type` | Categorical | One-hot encoded |
| `collision_type` | Categorical | `?` converted to missing, then imputed as `Unknown` |
| `incident_severity` | Categorical | One-hot encoded |
| `authorities_contacted` | Categorical | One-hot encoded |
| `incident_state` | Categorical | One-hot encoded |
| `incident_city` | Categorical | One-hot encoded |
| `incident_location` | Identifier/free-form location | Dropped to avoid memorization |
| `incident_hour_of_the_day` | Numeric | Median-imputed and scaled; also used for odd-hour flag |
| `number_of_vehicles_involved` | Numeric | Median-imputed and scaled |
| `property_damage` | Categorical | `?` converted to missing, then imputed as `Unknown` |
| `bodily_injuries` | Numeric | Median-imputed and scaled |
| `witnesses` | Numeric | Median-imputed and scaled; zero contributes to missing-document count |
| `police_report_available` | Categorical | `?` converted to missing, then imputed as `Unknown` |
| `total_claim_amount` | Numeric | Median-imputed and scaled; used in claim ratios |
| `injury_claim` | Numeric | Median-imputed and scaled; used in claim ratios |
| `property_claim` | Numeric | Median-imputed and scaled; used in claim ratios |
| `vehicle_claim` | Numeric | Median-imputed and scaled; used in claim ratios |
| `auto_make` | Categorical | One-hot encoded |
| `auto_model` | Categorical | One-hot encoded |
| `auto_year` | Numeric | Median-imputed and scaled; used for vehicle age |
| `fraud_reported` | Target | Mapped to 1/0 for training and excluded from features |

## Missing-value policy

The literal `?` is treated as missing only after schema validation records its
frequency. Categorical missing values are imputed as the explicit category
`Unknown`; numerical missing values are median-imputed within the fitted
training pipeline. No raw rows are deleted and the original CSV is preserved.

## Leakage and availability policy

The target, identifier-like fields, and raw date fields are not passed directly
to the estimator. Dates are available at claim processing time and are converted
to deterministic duration/age features. The high-value threshold is fitted only
on each training partition by `ClaimFeatureEngineer`, then reused unchanged.

