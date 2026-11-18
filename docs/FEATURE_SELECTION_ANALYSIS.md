# Feature Selection Analysis

The feature-selection experiments were leakage-safe ablations using the current
HGB configuration and the same five-fold stratified CV split.

| Feature set | CV PR-AUC | CV SD | CV ROC-AUC |
|---|---:|---:|---:|
| Full features | 0.6713 | 0.0594 | 0.8677 |
| Without `insured_hobbies` | 0.4961 | 0.0602 | 0.7377 |
| Compact claim context | 0.4953 | 0.0652 | 0.7401 |

The compact set removed `insured_hobbies`, `insured_occupation`,
`insured_relationship`, and `auto_model`. Both ablations performed materially
worse, so the fields were not removed solely because individual categories are
sparse.

## Audited features

| Source/category | Full count | Train | Test | Train prevalence | Test prevalence |
|---|---:|---:|---:|---:|---:|
| `insured_hobbies=chess` | 46 | 38 | 8 | 4.75% | 4.00% |
| `insured_hobbies=cross-fit` | 35 | 28 | 7 | 3.50% | 3.50% |
| `incident_severity=Major Damage` | 276 | 219 | 57 | 27.38% | 28.50% |

All are available in the raw input at prediction time. None is a target-derived
field or identifier. The hobby categories are sparse and may be spurious
correlations; the model must not be interpreted causally. Major Damage is more
stable in frequency, but can still reflect reporting or claims-handling
processes rather than fraud itself.

## Fold-level SHAP stability

Using the five training-only folds, mean absolute SHAP magnitude and rank were:

| Feature | Mean | SHAP SD | Mean rank | Rank range |
|---|---:|---:|---:|---:|
| `incident_severity_Major Damage` | 2.070 | 0.136 | 1.0 | 1–1 |
| `insured_hobbies_chess` | 0.569 | 0.077 | 2.0 | 2–2 |
| `insured_hobbies_cross-fit` | 0.331 | 0.039 | 4.8 | 3–7 |
| `policy_age_days` | 0.305 | 0.037 | 5.6 | 3–9 |
| `policy_annual_premium` | 0.276 | 0.066 | 6.4 | 4–9 |

The exact importance ordering is stable for Major Damage and chess, while the
other three features move several ranks across folds. This supports retaining
the full feature set for now while treating sparse hobby importance as a
potentially unstable association requiring external validation.

## Feature engineering

No invented columns were introduced. The existing engineered features—policy
duration, tenure-normalized claim amount, claim component ratios,
claim-to-premium ratio, vehicle age, missing-document count, incident timing,
and reconciliation gap—were retained and evaluated within the same fitted
pipeline. Their formulas and business rationale remain documented in
[FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md).

