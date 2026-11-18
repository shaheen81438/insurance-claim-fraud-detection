# Final Project Limitations

## Data and sampling

The dataset has 1,000 records and 247 fraud labels. The held-out evaluation has 200 records and 49 fraud labels. This is adequate for a reproducible academic experiment but too small to support claims of production effectiveness, stable subgroup performance, or generalization to other insurers, geographies, time periods, or fraud mechanisms. There is no external or temporal validation in this audit.

## Model uncertainty

The frozen production HGB/isotonic model reproduced its reported test metrics, but bootstrap intervals remain wide: precision `0.506–0.672`, recall `0.755–0.939`, and PR-AUC `0.448–0.677` at 95% percentile bootstrap intervals. Cross-validation also shows meaningful fold-to-fold variation. XGBoost was slightly better on this held-out accuracy comparison, but had lower mean CV PR-AUC and higher CV dispersion; this is not sufficient evidence to replace production.

## Calibration and threshold

The threshold `0.18` is frozen and was selected using training-side validation, not the held-out test set. It encodes a fraud-screening trade-off and is not a universal decision rule. Isotonic calibration was trained with a small calibration sample and should not be interpreted as proven probability calibration.

## Feature and fairness risks

The model learns associations, not causes. `incident_severity_Major Damage`, `insured_hobbies_chess`, and `insured_hobbies_cross-fit` show strong or non-trivial importance in this sample. Hobby indicators in particular could be spurious, proxy variables, or unstable under distribution change. The audit confirms they are present in the source records, but does not establish acceptable use, fairness, or causal meaning. Subgroup fairness, disparate impact, and protected-attribute analysis were not possible from the available project evidence.

## Data quality

- Three categorical fields contain missing-value tokens: `collision_type`, `property_damage`, and `police_report_available`. The preprocessing policy maps these to an explicit `Unknown` category and uses an unknown-safe one-hot encoder.
- One record has `incident_date` before `policy_bind_date`. It is surfaced as a validation warning rather than silently rewritten.
- One record has the sentinel `umbrella_limit=-1000000`. The feature engineering path converts this sentinel to missing before numeric imputation; the sentinel is retained in validation reporting.
- Rare categories and unseen prediction-time categories are handled by the categorical imputer/encoder policy, but low-frequency effects remain a modeling risk.

## Operational and security limitations

The dashboard is an inference-only interface for the saved artifact and is not a fraud determination system. It requires input validation and human review. No public deployment was performed or claimed in this audit. Dataset licensing, privacy, retention, access control, monitoring, drift detection, label feedback, and incident response require review before any real-world use.

## Required next validation

Before publication as a substantive fraud-detection claim, obtain a larger, independently sourced and temporally representative dataset; pre-register evaluation and threshold criteria; evaluate calibration and subgroup behavior; investigate the flagged feature associations; and validate with prospective human-review outcomes. Until then, report this project as a small-sample, leakage-controlled research prototype.
