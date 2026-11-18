# Initial Repository Audit

Date: 2026-11-18  
Repository: `insurance-claim-risk`  
Audit scope: read-only inspection before implementation changes

## Executive summary

This repository is a small, notebook-driven prototype for auto-insurance claim
fraud risk scoring. It contains one Streamlit application, four notebooks, three
CSV snapshots, and three serialized model artifacts. There is no `src/`, `tests/`,
`configs/`, `reports/`, or `docs/` structure yet, and the application currently
contains optional external LLM integration that conflicts with the project brief.

The raw dataset contains 1,000 records and 39 columns. The target is
`fraud_reported`, with 247 `Y` records (24.7%) and 753 `N` records (75.3%). No
exact duplicate rows were found in the raw CSV. The missing-value token `?` is
present in `collision_type` (178), `property_damage` (360), and
`police_report_available` (343).

## Repository inventory

| Area | Files | Current role |
|---|---|---|
| Application | `app/app.py` | Single-page Streamlit upload, scoring, SHAP display, and download flow |
| Raw/derived data | `data/insurance_claims.csv`, `insurance_claims_cleaned.csv`, `insurance_claims_features.csv` | Original data plus notebook-generated snapshots |
| Notebooks | `01_data_understanding_cleaning.ipynb` through `04_modeling_explainability.ipynb` | Cleaning, EDA, feature engineering, modeling, and SHAP exploration |
| Artifacts | `models/fraud_model.pkl`, `feature_columns.pkl`, `categorical_columns.pkl` | Random Forest and column metadata |
| Documentation | `README.md` | Prototype overview and setup instructions |
| Configuration | `requirements.txt`, `.gitignore` | Dependencies and basic ignore rules |

## Data audit

### Schema and target

- Raw shape: 1,000 rows × 39 columns.
- Target: `fraud_reported`; values are `Y` and `N` only.
- Positive class: 247 rows (24.7%); negative class: 753 rows (75.3%).
- Identifier-like fields include `policy_number`, `insured_zip`, and
  `incident_location`.
- Date fields are `policy_bind_date` and `incident_date`.
- No separate claim-submission date, previous-claim count, insured-asset value,
  repair estimate, or final invoice field exists in the supplied schema.

### Missingness and duplicates

- Missing values are represented by the literal `?`, not empty CSV fields.
- `collision_type`: 178 missing tokens.
- `property_damage`: 360 missing tokens.
- `police_report_available`: 343 missing tokens.
- No exact duplicate rows were found in the raw file.

### Values requiring validation

- `umbrella_limit` contains `-1000000`, which should be investigated as a
  sentinel or domain-invalid value before modeling.
- `capital-loss` is non-positive throughout the observed range, while
  `capital-gains` is non-negative; their semantics and sentinel behavior should
  be documented.
- Claim components are intended to reconcile to `total_claim_amount`; this must
  be checked in a formal validation step rather than assumed.
- Date ordering and plausible age, vehicle-year, hour, claim, and policy ranges
  need explicit validation rules.

## Existing architecture and execution flow

1. Notebook 01 loads the raw CSV, replaces `?` with missing values, fills three
   categorical fields with `Unknown`, checks duplicates, and writes the cleaned
   snapshot.
2. Notebook 02 reads the cleaned snapshot and produces exploratory plots and
   narrative interpretations. Its cells are partially executed and retain
   plots/text outputs.
3. Notebook 03 derives nine features and writes the feature snapshot.
4. Notebook 04 reads the feature snapshot, drops selected identifiers/raw dates,
   one-hot encodes all object columns with `pd.get_dummies`, performs one
   stratified 80/20 split, trains Logistic Regression and Random Forest, runs
   five-fold ROC-AUC cross-validation for Random Forest, produces SHAP plots,
   and serializes the final Random Forest plus column lists.
5. `app/app.py` reimplements feature engineering and encoding manually, aligns
   uploaded columns to the saved feature list, scores all uploaded rows, and
   displays TreeSHAP contributions.

## Existing model results found in executed notebook output

The recorded results below are historical notebook outputs, not new experiments
performed during this audit:

| Model | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.279 | 0.592 | 0.379 | 0.598 |
| Random Forest | 0.562 | 0.735 | 0.637 | 0.817 |

The notebook also records Random Forest five-fold training ROC-AUC values of
0.825, 0.838, 0.825, 0.879, and 0.740 (mean 0.821). Logistic Regression emitted
a convergence warning after 2,000 iterations, consistent with unscaled and
high-dimensional encoded inputs.

## Findings and limitations

### Data and feature engineering

- Cleaning, feature engineering, and encoding are performed as notebook side
  effects rather than reusable functions.
- The application duplicates feature engineering from Notebook 03. This creates
  a training/inference parity risk.
- `is_high_value_claim` uses the 90th percentile calculated over the entire
  dataset. A production pipeline must fit such a threshold on training data
  only, then reuse it for validation, test, and inference.
- `claim_amount_per_tenure_year` is a proxy, not a claims-per-year measure; the
  dataset has one row per claim and no prior-claim history.
- The application recomputes the high-value percentile on each uploaded file,
  which can change the feature definition at inference time.
- There is no formal schema validation, impossible-value handling, feature
  dictionary, or persisted preprocessing pipeline.

### Evaluation and leakage control

- Model selection and final-model choice are informal; Random Forest is assigned
  as final without systematic candidate comparison or validation-based threshold
  selection.
- Only a single holdout split is used for the reported test metrics.
- Cross-validation is performed only for Random Forest and uses ROC-AUC rather
  than the requested fraud-oriented PR-AUC/average precision objective.
- No hyperparameter search, calibration, uncertainty assessment, subgroup
  evaluation, drift analysis, or investigation-capacity threshold policy exists.
- One-hot encoding is performed before the split. Although category mapping is
  not target-derived, fitting preprocessing before the split prevents a clean
  train-only preprocessing boundary and should be corrected.
- No resampling experiments are implemented. If resampling is added, it must be
  inside fold-local imbalanced-learn pipelines.

### Explainability and application

- SHAP is implemented only for the Random Forest path and is not packaged as a
  reusable explanation service.
- The app calls Gemini through an HTTP API when a key is present and describes
  the output as AI-generated. This directly violates the brief's prohibition on
  LLMs/external AI APIs and must be removed.
- Risk thresholds are hardcoded at 0.30 and 0.60 in the app, rather than learned
  from validation data or an explicit investigation-capacity policy.
- The app is a single page; the requested multi-page dashboard, model monitoring,
  data-quality analysis, model-performance page, and batch-validation workflow
  are not yet present.
- Uploaded CSV columns are not validated before scoring. Missing required fields
  can silently result in zero-filled aligned columns, which is unsafe for a
  claims workflow.
- Ground-truth performance in the app uses a hardcoded 0.5 threshold rather than
  the model's operating threshold and does not guard all metric edge cases.

### Reproducibility and engineering

- The serialized artifacts do not include a complete preprocessing pipeline,
  selected threshold, model metadata, training fingerprint, configuration, or
  library versions.
- The dependency file lacks pinned versions and does not include
  `imbalanced-learn`, which would be needed for leakage-safe resampling studies.
- There are no automated tests, build/training scripts, structured reports, or
  logging configuration.
- The README contains claims and setup instructions for Gemini and a live app;
  these will need revision after implementation and verification.
- The working tree was clean at audit time. The repository is on `main` at
  commit `5ceb256` and tracks `origin/main`.

## Proposed implementation direction

1. Preserve the raw CSV and existing artifacts; write new outputs to dedicated
   `reports/`, `models/`, and `data/processed/` locations or use versioned names.
2. Move schema checks, feature engineering, and preprocessing into reusable
   modules with a single fitted train/inference path.
3. Use a stratified train/validation/test design, fit all preprocessing on the
   training partition, and compare class weighting against fold-local resampling
   only if the dependency is available.
4. Compare a justified set of baseline and tree/boosting candidates using
   stratified cross-validation optimized for PR-AUC, then select an operating
   threshold on validation or out-of-fold predictions.
5. Evaluate once on the untouched test partition and generate reproducible
   metrics/plots from that evaluation.
6. Persist the complete model bundle, deterministic SHAP explanations, risk
   policy, metadata, and training fingerprint.
7. Replace the current single-page app with validated deterministic pages while
   clearly labeling high risk as an investigation signal, not a fraud finding.
8. Add focused tests and documentation, then run the complete local verification
   flow before claiming completion.

## Audit commands and evidence

Read-only commands used:

- `rg --files -g '!*.pyc' -g '!__pycache__/**'`
- PowerShell `Get-Content` for `README.md`, `requirements.txt`, `.gitignore`, and
  `app/app.py`
- PowerShell JSON inspection of all notebook cells and outputs
- PowerShell CSV schema, counts, missing-token, duplicate, cardinality, and range
  checks using `Import-Csv`
- `git status --short --branch` and `git log --oneline --decorate -5`

No model was retrained and no project source, dataset, or existing artifact was
overwritten during Checkpoint A.
