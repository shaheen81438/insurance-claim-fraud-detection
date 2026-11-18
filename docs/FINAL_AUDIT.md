# Checkpoint E Final Technical Audit

Date: 2026-11-18  
Artifact audited: `models/claim_risk_pipeline_v2.joblib`  
Scope: reproducibility, evaluation consistency, feature audit, dashboard, hygiene, and deployment preparation

## Reproducibility checks

Passed in a fresh Python process:

- Loaded the versioned joblib bundle without retraining.
- Generated 200 held-out predictions.
- Loaded threshold: `0.18`.
- Confirmed transformed matrix and SHAP output alignment: 173 features.
- Generated SHAP output for an individual valid claim.
- Confirmed the dashboard starts without credentials or external services.

The original raw CSV and the pre-existing prototype artifacts were not modified.

## Metric recalculation

The original split was recreated with stratification and `random_state=42`.
Every saved metric matched recalculation from the loaded artifact:

| Metric | Saved | Recalculated |
|---|---:|---:|
| Accuracy | 0.815 | 0.815 |
| Precision | 0.583333 | 0.583333 |
| Recall | 0.857143 | 0.857143 |
| F1 | 0.694215 | 0.694215 |
| ROC-AUC | 0.834707 | 0.834707 |
| PR-AUC | 0.534576 | 0.534576 |
| Brier score | 0.122935 | 0.122935 |
| Specificity | 0.801325 | 0.801325 |

Held-out confusion matrix: `[[121, 30], [7, 42]]`.

## Cross-validation audit

Selected HistGradientBoosting fold-level PR-AUC values:

```text
0.657745, 0.742736, 0.716223, 0.669927, 0.569662
```

- Mean: `0.671259`
- Standard deviation: `0.059392`
- Held-out calibrated test PR-AUC: `0.534576`
- Difference: `0.136683`

The difference is not evidence of test-set tuning. The test set contains only
49 fraud cases, so one split has substantial sampling variance. The reported CV
value is from the uncalibrated candidate used for model comparison, while the
final test value is from the selected isotonic-calibrated artifact; isotonic
calibration can alter ranking ties and therefore PR-AUC. The uncalibrated test
PR-AUC was `0.558118`, which confirms calibration explains part, but not all, of
the gap.

Isotonic calibration was selected on an internal validation partition of 160
rows, not the held-out test set. Isotonic calibration can overfit small samples
and may produce stepwise probabilities; the Brier improvement supports the
choice for this experiment but does not establish population calibration.

## Feature audit

| Feature | Full count | Train count | Test count | Train prevalence | Test prevalence |
|---|---:|---:|---:|---:|---:|
| `insured_hobbies_chess` | 46 | 38 | 8 | 4.75% | 4.00% |
| `insured_hobbies_cross-fit` | 35 | 28 | 7 | 3.50% | 3.50% |
| `incident_severity_Major Damage` | 276 | 219 | 57 | 27.38% | 28.50% |

All three source fields are available at prediction time in the supplied
schema, and none is the target or an identifier. However, the hobby indicators
are sparse and likely vulnerable to sampling noise or spurious correlation;
their importance must not be interpreted causally. Incident severity is more
frequent and domain-plausible, but it can still reflect reporting or claims
handling practices rather than fraud itself. No direct leakage was found in
these features.

## Dashboard audit

The Streamlit `AppTest` audit passed for:

- Overview
- Single claim prediction form, including submit action
- Batch screening section
- Explainability section
- Model performance section
- Data quality section

The live Streamlit agent API also reached a ready state and rendered the model
performance plots. The dashboard loads the saved artifact only. It does not
train, call Gemini, call an LLM, or generate simulated predictions.

One form defect found during audit was fixed: identifier fields were initially
treated as numeric inputs, and the target column was briefly included in the
form slice. Both issues are now corrected and covered by the dashboard test.

## Security and repository hygiene

- No API keys, passwords, private keys, or credential files were found by the
  repository scan.
- No `.env` files were present.
- `.gitignore` now excludes `.venv`, Python caches, pytest/type-checker caches,
  Matplotlib caches, environment files, and private-key extensions.
- The raw dataset is preserved locally and was not altered.
- No `LICENSE`, `COPYING`, or `NOTICE` file exists in the current checkout.
  The README retains attribution to the original repository, but the dataset
  license and redistribution rights remain unresolved and must be clarified
  before publishing externally.

## Deployment preparation

- Entrypoint: `app/app.py`.
- Local command: `streamlit run app/app.py`.
- Versioned artifact size: approximately 1.54 MiB (`claim_risk_pipeline_v2.joblib`).
- Current environment verified with Python 3.14.7 and Streamlit 1.67.0.
- Requirements include the runtime and test dependencies, including
  `imbalanced-learn`, `joblib`, and `pytest`.
- No public deployment was performed, so no deployment success or public URL is
  claimed.

## Tests and commands

Final automated test result:

```text
12 passed in 9.56s
```

Key checks executed:

- Fresh-process artifact load and prediction recalculation.
- Fresh-process fold-level PR-AUC recalculation.
- Fresh-process SHAP alignment and individual explanation.
- Streamlit `AppTest` section and form audit.
- Streamlit agent API startup and Model performance rendering.
- Repository credential/license/hygiene scan.
- Full pytest suite.

## Remaining issues

1. The dataset has one invalid date-order record; inference rejects it rather
   than silently repairing it.
2. The dataset is small and lacks temporal, external, and post-investigation
   validation.
3. Isotonic calibration may be unstable with only 160 internal validation rows.
4. Sparse hobby indicators may be spurious and should be stress-tested on a
   larger, independently sampled dataset.
5. Dataset licensing and a repository license must be resolved before public
   redistribution.
6. The local Matplotlib cache emits a non-fatal permission warning; generated
   plots and tests still pass.
