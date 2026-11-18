"""Deterministic SHAP explanations for the versioned claim-risk artifact."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from .data_validation import validate_claims


def _estimator(bundle: dict[str, Any]) -> Any:
    model = bundle["model"]
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator
    if hasattr(model, "named_steps"):
        return model.named_steps["model"]
    return model


def _fitted_base_model(bundle: dict[str, Any]) -> Any:
    model = bundle["model"]
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def transformed_matrix(bundle: dict[str, Any], claims: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessing = _fitted_base_model(bundle).named_steps["preprocessing"]
    matrix = preprocessing.transform(claims)
    return np.asarray(matrix), list(preprocessing.named_steps["preprocessing"].get_feature_names_out())


def explain_claims(bundle: dict[str, Any], claims: pd.DataFrame) -> dict[str, Any]:
    """Return aligned SHAP values plus positive/negative local contributions."""

    report = validate_claims(claims, require_target=False)
    if report.errors:
        raise ValueError("Invalid claim input: " + "; ".join(report.errors))
    model = bundle["model"]
    matrix, names = transformed_matrix(bundle, claims)
    estimator = _estimator(bundle)
    try:
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(matrix)
    except (ValueError, TypeError, NotImplementedError):
        if hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, matrix)
            raw = explainer.shap_values(matrix)
        else:
            explainer = shap.Explainer(estimator.predict_proba, matrix, algorithm="permutation")
            raw = explainer(matrix).values
    if isinstance(raw, list):
        values = np.asarray(raw[1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, 1]
    if values.shape != matrix.shape:
        raise ValueError(f"SHAP shape {values.shape} does not match transformed matrix {matrix.shape}")
    global_importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    local = []
    for row_index, row_values in enumerate(values):
        contribution = pd.DataFrame({"feature": names, "shap_value": row_values})
        local.append({
            "row_index": row_index,
            "positive": contribution[contribution.shap_value > 0].sort_values("shap_value", ascending=False).head(5).to_dict("records"),
            "negative": contribution[contribution.shap_value < 0].sort_values("shap_value").head(5).to_dict("records"),
        })
    return {"values": values, "feature_names": names, "global_importance": global_importance, "local": local}


def save_explainability_reports(bundle: dict[str, Any], claims: pd.DataFrame, report_dir: str) -> dict[str, Any]:
    """Create global SHAP plots and one deterministic local explanation."""

    output = explain_claims(bundle, claims)
    output_path = __import__("pathlib").Path(report_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output["global_importance"].to_csv(output_path / "shap_global_importance.csv", index=False)
    matrix, _ = transformed_matrix(bundle, claims)
    shap.summary_plot(output["values"], matrix, feature_names=output["feature_names"], plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(output_path / "shap_summary_bar.png", dpi=160, bbox_inches="tight"); plt.close()
    shap.summary_plot(output["values"], matrix, feature_names=output["feature_names"], show=False, max_display=20)
    plt.tight_layout(); plt.savefig(output_path / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight"); plt.close()
    probability = float(bundle["model"].predict_proba(claims.iloc[[0]])[0, 1])
    local = output["local"][0]
    local_report = {
        "probability": probability,
        "risk_category": risk_category(probability, float(bundle["threshold"])),
        "positive_contributions": local["positive"],
        "negative_contributions": local["negative"],
        "plain_language": plain_language_explanation(probability, local, float(bundle["threshold"])),
        "note": "SHAP contributions describe model associations, not causal effects or proof of fraud.",
    }
    (output_path / "individual_explanation.json").write_text(__import__("json").dumps(local_report, indent=2, default=float), encoding="utf-8")
    return {"global_importance": output["global_importance"], "individual": local_report}


def risk_category(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= max(0.30, threshold / 2):
        return "Medium"
    return "Low"


def plain_language_explanation(probability: float, local: dict[str, Any], threshold: float) -> str:
    category = risk_category(probability, threshold)
    positive = ", ".join(item["feature"] for item in local["positive"][:3]) or "no positive contribution exceeded zero"
    negative = ", ".join(item["feature"] for item in local["negative"][:3]) or "no negative contribution exceeded zero"
    return (
        f"The model assigned {probability:.1%} estimated fraud risk ({category}). "
        f"The strongest factors increasing the score were {positive}; factors lowering it included {negative}. "
        "This is a model-based risk assessment for investigation prioritisation, not proof that fraud occurred."
    )

