"""Premium Streamlit workspace for insurance claim risk assessment."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_validation import RAW_COLUMNS, TARGET_COLUMN, validate_claims
from src.explainability import explain_claims, plain_language_explanation, risk_category
from src.gemini_explanation import GeminiResult, generate_investigator_summary
from src.predict import load_bundle, predict_claims

ARTIFACT = ROOT / "models" / "claim_risk_pipeline_v2.joblib"
DATASET = ROOT / "data" / "insurance_claims.csv"
REPORTS = ROOT / "reports" / "checkpoint_c_v2"
THRESHOLD = 0.18

st.set_page_config(
    page_title="Claim intelligence",
    page_icon=":material/shield_with_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Apply the high-contrast visual system used throughout the workspace."""

    st.markdown(
        """
        <style>
        :root {
            --navy-950: #071426;
            --navy-900: #0b1b33;
            --navy-800: #112b4d;
            --ink: #132238;
            --muted: #53657d;
            --line: #d9e2ee;
            --canvas: #f4f7fb;
            --card: #ffffff;
            --blue: #2368d1;
            --blue-soft: #e8f0ff;
            --red: #c83b4b;
            --amber: #b56a00;
            --green: #17834b;
        }

        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: var(--navy-950); border-right: 1px solid #1f3a5d; }
        [data-testid="stSidebar"] * { color: #f4f7fb !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c6d5e8 !important; }
        [data-testid="stSidebar"] .stCaption p { color: #a9bdd5 !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] { background: #102844; border-color: #315277; }
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button { color: #dbe8f7 !important; }
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-checked="true"] { background: #2368d1; color: #fff !important; }
        [data-testid="stSidebar"] hr { border-color: #294668; }

        .block-container { max-width: 1500px; padding: 2.25rem 3.25rem 3.5rem; }
        h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.025em; }
        h1 { font-size: 2rem !important; line-height: 1.15 !important; }
        h2 { font-size: 1.45rem !important; }
        h3 { font-size: 1.05rem !important; }
        p, li, label, [data-testid="stCaptionContainer"] p, .stMarkdown { color: var(--ink); }
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--ink) !important; font-weight: 600; }
        input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div { color: var(--ink) !important; background: #fff !important; border-color: var(--line) !important; }
        [data-baseweb="select"] span { color: var(--ink) !important; }
        [data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(25, 55, 95, .06); min-height: 112px; }
        [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 700; font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; }
        [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800; }
        [data-testid="stMetricDelta"] { color: var(--muted) !important; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #fff; }
        [data-testid="stDataFrame"] * { color: var(--ink); }
        .stButton > button, [data-testid="stFormSubmitButton"] button { border-radius: 9px; border: 1px solid #1b5ab9; background: var(--blue); color: #fff !important; font-weight: 700; min-height: 2.6rem; }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background: #174f9f; border-color: #174f9f; }
        .stDownloadButton > button { border-radius: 9px; border: 1px solid var(--line); background: #fff; color: var(--ink) !important; font-weight: 700; }
        [data-testid="stAlert"] { border-radius: 11px; }
        .brand-mark { display: flex; align-items: center; gap: .7rem; padding: .3rem 0 1.25rem; }
        .brand-icon { display: grid; place-items: center; width: 2.25rem; height: 2.25rem; border-radius: 10px; background: #2b78e4; color: #fff; font-weight: 900; font-size: 1.1rem; }
        .brand-name { color: #fff; font-size: 1.02rem; font-weight: 800; line-height: 1.1; }
        .brand-subtitle { color: #a9bdd5; font-size: .72rem; margin-top: .18rem; }
        .eyebrow { color: var(--blue); font-size: .73rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .35rem; }
        .page-subtitle { color: var(--muted); font-size: .95rem; margin: -.45rem 0 1.45rem; }
        .hero { background: linear-gradient(110deg, #fff 0%, #edf4ff 100%); border: 1px solid #cfddf1; border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.25rem; box-shadow: 0 10px 28px rgba(25, 55, 95, .06); }
        .hero h1 { margin: 0; }
        .hero p { color: var(--muted); margin: .55rem 0 0; max-width: 860px; }
        .section-kicker { color: var(--muted); font-size: .75rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; margin: 1.4rem 0 .6rem; }
        .risk-badge { display: inline-block; border-radius: 999px; padding: .35rem .72rem; font-size: .78rem; font-weight: 800; letter-spacing: .04em; }
        .risk-high { background: #ffe7ea; color: #a42536; }
        .risk-medium { background: #fff2d8; color: #8c5200; }
        .risk-low { background: #ddf7e9; color: #11653b; }
        .status-pill { display: inline-block; border-radius: 999px; background: #e0f6ea; color: #106239; padding: .28rem .55rem; font-size: .72rem; font-weight: 800; }
        .muted-card { background: #eef3f9; border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; color: var(--muted); }
        @media (max-width: 900px) { .block-container { padding: 1.4rem 1rem 2.5rem; } h1 { font-size: 1.7rem !important; } }
        @media (prefers-color-scheme: dark) { .stApp { background: #f4f7fb; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_bundle():
    if not ARTIFACT.exists():
        raise FileNotFoundError(f"Versioned model artifact not found: {ARTIFACT}")
    return load_bundle(ARTIFACT)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET)


def get_gemini_api_key() -> str | None:
    try:
        value = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def valid_inference_rows(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    bind = pd.to_datetime(frame["policy_bind_date"], errors="coerce")
    incident = pd.to_datetime(frame["incident_date"], errors="coerce")
    return frame.loc[incident >= bind].reset_index(drop=True)


@st.cache_data(show_spinner="Scoring the verified source claims…")
def score_source(_bundle, data: pd.DataFrame) -> pd.DataFrame:
    return predict_claims(_bundle, valid_inference_rows(data))


@st.cache_data(show_spinner="Preparing feature explanations…")
def explain_source(_bundle, data: pd.DataFrame) -> dict:
    return explain_claims(_bundle, valid_inference_rows(data))


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def risk_badge(category: str) -> None:
    css = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(category, "risk-low")
    st.markdown(f'<span class="risk-badge {css}">{category} risk</span>', unsafe_allow_html=True)


def chart_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=48, b=18),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#132238", size=12),
        title_font=dict(color="#132238", size=16),
        legend=dict(font=dict(color="#132238")),
        hoverlabel=dict(bgcolor="#132238", font_color="#ffffff"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9e2ee", tickfont=dict(color="#53657d"), title_font=dict(color="#53657d"))
    fig.update_yaxes(gridcolor="#e6edf5", zeroline=False, tickfont=dict(color="#53657d"), title_font=dict(color="#53657d"))
    return fig


def render_gemini_summary(probability: float, category: str, threshold: float, local: dict) -> None:
    result: GeminiResult = generate_investigator_summary(
        api_key=get_gemini_api_key(),
        probability=probability,
        risk_category=category,
        threshold=threshold,
        positive=local["positive"],
        negative=local["negative"],
    )
    if result.used_gemini:
        st.markdown('<div class="section-kicker">AI-assisted review note</div>', unsafe_allow_html=True)
        st.container(border=True).write(result.text)
        st.caption("Generated from the saved model probability and verified SHAP contributions; not proof of fraud.")
        return
    messages = {
        "missing_key": "Gemini is not configured. The verified rule-based explanation remains active.",
        "sdk_missing": "The Google GenAI SDK is unavailable. The verified rule-based explanation remains active.",
        "quota_exceeded": "Gemini quota or rate limit reached. The verified rule-based explanation is shown instead.",
        "network_error": "Gemini could not be reached. The verified rule-based explanation is shown instead.",
        "empty_response": "Gemini returned no summary. The verified rule-based explanation is shown instead.",
        "api_error": "Gemini returned an error. The verified rule-based explanation is shown instead.",
    }
    st.info(messages.get(result.status, messages["api_error"]), icon=":material/info:")


def dashboard_page(bundle: dict, data: pd.DataFrame) -> None:
    page_header("Claim intelligence / dashboard", "Insurance claim risk overview", "A high-contrast command centre for investigation prioritisation using the frozen v2 model.")
    scored = score_source(bundle, data)
    counts = scored["risk_category"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    avg_probability = float(scored["fraud_probability"].mean()) if len(scored) else 0.0
    with st.container(horizontal=True):
        st.metric("Total claims", f"{len(scored):,}", help="Valid claims available for inference; one invalid source date row is excluded from scoring.")
        st.metric("High-risk claims", f"{int(counts['High']):,}")
        st.metric("Medium-risk claims", f"{int(counts['Medium']):,}")
        st.metric("Low-risk claims", f"{int(counts['Low']):,}")
        st.metric("Average fraud probability", f"{avg_probability:.1%}")
    st.caption(f"Model status: HistGradientBoosting + isotonic calibration · threshold {bundle['threshold']:.2f} · source dataset: {len(data):,} rows")

    left, right = st.columns([1, 1], gap="large")
    with left:
        risk_frame = pd.DataFrame({"Risk category": counts.index, "Claims": counts.values})
        fig = px.bar(risk_frame, x="Risk category", y="Claims", color="Risk category", color_discrete_map={"High": "#c83b4b", "Medium": "#d69122", "Low": "#17834b"}, title="Fraud risk distribution", text_auto=True)
        st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})
    with right:
        fig = px.histogram(scored, x="fraud_probability", nbins=24, color_discrete_sequence=["#2368d1"], title="Prediction probability distribution")
        fig.update_xaxes(title="Estimated fraud probability", tickformat=".0%")
        st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})

    left, right = st.columns([1, 1], gap="large")
    with left:
        amounts = valid_inference_rows(data).copy()
        amounts["total_claim_amount"] = pd.to_numeric(amounts["total_claim_amount"], errors="coerce")
        fig = px.histogram(amounts, x="total_claim_amount", nbins=28, color_discrete_sequence=["#5b8def"], title="Claim amount distribution")
        fig.update_xaxes(title="Total claim amount")
        st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})
    with right:
        explanation = explain_source(bundle, data)
        top = explanation["global_importance"].head(10).sort_values("mean_abs_shap")
        fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=["#2368d1"], title="Top model-associated risk factors")
        fig.update_xaxes(title="Mean absolute SHAP value")
        fig.update_yaxes(title="")
        st.plotly_chart(chart_layout(fig, 370), width="stretch", config={"displayModeBar": False})

    valid = valid_inference_rows(data)
    if "incident_date" in valid.columns:
        trend = valid.assign(incident_month=pd.to_datetime(valid["incident_date"], errors="coerce").dt.to_period("M").astype(str))
        trend["fraud_probability"] = scored["fraud_probability"].to_numpy()
        trend = trend.groupby("incident_month", as_index=False).agg(claims=("fraud_probability", "size"), average_risk=("fraud_probability", "mean"))
        fig = go.Figure(go.Scatter(x=trend["incident_month"], y=trend["average_risk"], mode="lines+markers", line=dict(color="#2368d1", width=3), marker=dict(size=7)))
        fig.update_layout(title="Average predicted risk by incident month", xaxis_title="Incident month", yaxis_title="Average fraud probability")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})


def upload_page() -> None:
    page_header("Workspace / data intake", "Upload claims", "Inspect a CSV before scoring it. Upload validation never changes the preserved source dataset.")
    uploaded = st.file_uploader("Drop a claims CSV here", type=["csv"], key="preview_upload", help="The target column is optional for data-quality inspection.")
    if uploaded is None:
        st.markdown('<div class="muted-card">Choose a CSV to see its filename, shape, missing values, duplicate count, validation findings, and a preview.</div>', unsafe_allow_html=True)
        return
    try:
        frame = pd.read_csv(uploaded)
    except Exception:
        st.error("The file could not be read as a CSV.", icon=":material/error:")
        return
    report = validate_claims(frame, require_target=TARGET_COLUMN in frame.columns)
    st.success(f"Loaded {uploaded.name}", icon=":material/check_circle:")
    with st.container(horizontal=True):
        st.metric("Records", f"{len(frame):,}")
        st.metric("Columns", f"{len(frame.columns):,}")
        st.metric("Missing cells", f"{int(frame.replace('?', pd.NA).isna().sum().sum()):,}")
        st.metric("Duplicate rows", f"{report.duplicate_rows:,}")
    if report.errors:
        st.error("Validation requires attention: " + "; ".join(report.errors), icon=":material/error:")
    else:
        st.success("Schema checks passed for this upload.", icon=":material/verified:")
    if report.warnings:
        st.warning("; ".join(report.warnings), icon=":material/warning:")
    st.markdown('<div class="section-kicker">Data preview</div>', unsafe_allow_html=True)
    st.dataframe(frame.head(12), hide_index=True, width="stretch", height=360, alt="Uploaded claims preview")


def prediction_form(bundle: dict, data: pd.DataFrame) -> None:
    page_header("Workspace / assessment", "Fraud prediction", "Assess one claim with the frozen production pipeline and inspect the verified reasons behind its score.")
    st.warning("A high-risk result is an indicator for investigation, not proof that fraud occurred.", icon=":material/warning:")
    sample_frame = data.copy()
    bind = pd.to_datetime(sample_frame["policy_bind_date"], errors="coerce")
    incident = pd.to_datetime(sample_frame["incident_date"], errors="coerce")
    sample = sample_frame.loc[incident >= bind].iloc[0]
    categorical = {"policy_state", "policy_csl", "insured_sex", "insured_education_level", "insured_occupation", "insured_hobbies", "insured_relationship", "incident_type", "collision_type", "incident_severity", "authorities_contacted", "incident_state", "incident_city", "property_damage", "police_report_available", "auto_make", "auto_model"}
    form_values = {}
    with st.form("claim_form"):
        st.markdown('<div class="section-kicker">Claim details</div>', unsafe_allow_html=True)
        for start in range(0, len(RAW_COLUMNS[:-1]), 4):
            cols = st.columns(4)
            for col, field in zip(cols, RAW_COLUMNS[:-1][start:start + 4]):
                value = sample[field]
                if field in ("policy_bind_date", "incident_date"):
                    form_values[field] = col.text_input(field.replace("_", " ").title(), str(value), help="Use YYYY-MM-DD")
                elif field in ("policy_number", "insured_zip", "incident_location"):
                    form_values[field] = col.text_input(field.replace("_", " ").title(), str(value))
                elif field in categorical:
                    options = sorted(set(data[field].dropna().astype(str)) | {"Unknown"})
                    current = str(value) if str(value) in options else options[0]
                    form_values[field] = col.selectbox(field.replace("_", " ").title(), options, index=options.index(current))
                else:
                    form_values[field] = col.number_input(field.replace("_", " ").title(), value=float(value), step=1.0)
        submitted = st.form_submit_button("Assess claim", width="stretch", icon=":material/analytics:")
    if not submitted:
        return
    claim = pd.DataFrame([form_values])
    report = validate_claims(claim, require_target=False)
    if report.errors:
        st.error("Input validation failed: " + "; ".join(report.errors), icon=":material/error:")
        return
    scored = predict_claims(bundle, claim)
    probability = float(scored.iloc[0]["fraud_probability"])
    category = str(scored.iloc[0]["risk_category"])
    st.markdown('<div class="section-kicker">Assessment result</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1], gap="large")
    with c1:
        st.metric("Estimated fraud probability", f"{probability:.1%}")
        risk_badge(category)
    with c2:
        st.metric("Total claim amount", f"{float(claim['total_claim_amount'].iloc[0]):,.0f}")
    with c3:
        st.metric("Decision threshold", f"{bundle['threshold']:.2f}")
    fig = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, number={"suffix": "%"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#c83b4b" if category == "High" else "#17834b"}, "steps": [{"range": [0, bundle["threshold"] * 100], "color": "#ddf7e9"}, {"range": [bundle["threshold"] * 100, 100], "color": "#ffe7ea"}]}, title={"text": "Risk score"}))
    st.plotly_chart(chart_layout(fig, 280), width="stretch", config={"displayModeBar": False})
    explanation = explain_claims(bundle, claim)
    local = explanation["local"][0]
    st.markdown('<div class="section-kicker">Why this score?</div>', unsafe_allow_html=True)
    st.write(plain_language_explanation(probability, local, bundle["threshold"]))
    with st.container(border=True):
        st.dataframe(pd.DataFrame(local["positive"] + local["negative"]), hide_index=True, width="stretch", alt="Verified local SHAP contributions")
    render_gemini_summary(probability, category, bundle["threshold"], local)


def bulk_page(bundle: dict) -> None:
    page_header("Workspace / operations", "Bulk analysis", "Score a validated claims file, filter risk categories, and download an investigation queue.")
    uploaded = st.file_uploader("Upload claims for batch screening", type=["csv"], key="bulk_upload")
    if uploaded is None:
        st.markdown('<div class="muted-card">The upload must contain the original claim fields and must not include fabricated values. The target column is ignored for prediction.</div>', unsafe_allow_html=True)
        return
    try:
        frame = pd.read_csv(uploaded)
    except Exception:
        st.error("The file could not be read as a CSV.", icon=":material/error:")
        return
    report = validate_claims(frame, require_target=False)
    if report.errors:
        st.error("The upload was rejected: " + "; ".join(report.errors), icon=":material/error:")
        return
    scored = predict_claims(bundle, frame.drop(columns=[TARGET_COLUMN], errors="ignore"))
    counts = scored["risk_category"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    with st.container(horizontal=True):
        st.metric("Processed claims", f"{len(scored):,}")
        st.metric("High risk", f"{int(counts['High']):,}")
        st.metric("Medium risk", f"{int(counts['Medium']):,}")
        st.metric("Low risk", f"{int(counts['Low']):,}")
    search, category, sort_order = st.columns([2, 1, 1])
    query = search.text_input("Search records", placeholder="Policy number or location", key="bulk_query")
    category_filter = category.selectbox("Risk category", ["All", "High", "Medium", "Low"], key="bulk_category")
    descending = sort_order.toggle("Highest risk first", value=True, key="bulk_sort")
    view = scored.copy()
    if category_filter != "All":
        view = view.loc[view["risk_category"] == category_filter]
    if query:
        mask = view.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        view = view.loc[mask]
    view = view.sort_values("fraud_probability", ascending=not descending)
    st.caption(f"Showing {len(view):,} of {len(scored):,} processed claims")
    st.dataframe(view, hide_index=True, width="stretch", height=440, column_config={"fraud_probability": st.column_config.ProgressColumn("Fraud probability", min_value=0, max_value=1, format="%.1f%%"), "risk_category": st.column_config.TextColumn("Risk category")}, alt="Bulk claim risk predictions")
    st.download_button("Download scored report", scored.to_csv(index=False), "claim_risk_scored.csv", "text/csv", width="content", icon=":material/download:")


def explainability_page(bundle: dict, data: pd.DataFrame) -> None:
    page_header("Workspace / transparency", "Explainable AI", "Inspect global model associations and the local evidence behind an individual risk estimate.")
    valid = valid_inference_rows(data)
    explanation = explain_source(bundle, data)
    st.info("SHAP contributions describe model associations on transformed features. They are not causal explanations or proof of fraud.", icon=":material/lightbulb:")
    top = explanation["global_importance"].head(15).sort_values("mean_abs_shap")
    fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=["#2368d1"], title="Global feature importance")
    fig.update_xaxes(title="Mean absolute SHAP value")
    fig.update_yaxes(title="")
    st.plotly_chart(chart_layout(fig, 450), width="stretch", config={"displayModeBar": False})
    selected = st.number_input("Claim row to explain", min_value=0, max_value=max(0, len(valid) - 1), value=0, step=1, key="explanation_row")
    local = explanation["local"][int(selected)]
    probability = float(bundle["model"].predict_proba(valid.iloc[[int(selected)]])[0, 1])
    category = risk_category(probability, bundle["threshold"])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Estimated fraud probability", f"{probability:.1%}")
        risk_badge(category)
    with c2:
        st.write(plain_language_explanation(probability, local, bundle["threshold"]))
    st.dataframe(pd.DataFrame(local["positive"] + local["negative"]), hide_index=True, width="stretch", alt="Individual SHAP contributions")
    if st.button("Generate Gemini investigator summary", key="gemini_explainability_summary", width="content", icon=":material/auto_awesome:"):
        render_gemini_summary(probability, category, bundle["threshold"], local)


def performance_page(bundle: dict) -> None:
    page_header("Workspace / validation", "Model performance", "Verified held-out metrics for the versioned HistGradientBoosting pipeline.")
    st.warning("These results come from one small held-out split and are not external validation of real-world fraud detection effectiveness.", icon=":material/warning:")
    metrics = bundle["metrics"]
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1 score"], ["accuracy", "precision", "recall", "f1"]):
        col.metric(label, f"{metrics[key]:.1%}")
    st.markdown('<div class="section-kicker">Ranking and calibration</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, label, key in zip(cols, ["ROC-AUC", "PR-AUC", "Brier score", "Specificity"], ["roc_auc", "pr_auc", "brier_score", "specificity"]):
        col.metric(label, f"{metrics[key]:.3f}")
    cm = metrics["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, x=["Predicted non-fraud", "Predicted fraud"], y=["Actual non-fraud", "Actual fraud"], color_continuous_scale=[[0, "#e8f0ff"], [1, "#2368d1"]], title="Held-out confusion matrix")
    fig.update_xaxes(side="top")
    st.plotly_chart(chart_layout(fig, 390), width="stretch", config={"displayModeBar": False})
    for filename, caption in [("roc_curve.png", "ROC curve"), ("precision_recall_curve.png", "Precision-recall curve"), ("calibration_curve.png", "Calibration curve")]:
        path = REPORTS / filename
        if path.exists():
            st.image(str(path), caption=caption, alt=caption)


def about_page(bundle: dict, data: pd.DataFrame) -> None:
    page_header("Workspace / project", "About this project", "A transparent academic prototype for explainable claim investigation prioritisation.")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("### Research scope")
        st.write("The project combines leakage-safe preprocessing, domain feature engineering, calibrated classification, and SHAP explanations for structured insurance claims.")
        st.markdown("### Guardrails")
        st.write("A prediction is a risk estimate, not proof of fraud. Human review is required. SHAP values are associative, not causal. The dataset is small and has no external or temporal validation.")
    with right:
        with st.container(border=True):
            st.markdown("### Production status")
            st.write(f"**Artifact:** `{ARTIFACT.name}`")
            st.write(f"**Model:** {bundle['selected_model_name']}")
            st.write(f"**Calibration:** {bundle['calibration_method']}")
            st.write(f"**Threshold:** {bundle['threshold']:.2f}")
            st.write(f"**Source rows:** {len(data):,}")
            st.markdown('<span class="status-pill">Inference ready</span>', unsafe_allow_html=True)


inject_styles()

try:
    bundle = get_bundle()
    dataset = get_dataset()
except Exception as exc:
    st.error(f"Dashboard could not load its versioned artifact: {exc}", icon=":material/error:")
    st.stop()

nav_options = ["Dashboard", "Upload claims", "Claim risk prediction", "Batch screening", "Explainability", "Model performance", "About project"]
nav_labels = {"Claim risk prediction": "Fraud prediction", "Batch screening": "Bulk analysis", "Explainability": "Explainable AI"}
with st.sidebar:
    st.markdown('<div class="brand-mark"><div class="brand-icon">CI</div><div><div class="brand-name">Claim intelligence</div><div class="brand-subtitle">Fraud risk workspace</div></div></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    selection = st.segmented_control("Workspace", nav_options, default="Dashboard", format_func=lambda item: nav_labels.get(item, item), key="dashboard_section", width="stretch", wrap=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<span class='status-pill'>Model online</span>", unsafe_allow_html=True)
    st.caption("v2 production bundle")
    st.caption("Risk threshold 0.18")

st.markdown('<div class="hero"><div class="eyebrow">Insurance claim fraud intelligence</div><h1>Decision support for claim investigators</h1><p>Prioritise review with a calibrated risk score, transparent feature contributions, and a workflow designed for careful human judgement.</p></div>', unsafe_allow_html=True)

if selection == "Dashboard":
    dashboard_page(bundle, dataset)
elif selection == "Upload claims":
    upload_page()
elif selection == "Claim risk prediction":
    prediction_form(bundle, dataset)
elif selection == "Batch screening":
    bulk_page(bundle)
elif selection == "Explainability":
    explainability_page(bundle, dataset)
elif selection == "Model performance":
    performance_page(bundle)
else:
    about_page(bundle, dataset)
