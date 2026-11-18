import sys
import types

from src.gemini_explanation import build_investigator_prompt, generate_investigator_summary


def test_missing_key_uses_safe_status_without_importing_sdk():
    result = generate_investigator_summary(
        api_key=None,
        probability=0.81,
        risk_category="High",
        threshold=0.18,
        positive=[{"feature": "incident_severity_Major Damage", "shap_value": 1.2}],
        negative=[],
    )
    assert result.status == "missing_key"
    assert not result.used_gemini
    assert result.text is None


def test_prompt_contains_only_verified_prediction_and_shap_values():
    prompt = build_investigator_prompt(
        0.81,
        "High",
        0.18,
        [{"feature": "incident_severity_Major Damage", "shap_value": 1.2}],
        [{"feature": "policy_age_days", "shap_value": -0.4}],
    )
    assert "0.810000" in prompt
    assert "incident_severity_Major Damage" in prompt
    assert "policy_age_days" in prompt
    assert "not proof of fraud" in prompt
    assert "Do not invent claim facts" in prompt


def test_google_genai_success_is_wrapped_without_exposing_key(monkeypatch):
    class FakeResponse:
        text = "Review this claim because the verified model score is elevated."

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-2.5-flash"
            assert "incident_severity_Major Damage" in kwargs["contents"]
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "secret-value"}
            self.models = FakeModels()

    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_types = types.ModuleType("google.genai.types")
    fake_genai.Client = FakeClient
    fake_types.GenerateContentConfig = lambda **kwargs: kwargs
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    result = generate_investigator_summary(
        api_key="secret-value",
        probability=0.81,
        risk_category="High",
        threshold=0.18,
        positive=[{"feature": "incident_severity_Major Damage", "shap_value": 1.2}],
        negative=[],
    )
    assert result.status == "ok"
    assert result.used_gemini
    assert "elevated" in result.text
