from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_dashboard_starts_without_training_or_external_services():
    app = AppTest.from_file(Path(__file__).parents[1] / "app" / "app.py", default_timeout=30).run()
    assert not app.exception
    assert app.title
    assert app.segmented_control


def test_dashboard_sections_and_single_claim_form_render():
    app = AppTest.from_file(Path(__file__).parents[1] / "app" / "app.py", default_timeout=60).run()
    for section in ["Claim risk prediction", "Batch screening", "Explainability", "Model performance", "Data quality"]:
        app.get_by_key("dashboard_section").set_value(section).run()
        assert not app.exception, section
    app.get_by_key("dashboard_section").set_value("Claim risk prediction").run()
    app.form_submit_button[0].click().run()
    assert not app.exception

