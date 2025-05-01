from app.models import load_models


def test_load_models():
    """
    Test that models are loaded successfully.
    """
    models = load_models()
    assert "CatBoost" in models
    assert "AdaBoost" in models
    assert "Random Forest" in models
