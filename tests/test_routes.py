import pytest
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run import create_app


@pytest.fixture
def client():
    """
    Create a test client for the Flask app.
    """
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    """
    Test the index route.
    """
    response = client.get("/")
    assert response.status_code == 200


def test_api_predict(client):
    """
    Test the /api/predict route.
    """
    response = client.post(
        "/api/predict",
        json={
            "model": "CatBoost",
            "pelvic_incidence": 45.0,
            "pelvic_tilt": 10.0,
            "lumbar_lordosis_angle": 35.0,
            "sacral_slope": 30.0,
            "pelvic_radius": 120.0,
            "degree_spondylolisthesis": 5.0,
        },
    )
    assert response.status_code == 200
    assert "predicted_label" in response.get_json()
