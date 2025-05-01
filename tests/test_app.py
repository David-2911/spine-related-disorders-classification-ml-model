import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run import create_app


def test_create_app():
    """
    Test that the Flask application is created successfully.
    """
    app = create_app()
    assert app is not None
    assert app.name == "app.app"  # Ensure the app name is correct
    assert "routes" in app.blueprints  # Ensure the routes blueprint is registered
