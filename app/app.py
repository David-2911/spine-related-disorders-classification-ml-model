from flask import Flask
from app.routes import routes


# Flask app initialization
def create_app():
    app = Flask(__name__, static_folder="static")
    app.register_blueprint(routes)
    return app
