import joblib  # type: ignore
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine the base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))


# Load models and their scores
def load_models():
    try:
        models = {
            "CatBoost": {
                "name": "CatBoost",
                "model": joblib.load(os.path.join(MODEL_DIR, "CatBoost_multi.pkl")),
                "score": round(0.884, 2),
            },
            "AdaBoost": {
                "name": "AdaBoost",
                "model": joblib.load(os.path.join(MODEL_DIR, "AdaBoost_multi.pkl")),
                "score": round(0.800, 2),
            },
            "Random Forest": {
                "name": "Random Forest",
                "model": joblib.load(
                    os.path.join(MODEL_DIR, "Random Forest_multi.pkl")
                ),
                "score": round(0.874, 2),
            },
        }
        return models
    except Exception as e:
        logging.error("Error loading models: %s", e)
        return {}
