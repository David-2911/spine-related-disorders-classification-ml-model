from flask import Blueprint, render_template, request, jsonify
import logging
from app.models import load_models

# Initialize models and class labels
models = load_models()
class_labels = ["Normal", "Disk Hernia", "Spondylolisthesis"]

# Create a Blueprint for routes
routes = Blueprint("routes", __name__)


@routes.route("/")
def index():
    # Round scores for display
    for model in models.values():
        model["score"] = round(model["score"], 2)
    return render_template(
        "index.html",
        best_model=max(models.values(), key=lambda x: x["score"]),
        models=models,
        class_labels=class_labels,
    )


@routes.route("/select_model", methods=["POST"])
def select_model():
    logging.info("Form data received: %s", request.form)

    # Retrieve the selected model and normalize the value
    selected_model = request.form.get("model")
    selected_model_normalized = selected_model.lower() if selected_model else None

    # Find the matching model key ignoring case
    matched_model_key = None
    if selected_model_normalized:
        for key in models.keys():
            # Support both 'extra_trees' and 'Extra Trees' for frontend-backend consistency
            if key.lower().replace(' ', '_') == selected_model_normalized:
                matched_model_key = key
                break

    # Retrieve numerical inputs
    try:
        pelvic_incidence = float(request.form.get("pelvic_incidence"))
        pelvic_tilt = float(request.form.get("pelvic_tilt"))
        lumbar_lordosis_angle = float(request.form.get("lumbar_lordosis_angle"))
        sacral_slope = float(request.form.get("sacral_slope"))
        pelvic_radius = float(request.form.get("pelvic_radius"))
        degree_spondylolisthesis = float(request.form.get("degree_spondylolisthesis"))
    except (TypeError, ValueError) as e:
        logging.error("Error parsing numerical inputs: %s", e)
        return jsonify(
            {"error": "Invalid input. Please enter valid numerical values."}
        ), 400

    # Validate the selected model
    if matched_model_key in models:
        features = [
            pelvic_incidence,
            pelvic_tilt,
            lumbar_lordosis_angle,
            sacral_slope,
            pelvic_radius,
            degree_spondylolisthesis,
        ]
        model = models[matched_model_key]["model"]
        try:
            prediction = model.predict([features])[0].item()
            predicted_label = (
                class_labels[prediction]
                if prediction < len(class_labels)
                else "Unknown"
            )
        except Exception as e:
            logging.error("Error during prediction: %s", e)
            return jsonify({"error": "Prediction failed."}), 500

        logging.info("Prediction successful: %s", predicted_label)
        return render_template(
            "result.html",
            model_name=models[matched_model_key]["name"],
            pelvic_incidence=pelvic_incidence,
            pelvic_tilt=pelvic_tilt,
            lumbar_lordosis_angle=lumbar_lordosis_angle,
            sacral_slope=sacral_slope,
            pelvic_radius=pelvic_radius,
            degree_spondylolisthesis=degree_spondylolisthesis,
            predicted_label=predicted_label,
            error=None,
        )

    logging.error("Invalid model selected.")
    return render_template("result.html", error="Invalid model selected.")


@routes.route("/api/predict", methods=["POST"])
def api_predict():
    logging.info("API request received: %s", request.json)

    # Retrieve the selected model
    selected_model = request.json.get("model")
    selected_model_normalized = selected_model.lower() if selected_model else None

    # Find the matching model key ignoring case
    matched_model_key = None
    if selected_model_normalized:
        for key in models.keys():
            # Support both 'extra_trees' and 'Extra Trees' for frontend-backend consistency
            if key.lower().replace(' ', '_') == selected_model_normalized:
                matched_model_key = key
                break

    # Retrieve numerical inputs
    try:
        pelvic_incidence = float(request.json.get("pelvic_incidence"))
        pelvic_tilt = float(request.json.get("pelvic_tilt"))
        lumbar_lordosis_angle = float(request.json.get("lumbar_lordosis_angle"))
        sacral_slope = float(request.json.get("sacral_slope"))
        pelvic_radius = float(request.json.get("pelvic_radius"))
        degree_spondylolisthesis = float(request.json.get("degree_spondylolisthesis"))
    except (TypeError, ValueError) as e:
        logging.error("Error parsing numerical inputs: %s", e)
        return jsonify(
            {"error": "Invalid input. Please provide valid numerical values."}
        ), 400

    # Validate the selected model
    if matched_model_key in models:
        features = [
            pelvic_incidence,
            pelvic_tilt,
            lumbar_lordosis_angle,
            sacral_slope,
            pelvic_radius,
            degree_spondylolisthesis,
        ]
        model = models[matched_model_key]["model"]
        try:
            prediction = model.predict([features])[0].item()
            predicted_label = (
                class_labels[prediction]
                if prediction < len(class_labels)
                else "Unknown"
            )
        except Exception as e:
            logging.error("Error during prediction: %s", e)
            return jsonify({"error": "Prediction failed. Please try again later."}), 500

        return jsonify(
            {
                "model_name": models[matched_model_key]["name"],
                "inputs": {
                    "pelvic_incidence": pelvic_incidence,
                    "pelvic_tilt": pelvic_tilt,
                    "lumbar_lordosis_angle": lumbar_lordosis_angle,
                    "sacral_slope": sacral_slope,
                    "pelvic_radius": pelvic_radius,
                    "degree_spondylolisthesis": degree_spondylolisthesis,
                },
                "predicted_label": predicted_label,
            }
        ), 200

    logging.error("Invalid model selected.")
    return jsonify(
        {"error": "Invalid model selected. Please choose a valid model."}
    ), 400
