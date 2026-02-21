import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Load ML models ────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Random Forest": "models/RandomForest.pkl",
    "XGBoost": "models/XGBoost.pkl",
    "SVM": "models/SVMClassifier.pkl",
    "Decision Tree": "models/DecisionTree.pkl",
    "Naive Bayes": "models/NBClassifier.pkl",
    "Logistic Regression": "models/LogisticRegression.pkl",
}

models = {}
for name, path in MODEL_FILES.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

# Load label encoder
label_encoder = None
if os.path.exists("models/label_encoder.pkl"):
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

# ── Load data ─────────────────────────────────────────────────────────────────
fertilizer_df = pd.read_csv("data/fertilizer.csv")
from data.fertilizerdic import fertilizer_dic

CROPS = fertilizer_df["Crop"].tolist()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/crop")
def crop():
    return render_template("crop.html")


@app.route("/fertilizer")
def fertilizer():
    return render_template("fertilizer.html", crops=CROPS)


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/api/insights")
def api_insights():
    with open("data/model_metrics.json") as f:
        return jsonify(json.load(f))


# ── API: Crop Prediction ──────────────────────────────────────────────────────
@app.route("/predict/crop", methods=["POST"])
def predict_crop():
    data = request.get_json()
    try:
        features = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"]),
        ]])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    results = {}
    for name, model in models.items():
        try:
            pred_encoded = model.predict(features)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0] if label_encoder else str(pred_encoded)
            results[name] = pred_label
        except Exception:
            results[name] = "Error"

    predictions = [v for v in results.values() if v != "Error"]
    recommendation = Counter(predictions).most_common(1)[0][0] if predictions else "Unknown"

    return jsonify({"results": results, "recommendation": recommendation})


# ── API: Fertilizer Recommendation ───────────────────────────────────────────
@app.route("/predict/fertilizer", methods=["POST"])
def predict_fertilizer():
    data = request.get_json()
    try:
        crop_name = data["crop"]
        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    row = fertilizer_df[fertilizer_df["Crop"] == crop_name]
    if row.empty:
        return jsonify({"error": "Crop not found"}), 400

    n_diff = row["N"].iloc[0] - N
    p_diff = row["P"].iloc[0] - P
    k_diff = row["K"].iloc[0] - K

    nutrient_map = {abs(n_diff): ("N", n_diff), abs(p_diff): ("P", p_diff), abs(k_diff): ("K", k_diff)}
    nutrient, diff = nutrient_map[max(nutrient_map.keys())]

    key = f"{nutrient}{'High' if diff < 0 else 'Low'}"
    return jsonify({"recommendation": fertilizer_dic[key], "key": key})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
