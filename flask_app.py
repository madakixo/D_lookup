from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import pandas as pd
import numpy as np
import os
import plotly.express as px
import json
import plotly
import logging
from core.utils import load_model_and_artifacts, load_dataset_and_symptoms, predict

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Paths
MODELS_DIR = "models"
DATA_PATH = "data/processed_diseases-priority.csv"

# Global variables (loaded once)
try:
    num_classes = len(pd.read_csv(DATA_PATH)["Disease"].unique())
    model, tfidf, label_encoder = load_model_and_artifacts(MODELS_DIR, num_classes)
    df_filtered, common_symptoms = load_dataset_and_symptoms(DATA_PATH, label_encoder)
    logger.info("Flask app initialized with core models and data.")
except Exception as e:
    logger.error(f"Flask initialization failed: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html', common_symptoms=common_symptoms, history=[])

@app.route('/predict', methods=['POST'])
def predict_route():
    symptoms = request.form.get('symptoms', '')
    selected_symptoms = request.form.getlist('common_symptoms')
    all_symptoms = ", ".join(selected_symptoms + [s.strip() for s in symptoms.split(",") if s.strip()])

    if not all_symptoms:
        flash("Please enter or select at least one symptom.", "error")
        return redirect(url_for('index'))

    try:
        disease, treatment, confidence, top_diseases, top_confidences = predict(
            all_symptoms, model, tfidf, label_encoder, df_filtered
        )

        # Generate Plotly chart
        fig = px.bar(
            x=top_diseases,
            y=top_confidences,
            labels={"x": "Disease", "y": "Confidence"},
            title="Top 3 Predicted Diseases",
            color=top_confidences,
            color_continuous_scale="Blues"
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        history = [{
            "Symptoms": all_symptoms,
            "Disease": disease,
            "Treatment": treatment,
            "Confidence": f"{confidence:.2%}"
        }]

        return render_template(
            'index.html',
            disease=disease,
            treatment=treatment,
            confidence=confidence,
            graph_json=graph_json,
            common_symptoms=common_symptoms,
            history=history
        )
    except Exception as e:
        logger.error(f"Prediction error in Flask: {e}")
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
