import pytest
import pandas as pd
import numpy as np
import torch
import os
import pickle
from core.utils import load_model_and_artifacts, load_dataset_and_symptoms, predict
from core.model import OptimizedDiseaseClassifier

# Mock data and model paths
MODELS_DIR = "models"
DATA_PATH = "data/processed_diseases-priority.csv"

def test_restructured_files():
    assert os.path.isdir("models")
    assert os.path.isdir("data")
    assert os.path.isdir("core")
    assert os.path.isfile(DATA_PATH)
    assert os.path.isfile(os.path.join(MODELS_DIR, "best_model.pth"))
    assert os.path.isfile(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    assert os.path.isfile(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

def test_model_loading():
    num_classes = len(pd.read_csv(DATA_PATH)["Disease"].unique())
    model, tfidf, label_encoder = load_model_and_artifacts(MODELS_DIR, num_classes)
    assert model is not None
    assert tfidf is not None
    assert label_encoder is not None
    assert isinstance(model, OptimizedDiseaseClassifier)

def test_dataset_loading():
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    df, symptoms = load_dataset_and_symptoms(DATA_PATH, label_encoder)
    assert not df.empty
    assert len(symptoms) > 0
    assert "Disease_Encoded" in df.columns

def test_prediction():
    num_classes = len(pd.read_csv(DATA_PATH)["Disease"].unique())
    model, tfidf, label_encoder = load_model_and_artifacts(MODELS_DIR, num_classes)
    df, symptoms = load_dataset_and_symptoms(DATA_PATH, label_encoder)

    # Use a real symptom from the dataset
    test_symptom = symptoms[0]
    disease, treatment, conf, top_d, top_c = predict(
        test_symptom, model, tfidf, label_encoder, df, confidence_threshold=0.1
    )

    assert isinstance(disease, str)
    assert isinstance(conf, float)
    assert len(top_d) == 3
    assert len(top_c) == 3
