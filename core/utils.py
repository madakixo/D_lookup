import os
import pickle
import torch
import pandas as pd
import numpy as np
from core.model import OptimizedDiseaseClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_artifacts(models_dir, num_classes):
    # Check if files are actual files or LFS pointers
    model_path = os.path.join(models_dir, "best_model.pth")
    with open(model_path, "r") as f:
        head = f.read(100)
        if "version https://git-lfs.github.com" in head:
            raise RuntimeError(f"File {model_path} is an LFS pointer. Please pull the actual file.")

    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    input_dim = len(tfidf.vocabulary_)

    model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    return model, tfidf, label_encoder

def load_dataset_and_symptoms(data_path, label_encoder):
    df = pd.read_csv(data_path)
    df.dropna(subset=["Disease", "Symptoms"], inplace=True)
    df["Disease_Encoded"] = label_encoder.transform(df["Disease"])
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

def predict(symptoms, model, tfidf, label_encoder, df, confidence_threshold=0.6):
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    disease_name = label_encoder.inverse_transform([predicted_class])[0] if confidence >= confidence_threshold else "Uncertain"

    # Matching based on encoded disease
    matched_rows = df[df["Disease_Encoded"] == predicted_class]
    treatment = matched_rows.iloc[0].get("Treatment", "N/A") if not matched_rows.empty and confidence >= confidence_threshold else "N/A"

    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_diseases = label_encoder.inverse_transform(top_indices.cpu().numpy())
    top_confidences = top_probs.cpu().numpy()

    return disease_name, treatment, confidence, top_diseases, top_confidences
