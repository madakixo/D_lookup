# Streamlit app for modeling task 4
## @madakixo
# Purpose: Deploy a neural network to classify diseases based on symptoms, using TF-IDF features.
# Designed for interactive use with Streamlit and Google Drive integration.
# Enhanced with disease lookup functionality – now in main area

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import plotly.express as px
import time

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Leveraging NLP in Medical Prescription",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# Model Definition (same as before)
# ────────────────────────────────────────────────
class OptimizedDiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OptimizedDiseaseClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.dropout_layers = [layer for layer in self.network if isinstance(layer, nn.Dropout)]

    def forward(self, x):
        return self.network(x)

# ────────────────────────────────────────────────
# Load model + artifacts (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_artifacts(num_classes):
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        input_dim = len(tfidf.vocabulary_)

        model = OptimizedDiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
        checkpoint = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        return model, tfidf, label_encoder
    except Exception as e:
        st.error(f"Error loading model or artifacts: {e}")
        raise

# ────────────────────────────────────────────────
# Load dataset + extract symptoms
# ────────────────────────────────────────────────
@st.cache_data
def load_dataset_and_symptoms(data_path, _label_encoder):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    df["Disease"] = _label_encoder.transform(df["Disease"])
    all_symptoms = set()
    for symptoms in df["Symptoms"]:
        all_symptoms.update([s.strip() for s in symptoms.split(",")])
    return df, sorted(list(all_symptoms))

# ────────────────────────────────────────────────
# Prediction logic (unchanged)
# ────────────────────────────────────────────────
def predict(symptoms, model, tfidf, label_encoder, df, confidence_threshold=0.6):
    symptoms_tfidf = tfidf.transform([symptoms]).toarray()
    symptoms_tensor = torch.tensor(symptoms_tfidf, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(symptoms_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    disease_name = label_encoder.inverse_transform([predicted_class])[0] if confidence >= confidence_threshold else "Uncertain"
    treatment = df[df["Disease"] == predicted_class].iloc[0].get("Treatment", "N/A") if confidence >= confidence_threshold else "N/A"

    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, top_k)
    top_diseases = label_encoder.inverse_transform(top_indices.cpu().numpy())
    top_confidences = top_probs.cpu().numpy()

    return disease_name, treatment, confidence, top_diseases, top_confidences

# ────────────────────────────────────────────────
# Custom CSS – larger fonts & better spacing
# ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 19px !important;
    }
    h1 { font-size: 42px !important; }
    h2 { font-size: 34px !important; }
    h3 { font-size: 28px !important; }
    .stButton>button { font-size: 20px !important; padding: 0.6rem 1.2rem; }
    .stSelectbox, .stTextArea, .stMultiSelect {
        font-size: 19px !important;
    }
    .block-container { padding-top: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────
def main():
    data_path = "processed_diseases-priority.csv"

    # Load everything
    try:
        num_classes = len(pd.read_csv(data_path)["Disease"].unique())
        model, tfidf, label_encoder = load_model_and_artifacts(num_classes)
        df_filtered, common_symptoms = load_dataset_and_symptoms(data_path, label_encoder)
        st.success("Model, vectorizer & label encoder loaded successfully.", icon="✅")
    except Exception as e:
        st.error(f"Loading failed: {e}")
        return

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    # ──────── SIDEBAR ────────────────────────────────────────
    st.sidebar.title("🩺 Disease Predictor")
    st.sidebar.markdown("Enter symptoms → get prediction")

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053171.png", width=120)

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.60, 0.05,
        help="Predictions below this confidence are marked as 'Uncertain'."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("This is an educational tool — always consult a doctor.")

    # ──────── MAIN AREA LAYOUT ───────────────────────────────
    col_left, col_right = st.columns([5, 3])

    with col_left:
        st.title("Interactive Disease Prediction")
        st.markdown("Select or type symptoms and get real-time predictions.")

        st.subheader("Your Symptoms")

        selected_symptoms = st.multiselect(
            "Pick common symptoms",
            options=common_symptoms,
            placeholder="Start typing or select...",
            help="Select one or multiple symptoms"
        )

        manual_symptoms = st.text_area(
            "Additional / custom symptoms (comma-separated)",
            placeholder="fatigue, night sweats, weight loss",
            height=110
        )

        all_symptoms_input = ", ".join(
            selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()]
        )

        if all_symptoms_input:
            st.info(f"**Current input:** {all_symptoms_input}")
        else:
            st.warning("Please select or type at least one symptom.")

        if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
            if all_symptoms_input:
                with st.spinner("Analyzing..."):
                    time.sleep(0.6)  # slight delay for UX
                    disease, treatment, conf, top_d, top_c = predict(
                        all_symptoms_input, model, tfidf, label_encoder,
                        df_filtered, confidence_threshold
                    )

                    badge_color = "#2ecc71" if conf >= 0.75 else "#f39c12" if conf >= 0.6 else "#e74c3c"
                    st.markdown(
                        f"**Predicted Disease:** {disease}   "
                        f"<span style='background-color:{badge_color}; color:white; padding:4px 10px; border-radius:6px; font-weight:bold;'>{conf:.1%}</span>",
                        unsafe_allow_html=True
                    )

                    st.write(f"**Suggested Treatment (dataset):** {treatment}")

                    st.subheader("Top 3 Predictions")
                    fig = px.bar(
                        x=top_d,
                        y=top_c,
                        labels={"x": "Disease", "y": "Confidence"},
                        title="",
                        color=top_c,
                        color_continuous_scale="Blues",
                        text_auto=".1%"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Save to history
                    st.session_state.history.append({
                        "Symptoms": all_symptoms_input,
                        "Disease": disease,
                        "Confidence": f"{conf:.1%}",
                        "Treatment": treatment
                    })
            else:
                st.error("No symptoms provided.")

        # History
        if st.session_state.history:
            st.subheader("Prediction History")
            st.dataframe(
                pd.DataFrame(st.session_state.history),
                use_container_width=True,
                hide_index=True
            )
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    # ──────── RIGHT COLUMN – DISEASE LOOKUP ───────────────────
    with col_right:
        st.subheader("🔎 Disease → Symptoms Lookup")
        st.markdown("Browse symptoms and treatment of known diseases.")

        # You had a very long hard-coded list → better to derive from data if possible
        # For now keeping your list, but consider replacing with:
        # disease_list = sorted(label_encoder.classes_)

        disease_list = sorted([
            "Vulvodynia", "Cold Sores", "Renal Cell Carcinoma", "Scabies", "Parkinson’s Disease",
            # ... (your full list here – truncated for brevity)
            "Tinnitus"
        ])

        selected_disease = st.selectbox(
            "Select Disease",
            options=disease_list,
            index=0,
            help="Choose a disease to see typical symptoms and treatment"
        )

        if selected_disease:
            try:
                encoded = label_encoder.transform([selected_disease])[0]
                row = df_filtered[df_filtered["Disease"] == encoded].iloc[0]

                st.markdown(f"**Symptoms:**")
                st.info(row["Symptoms"])

                if "Treatment" in row and pd.notna(row["Treatment"]):
                    st.markdown(f"**Treatment (from dataset):**")
                    st.success(row["Treatment"])
                else:
                    st.warning("No treatment information in dataset.")

            except Exception as e:
                st.error(f"Could not retrieve data for {selected_disease} → {e}")

    # ──────── BOTTOM SHARED SECTION ───────────────────────────
    st.markdown("---")

    st.subheader("Feedback")
    with st.expander("Quick Feedback"):
        feedback = st.text_area("Your thoughts / suggestions", placeholder="How can we improve?")
        if st.button("Submit Feedback") and feedback.strip():
            st.session_state.feedback_submitted = True
            st.success("Thank you! ❤️")

    st.markdown(
        "[Fill detailed feedback form →](https://forms.gle/your-google-form-link)",
        unsafe_allow_html=True
    )

    st.caption(
        "Educational project only • Not a substitute for professional medical advice • "
        "Built with Streamlit, PyTorch & Plotly • Omdena Kaduna Impact Hub"
    )

if __name__ == "__main__":
    main()
