import streamlit as st
import pandas as pd
import time
import plotly.express as px
import os
import logging
from core.utils import load_model_and_artifacts, load_dataset_and_symptoms, predict
from core.sheets import get_sheets_client, log_prediction_to_sheets

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = "models"
DATA_PATH = "data/processed_diseases-priority.csv"
CREDENTIALS_PATH = "credentials.json"
SHEET_NAME = "MedicalPredictions"

# Custom CSS
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 19px !important; }
    h1 { font-size: 42px !important; }
    h2 { font-size: 34px !important; }
    h3 { font-size: 28px !important; }
    .stButton>button { font-size: 20px !important; padding: 0.6rem 1.2rem; }
    .stSelectbox, .stTextArea, .stMultiSelect { font-size: 19px !important; }
    .block-container { padding-top: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Load everything
    try:
        num_classes = len(pd.read_csv(DATA_PATH)["Disease"].unique())
        model, tfidf, label_encoder = load_model_and_artifacts(MODELS_DIR, num_classes)
        df_filtered, common_symptoms = load_dataset_and_symptoms(DATA_PATH, label_encoder)
        st.success("System loaded successfully.", icon="✅")
    except Exception as e:
        st.error(f"Loading failed: {e}")
        logger.error(f"Initialization failed: {e}")
        return

    # Google Sheets Integration
    sheets_client = None
    if os.path.exists(CREDENTIALS_PATH):
        sheets_client = get_sheets_client(CREDENTIALS_PATH)
    else:
        logger.warning("Google Sheets credentials not found. Data collection disabled.")

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # ──────── SIDEBAR ────────────────────────────────────────
    st.sidebar.title("🩺 AI Health Assistant")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053171.png", width=120)

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.60, 0.05,
        help="Predictions below this confidence are marked as 'Uncertain'."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("This is an educational tool — always consult a doctor.")

    # ──────── MAIN AREA LAYOUT ───────────────────────────────
    col_left, col_right = st.columns([5, 3])

    with col_left:
        st.title("Interactive Disease Prediction")
        st.markdown("Enter your symptoms to receive an AI-powered prediction and suggested care.")

        st.subheader("Symptom Input")
        selected_symptoms = st.multiselect(
            "Select common symptoms",
            options=common_symptoms,
            placeholder="Start typing or select...",
        )

        manual_symptoms = st.text_area(
            "Additional symptoms (comma-separated)",
            placeholder="e.g., fatigue, night sweats",
            height=100
        )

        all_symptoms_input = ", ".join(
            selected_symptoms + [s.strip() for s in manual_symptoms.split(",") if s.strip()]
        )

        if all_symptoms_input:
            st.info(f"**Selected Symptoms:** {all_symptoms_input}")

        if st.button("🔍 Analyze & Predict", type="primary", use_container_width=True):
            if all_symptoms_input:
                with st.spinner("Analyzing data..."):
                    disease, treatment, conf, top_d, top_c = predict(
                        all_symptoms_input, model, tfidf, label_encoder,
                        df_filtered, confidence_threshold
                    )

                    badge_color = "#2ecc71" if conf >= 0.75 else "#f39c12" if conf >= 0.6 else "#e74c3c"
                    st.markdown(
                        f"### Predicted Result: {disease}   "
                        f"<span style='background-color:{badge_color}; color:white; padding:4px 12px; border-radius:8px; font-weight:bold;'>{conf:.1%} Confidence</span>",
                        unsafe_allow_html=True
                    )

                    st.write(f"**Suggested Care/Treatment:** {treatment}")

                    # Visualization
                    fig = px.bar(
                        x=top_d, y=top_c,
                        labels={"x": "Disease", "y": "Confidence"},
                        title="Top Predictions",
                        color=top_c,
                        color_continuous_scale="Blues",
                        text_auto=".1%"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Log to history
                    st.session_state.history.append({
                        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Symptoms": all_symptoms_input,
                        "Disease": disease,
                        "Confidence": f"{conf:.1%}"
                    })

                    # Data collection to Google Sheets
                    if sheets_client:
                        # Extracting additional info from dataset for logging
                        try:
                            # Search for the predicted disease's row to get more details
                            if disease != "Uncertain":
                                info_row = df_filtered[df_filtered["Disease_Encoded"] == label_encoder.transform([disease])[0]].iloc[0]
                                cause = info_row.get("Cause", "N/A")
                                reason = "Based on symptom match"
                                diagnosis = disease
                                log_prediction_to_sheets(sheets_client, SHEET_NAME, [
                                    time.strftime("%Y-%m-%d %H:%M:%S"),
                                    all_symptoms_input,
                                    disease,
                                    f"{conf:.1%}",
                                    cause,
                                    reason,
                                    diagnosis
                                ])
                                logger.info(f"Logged prediction for {disease} to Google Sheets.")
                        except Exception as e:
                            logger.error(f"Google Sheets logging failed: {e}")

                    logger.info(f"Prediction made: {disease} (Conf: {conf:.2f})")
            else:
                st.error("Please provide at least one symptom.")

        # History Section
        if st.session_state.history:
            st.divider()
            st.subheader("Your Search History")
            st.table(st.session_state.history)
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    # ──────── RIGHT COLUMN – LOOKUP ──────────────────────────
    with col_right:
        st.subheader("🔎 Health Database")
        st.markdown("Explore typical symptoms and treatments.")

        # Dynamic disease list from label_encoder
        disease_list = sorted(label_encoder.classes_)

        selected_disease = st.selectbox(
            "Search Disease",
            options=disease_list,
            index=0,
            help="Get detailed information about a specific condition."
        )

        if selected_disease:
            try:
                encoded = label_encoder.transform([selected_disease])[0]
                row = df_filtered[df_filtered["Disease_Encoded"] == encoded].iloc[0]

                st.info(f"**Common Symptoms:**\n\n{row['Symptoms']}")

                if "Treatment" in row and pd.notna(row["Treatment"]):
                    st.success(f"**Typical Treatment:**\n\n{row['Treatment']}")

                if "Cause" in row and pd.notna(row["Cause"]):
                    st.warning(f"**Likely Cause:**\n\n{row['Cause']}")

            except Exception as e:
                st.error(f"Error retrieving data: {e}")

    # ──────── FOOTER ──────────────────────────────────────────
    st.divider()
    st.caption(
        "Educational project • Not a medical diagnosis • "
        "Built with Streamlit, PyTorch & Omdena • Kaduna Local Chapter"
    )

if __name__ == "__main__":
    main()
