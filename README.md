# 🩺 Disease Prediction & AI Health Assistant
### Leveraging NLP for Medical Prescription Administration
**Omdena Kaduna Local Chapter** • *Initiated by Jamaludeen Madaki*

---

## 📖 Overview
This application uses **Natural Language Processing (NLP)** and **PyTorch** to predict potential diseases and suggest treatments based on user-provided symptoms. Designed by the Omdena Kaduna Chapter, it serves as an educational tool to demonstrate how AI can support healthcare in regions with limited medical expertise.

## 🚀 Features
- **Intelligent Symptom Input**: Select common symptoms from a pre-defined list or enter custom descriptions in plain text.
- **AI-Powered Prediction**: Get real-time disease predictions with confidence scores using a custom-trained neural network.
- **Dynamic Disease Lookup**: Explore an extensive health database with symptoms, treatments, and likely causes for hundreds of conditions.
- **Interactive Visualizations**: View top alternative predictions through intuitive Plotly charts.
- **Data-Driven Analysis**: Automatic logging of predictions and integration with Google Sheets for health trend analysis.
- **Search History**: Track your recent symptom searches within the session.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Omdena-Kaduna/NLP-Medical-Prescription.git
cd NLP-Medical-Prescription
```

### 2. Set Up Virtual Environment
```bash
# Example for Linux/Mac
# python -m venv venv
# source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Project Structure
The project is organized into a modular structure for better maintainability:
- `core/`: Shared logic, model definitions, and utility functions.
- `models/`: Machine learning artifacts (`best_model.pth`, `tfidf_vectorizer.pkl`, etc.).
- `data/`: Dataset files (`processed_diseases-priority.csv`).
- `tests/`: Automated test suite.

> **Note**: Model files are managed via Git LFS. Ensure you have the actual `.pth` and `.pkl` files in the `models/` directory.

### 5. Google Sheets Integration (Optional)
To enable data collection, place your Google Cloud service account JSON in the root as `credentials.json`.

---

## 🕹️ How to Use & Operate

### Running the App
Launch the main interface using Streamlit:
```bash
streamlit run app2026.py
```
The application will open in your default browser at `http://localhost:8501`.

### Making a Prediction (Quick Brief)
1. **Input Symptoms**: Use the multi-select box to pick from known symptoms. For symptoms not listed, type them into the text area below (comma-separated).
2. **Set Confidence**: Use the sidebar slider to adjust the "Confidence Threshold." Predictions below this level will be marked as "Uncertain."
3. **Analyze**: Click the **"🔍 Analyze & Predict"** button. The AI will process your input and provide:
   - The most likely disease.
   - A suggested treatment plan from the dataset.
   - A probability score (confidence).
   - A chart showing the top 3 alternative possibilities.

### Performing Analysis
- **Review History**: Scroll down to the "Search History" section to see a record of your symptoms and results during this session.
- **Explore the Database**: Use the "Health Database" section in the right column to lookup specific diseases and see their typical profiles (Symptoms, Treatments, and Causes).

---

## 🧪 Testing
Run the automated test suite to verify system integrity:
```bash
./run_tests.sh
```

## ⚠️ Limitations
- **Educational Purpose Only**: This app is **not** a substitute for professional medical advice, diagnosis, or treatment.
- **Dataset Dependency**: Predictions are only as accurate as the underlying training data.
- **LFS Required**: The application will not function with Git LFS pointer files; actual weights must be downloaded.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and open a pull request with your improvements.

## 📜 License
This project is licensed under the MIT License.

## 📧 Contact
Omdena Kaduna Local Chapter via Omdena or Jamaludeen Madaki.
