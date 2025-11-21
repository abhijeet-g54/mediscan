# app/main_app.py

import streamlit as st
import joblib
import numpy as np
import os
from pathlib import Path
import sys

# --- Ensure imports from project root ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.feature_engineering import normalize_symptom, build_feature_vector

st.set_page_config(page_title="MediScan - AI Symptom Checker", page_icon="💊", layout="centered")

MODEL_PATH = "models/symptom_disease_model.pkl"

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(" Model file not found. Please train the model first.")
        st.stop()
    pkg = joblib.load(MODEL_PATH)
    required_keys = {"model", "label_encoder"}
    if not all(k in pkg for k in required_keys):
        st.error(" Incomplete model file. Retrain to include model and label encoder.")
        st.stop()
    return pkg

pkg = load_model()
model = pkg["model"]
label_encoder = pkg["label_encoder"]
vectorizer = pkg.get("vectorizer")
features = pkg.get("features")

# -----------------------------
# Clarifying question rules
# -----------------------------
CLARIFY_RULES = {
    "fever": [
        "Do you have chills or body aches?",
        "Have you experienced cough or sore throat?",
        "Any recent travel or mosquito exposure?",
    ],
    "cough": [
        "Is the cough dry or with phlegm?",
        "Do you also feel shortness of breath or chest pain?",
    ],
    "pain": [
        "Where exactly is the pain (e.g., stomach, back, chest)?",
        "How severe is it on a scale of 1–10?",
    ],
}

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("💡 About MediScan")
    st.markdown("""
    MediScan predicts likely conditions based on symptoms.
    It uses an XGBoost model trained on curated symptom–disease data.
    """)
    st.caption("Tip: add multiple symptoms for better results.")

st.title("🩺 MediScan – AI Symptom Checker")

# -----------------------------
# Stage 1: Enter symptoms
# -----------------------------
if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = []
if "clarify_questions" not in st.session_state:
    st.session_state["clarify_questions"] = []
if "show_details" not in st.session_state:
    st.session_state["show_details"] = False

raw_input = st.text_area(
    "Enter your symptoms (comma separated):",
    placeholder="e.g. fever, headache, sore throat"
)

if st.button("Analyze Symptoms"):
    if not raw_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        raw_symptoms = [s.strip().lower() for s in raw_input.split(",") if s.strip()]
        normalized = [normalize_symptom(s) for s in raw_symptoms]
        st.session_state["symptoms"] = normalized

        # build clarifying questions
        questions = []
        for s in normalized:
            if s in CLARIFY_RULES:
                questions.extend(CLARIFY_RULES[s])
        st.session_state["clarify_questions"] = questions
        st.session_state["show_details"] = True
        st.rerun()

# -----------------------------
# Stage 2: Show clarifications + intensity/duration
# -----------------------------
if st.session_state["show_details"]:
    st.markdown("### 🔍 Clarifying Questions")
    answers = {}
    for q in st.session_state["clarify_questions"]:
        answers[q] = st.text_input(f"• {q}", key=f"q_{q}")

    st.markdown("### ⚙️ Symptom Details")
    intensity_map = {}
    duration_map = {}

    for sym in st.session_state["symptoms"]:
        col1, col2 = st.columns(2)
        with col1:
            intensity_map[sym] = st.selectbox(
                f"Intensity of '{sym}'",
                ["mild", "moderate", "severe"],
                key=f"intensity_{sym}"
            )
        with col2:
            duration_map[sym] = st.number_input(
                f"Duration of '{sym}' (days)",
                min_value=0.0,
                max_value=60.0,
                value=1.0,
                step=0.5,
                key=f"duration_{sym}"
            )

    if st.button("Finalize Analysis"):
        normalized = st.session_state["symptoms"]

        # Build feature vector
        if vectorizer is not None:
            joined = ", ".join(normalized)
            X_input = vectorizer.transform([joined])
        elif features is not None:
            X_input = build_feature_vector(
                normalized,
                features,
                intensity_map={k: {"mild": 0, "moderate": 1, "severe": 2}[v] for k, v in intensity_map.items()},
                duration_map=duration_map
            ).reshape(1, -1)
        else:
            st.error("❌ Model missing both vectorizer and features.")
            st.stop()

        # Predict top 3
        probs = model.predict_proba(X_input)[0]
        topk = np.argsort(probs)[-3:][::-1]
        labels = [label_encoder.inverse_transform([i])[0] for i in topk]

        st.subheader("🧠 Top Predictions")
        for i, idx in enumerate(topk):
            st.write(f"{i+1}. **{labels[i]}** — {probs[idx]*100:.1f}% confidence")

        st.markdown("---")
        correct = st.text_input("If incorrect, type the correct disease (optional):")
        if st.button("Submit Feedback"):
            import csv, datetime
            os.makedirs("data", exist_ok=True)
            with open("data/feedback.csv", "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now().isoformat(), raw_input, labels[0], correct])
            st.success("✅ Feedback saved. Thank you!")
