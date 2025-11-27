# app/main_app.py

import streamlit as st
import joblib
import numpy as np
import os
import json
from pathlib import Path
import sys

# -----------------------------
# Ensure project root import
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.feature_engineering import normalize_symptom, build_feature_vector

st.set_page_config(page_title="MediScan - AI Symptom Checker", page_icon="💊", layout="wide")

MODEL_PATH = "models/symptom_disease_model.pkl"
CLARIFY_JSON = ROOT / "utils" / "clarifying_questions.json"

# -----------------------------
# Load clarifying rules
# -----------------------------
if os.path.exists(CLARIFY_JSON):
    with open(CLARIFY_JSON, "r", encoding="utf-8") as f:
        CLARIFY_RULES = json.load(f)
else:
    CLARIFY_RULES = {}

# -----------------------------
# Load model bundle
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train it first.")
        st.stop()
    pkg = joblib.load(MODEL_PATH)

    required = {"model", "label_encoder", "symptoms"}
    if not required.issubset(pkg.keys()):
        st.error("Incomplete model file. Retrain the model.")
        st.stop()

    return pkg

pkg = load_model()
model = pkg["model"]
label_encoder = pkg["label_encoder"]
ALL_SYMPTOMS = pkg["symptoms"]

# -----------------------------
# Helper: Highest-prob diseases
# -----------------------------
def get_top_diseases(probs, k=3):
    idx = np.argsort(probs)[-k:][::-1]
    diseases = [label_encoder.inverse_transform([i])[0] for i in idx]
    return diseases, idx

# -----------------------------
# Helper: Missing-symptom questions
# -----------------------------
PROMPTED_SYMPTOMS = {}  # maps question text -> list of missing symptoms

def get_missing_symptom_questions(user_syms, top_diseases, probs):
    questions = []
    PROMPTED_SYMPTOMS.clear()  # reset

    for i, disease in enumerate(top_diseases):
        if disease not in CLARIFY_RULES:
            continue

        key_syms = CLARIFY_RULES[disease].get("key_symptoms", [])
        missing = [s for s in key_syms if s not in user_syms]

        # ask if probability low or missing key symptoms exist
        if probs[i] < 0.1 or missing:
            if missing:
                q = (
                    f"For suspected **{disease}**, you haven't mentioned symptoms like "
                    f"**{', '.join(missing)}**. Do you have any of these?"
                )
                questions.append(q)
                PROMPTED_SYMPTOMS[q] = missing

            # add extra general questions
            for extra_q in CLARIFY_RULES[disease].get("extra_questions", []):
                if extra_q not in questions:
                    questions.append(extra_q)
                    PROMPTED_SYMPTOMS[extra_q] = []

    return questions[:5]  # limit to 5

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("💡 About MediScan")
    st.markdown("""
    MediScan predicts possible diseases from symptoms  
    using a machine-learning model trained on symptoms.
    """)
    st.caption("More symptoms → better accuracy.")


st.title("🩺 MediScan – AI Symptom Checker")

# -----------------------------
# Session states
# -----------------------------
if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = []
if "clarify_questions" not in st.session_state:
    st.session_state["clarify_questions"] = []
if "show_details" not in st.session_state:
    st.session_state["show_details"] = False

# -----------------------------
# STEP 1: Enter symptoms
# -----------------------------
raw_input = st.text_area(
    "Enter your symptoms (comma separated):",
    placeholder="e.g. fever, headache, sore throat"
)

if st.button("Analyze Symptoms"):
    if not raw_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        raw_syms = [s.strip().lower() for s in raw_input.split(",") if s.strip()]
        normalized = [normalize_symptom(s) for s in raw_syms]

        st.session_state["symptoms"] = normalized

        # Build temporary vector to estimate top probable diseases → for clarifying
        temp_vec = build_feature_vector(normalized, ALL_SYMPTOMS).reshape(1, -1)
        probs = model.predict_proba(temp_vec)[0]

        top_diseases, _ = get_top_diseases(probs, k=3)

        # Dynamic clarifying questions
        clarifying = get_missing_symptom_questions(normalized, top_diseases, probs)

        st.session_state["clarify_questions"] = clarifying
        st.session_state["prompted_symptoms"] = PROMPTED_SYMPTOMS.copy()  # store in session_state
        st.session_state["show_details"] = True
        st.rerun()

# -----------------------------
# STEP 2: Clarifying + Intensity + Duration
# -----------------------------
if st.session_state["show_details"]:

    st.markdown("### 🔍 Clarifying Questions")
    answers = {}

    # Ensure prompted symptoms are stored in session_state
    if "prompted_symptoms" not in st.session_state:
        st.session_state["prompted_symptoms"] = {}

    for q in st.session_state["clarify_questions"]:
        answers[q] = st.text_input(f"• {q}", key=f"clarify_{q}")

    # integrate clarifying answers into symptoms
    for q, ans in answers.items():
        if ans.strip().lower() in ["yes", "y"]:
            # only add the specific symptoms prompted for this question
            for s in st.session_state["prompted_symptoms"].get(q, []):
                if s not in st.session_state["symptoms"]:
                    st.session_state["symptoms"].append(s)
    # Intensity / duration
    st.markdown("### ⚙ Symptom Details")

    intensity_map = {}
    duration_map = {}

    with st.expander("Set Symptom Intensity and Duration"):
        for sym in st.session_state["symptoms"]:
            col1, col2 = st.columns(2)

            with col1:
                intensity_map[sym] = st.selectbox(
                    f"Intensity of '{sym}'",
                    ["mild", "moderate", "severe"],
                    key=f"int_{sym}"
                )

            with col2:
                duration_map[sym] = st.number_input(
                    f"Duration of '{sym}' (days)",
                    min_value=0.0, max_value=60.0, step=0.5, value=1.0,
                    key=f"dur_{sym}"
                )

    # Final analysis button
    if st.button("Finalize Analysis"):
        final_syms = st.session_state["symptoms"]

        X_input = build_feature_vector(
            final_syms,
            ALL_SYMPTOMS,
            intensity_map={k: {"mild": 0, "moderate": 1, "severe": 2}[v] for k, v in intensity_map.items()},
            duration_map=duration_map
        ).reshape(1, -1)

        probs = model.predict_proba(X_input)[0]
        top_diseases, idxs = get_top_diseases(probs)

        st.subheader("🧠 Top Predictions")
        for rank, i in enumerate(idxs):
            st.write(f"**{rank+1}. {label_encoder.inverse_transform([i])[0]}** — {probs[i]*100:.1f}%")

        # Feedback
        st.markdown("---")
        correct = st.text_input("If the prediction is wrong, type the correct disease (optional):")
        if st.button("Submit Feedback"):
            import csv, datetime
            os.makedirs("data", exist_ok=True)
            with open("data/feedback.csv", "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    raw_input,
                    top_diseases[0],
                    correct
                ])
            st.success("Feedback saved. Thank you!")
