"""
utils/feature_engineering.py

This module handles:
1. Symptom normalization (using symptom_dictionary.json + fuzzy matching)
2. Feature vector construction using:
      - binary presence + intensity weighting
      - duration weighting
3. Optional clarifying question answers (text) → numeric embedding

NOTE:
The clarifying question logic is handled by main_app.py,
but this file provides helper conversions FOR the answers.
"""

import json
import numpy as np
from fuzzywuzzy import process


# -------------------------------------------------------------------
# Load Symptom Dictionary
# -------------------------------------------------------------------
SYM_DICT_PATH = "utils/symptom_dictionary.json"

try:
    with open(SYM_DICT_PATH, "r", encoding="utf-8") as f:
        SYM_DICT = json.load(f)
except Exception:
    SYM_DICT = {}


# -------------------------------------------------------------------
# Build reverse mapping (canonical -> list of user terms)
# -------------------------------------------------------------------
CANONICALS = {}

for user_term, canonical in SYM_DICT.items():
    if isinstance(canonical, list):
        for c in canonical:
            CANONICALS.setdefault(c, []).append(user_term)
    elif isinstance(canonical, str):
        CANONICALS.setdefault(canonical, []).append(user_term)
    # ignore malformed entries


# -------------------------------------------------------------------
# Symptom Normalization
# -------------------------------------------------------------------
def normalize_symptom(user_text):
    """
    Convert user-entered symptom into a canonical token:
    - Direct dictionary match first
    - Fuzzy match against dictionary keys
    - Fallback: replace spaces with underscores
    """

    if not user_text:
        return ""

    t = user_text.strip().lower()

    # direct match
    if t in SYM_DICT:
        val = SYM_DICT[t]
        return val[0] if isinstance(val, list) else val

    # fuzzy match
    if SYM_DICT:
        best = process.extractOne(t, list(SYM_DICT.keys()))
        if best and best[1] >= 85:
            val = SYM_DICT[best[0]]
            return val[0] if isinstance(val, list) else val

    # fallback
    return t.replace(" ", "_")


# -------------------------------------------------------------------
# Clarifying Question Embedding (text → number)
# -------------------------------------------------------------------
def encode_clarifying_answer(text):
    """
    Convert clarifying question answers (free text) to numeric features.
    Lightweight because we avoid large TF-IDF on answers.

    Encoding:
      - Yes / y / true => 1.0
      - No / n / false => -1.0
      - Empty or other => 0.0
    """

    if not text or not isinstance(text, str):
        return 0.0

    t = text.strip().lower()

    if t in {"yes", "y", "true", "yeah", "yep"}:
        return 1.0
    if t in {"no", "n", "false", "nope"}:
        return -1.0

    return 0.0


# -------------------------------------------------------------------
# Build Full Feature Vector
# -------------------------------------------------------------------
def build_feature_vector(
    selected_symptoms,
    all_symptoms,
    intensity_map=None,
    duration_map=None,
    clarifying_answers=None,
    clarifying_questions_order=None,
):
    """
    Constructs the final input vector for the ML model.

    selected_symptoms: list[str]
    all_symptoms: list[str] (fixed order from training)
    intensity_map: {symptom: 0/1/2}
    duration_map: {symptom: float}
    clarifying_answers: {question_id: answer_text}
    clarifying_questions_order: list of question_ids MUST match training order

    Returns: numpy array of length len(all_symptoms) + len(clarifying_questions_order)
    """

    # base symptom vector
    vec = np.zeros(len(all_symptoms), dtype=float)

    for s in selected_symptoms:
        canon = normalize_symptom(s)

        if canon in all_symptoms:
            idx = all_symptoms.index(canon)

            # intensity weighting
            intensity_score = 1.0
            if intensity_map and canon in intensity_map:
                iv = intensity_map[canon]
                try:
                    iv = int(iv)
                except:
                    iv = 0
                intensity_score += iv * 0.5  # mild=1.0, moderate=1.5, severe=2.0

            # duration weighting
            duration_score = 1.0
            if duration_map and canon in duration_map:
                try:
                    d = float(duration_map[canon])
                    duration_score += np.log1p(max(d, 0))
                except:
                    pass

            vec[idx] = 1.0 * intensity_score * duration_score

    # -----------------------------------
    # Add clarifying question embeddings
    # -----------------------------------
    if clarifying_questions_order:
        clar_vec = []
        for q_id in clarifying_questions_order:
            ans = clarifying_answers.get(q_id, "") if clarifying_answers else ""
            clar_vec.append(encode_clarifying_answer(ans))
        clar_vec = np.array(clar_vec, dtype=float)
        return np.concatenate([vec, clar_vec])

    return vec
