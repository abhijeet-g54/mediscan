"""
utils/feature_engineering.py

Normalization and feature builder used at inference time.
- normalize_symptom: map user text to canonical token (best-effort)
- build_feature_vector: convert list of canonical symptoms + intensity/duration maps to numeric vector
"""

import json
import numpy as np
from fuzzywuzzy import process

# optional: a symptom_dictionary file that maps common user phrases to canonical token
SYM_DICT_PATH = "utils/symptom_dictionary.json"
try:
    with open(SYM_DICT_PATH, "r", encoding="utf-8") as f:
        SYM_DICT = json.load(f)
except Exception:
    SYM_DICT = {}

# --- Build reverse lookup safely ---
CANONICALS = {}
for k, v in SYM_DICT.items():
    # Sometimes v might be a list of canonical names — handle that safely
    if isinstance(v, list):
        for canon in v:
            CANONICALS[canon] = k
    elif isinstance(v, str):
        CANONICALS[v] = k
    else:
        continue  # ignore unexpected formats


def normalize_symptom(user_text):
    """
    Attempt to map freeform user text to a canonical symptom token.
    If mapping not found, return a sanitized token (spaces->underscores).
    """
    if not user_text:
        return ""
    t = user_text.strip().lower()

    # Direct mapping
    if t in SYM_DICT:
        val = SYM_DICT[t]
        # If multiple canonical forms exist, just take the first
        if isinstance(val, list):
            return val[0]
        return val

    # Fuzzy match
    best = process.extractOne(t, list(SYM_DICT.keys())) if SYM_DICT else None
    if best and best[1] >= 85:
        val = SYM_DICT[best[0]]
        if isinstance(val, list):
            return val[0]
        return val

    # Fall back to simple normalization
    return t.replace(" ", "_")


def build_feature_vector(selected_symptoms, all_symptoms, intensity_map=None, duration_map=None):
    """
    selected_symptoms: list of canonical tokens (strings)
    all_symptoms: ordered list of canonical tokens (feature names)
    intensity_map: dict token->int (0=mild,1=moderate,2=severe)
    duration_map: dict token->days (float)
    Returns: numpy array len(all_symptoms)
    """
    vec = np.zeros(len(all_symptoms), dtype=float)

    for s in selected_symptoms:
        canon = normalize_symptom(s)
        if canon in all_symptoms:
            idx = all_symptoms.index(canon)

            # Handle intensity (mild/moderate/severe)
            intensity = 0
            if intensity_map and canon in intensity_map:
                try:
                    intensity = int(intensity_map[canon])
                except ValueError:
                    intensity = 0
            intensity_mult = 1.0 + 0.5 * float(intensity)  # mild=1.0, mod=1.5, sev=2.0

            # Handle duration (in days)
            duration = 0.0
            if duration_map and canon in duration_map:
                try:
                    duration = float(duration_map[canon])
                except ValueError:
                    duration = 0.0
            duration_mult = 1.0 + np.log1p(duration)

            vec[idx] = 1.0 * intensity_mult * duration_mult

    return vec
