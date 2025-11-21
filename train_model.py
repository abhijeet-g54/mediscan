# train_model.py
"""
Robust training script for MediScan.

- Auto-detects dataset format:
    * TEXT mode: expects 'Symptom' (text) + 'Disease'
    * WIDE mode: expects 'Disease' and many symptom columns (one column per symptom)
- Trains an XGBoost classifier (hist) when available; otherwise falls back to HistGradientBoostingClassifier.
- Saves a single package: models/symptom_disease_model.pkl with keys:
    { "model", "label_encoder", "vectorizer" or None, "features" or None, "format" }
- Safe defaults for modest RAM. Adjust params at top if needed.
"""

import os
import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# try xgboost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# fallback
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HGB_AVAILABLE = True
except Exception:
    HGB_AVAILABLE = False

# -------------------------
# Configuration
# -------------------------
DATA_PATH = Path("data/processed/disease_symptom_dataset.csv")   
MODEL_OUT = Path("models/symptom_disease_model.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.20

# TF-IDF settings (used when text mode)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# XGBoost / HGB settings (conservative defaults)
XGB_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "n_jobs": 4,   # adjust down to 1 or 2 if memory issues
}

HGB_PARAMS = {
    "max_iter": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
}

# -------------------------
# Helpers
# -------------------------
def die(msg):
    print("ERROR:", msg)
    sys.exit(1)

def ensure_dirs():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

def load_data(path: Path):
    if not path.exists():
        die(f"Processed dataset not found at: {path}\nPlace your processed CSV there (or set DATA_PATH correctly).")
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")
    return df

def detect_format(df: pd.DataFrame):
    # If single text column 'Symptom' exists -> text mode
    if "Symptom" in df.columns and "Disease" in df.columns:
        return "text"
    # Otherwise, if 'Disease' exists and at least one other column -> wide
    if "Disease" in df.columns:
        # treat as wide if more than 2 columns (Disease + symptom columns)
        if len(df.columns) >= 3:
            return "wide"
    return None

# -------------------------
# Text pipeline
# -------------------------
def train_text_pipeline(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # ensure columns
    if "Symptom" not in df.columns or "Disease" not in df.columns:
        die("Text mode requires 'Symptom' and 'Disease' columns in the CSV.")

    df = df.dropna(subset=["Symptom", "Disease"]).reset_index(drop=True)

    texts = df["Symptom"].astype(str)
    labels = df["Disease"].astype(str)

    print("Building TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    print("TF-IDF shape:", X.shape)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print("Number of classes:", len(le.classes_))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("Train/Val split:", X_train.shape[0], X_val.shape[0])

    model = train_model(X_train, y_train, X_val, y_val)

    # evaluate
    evaluate_model(model, X_val, y_val, le)

    # save package
    package = {
        "model": model,
        "label_encoder": le,
        "vectorizer": vectorizer,
        "features": None,
        "format": "text"
    }
    joblib.dump(package, MODEL_OUT)
    print("Saved model package to:", MODEL_OUT)

# -------------------------
# Wide pipeline
# -------------------------
def train_wide_pipeline(df: pd.DataFrame):
    # expects 'Disease' column and many symptom columns (binary/0-1/numeric)
    if "Disease" not in df.columns:
        die("Wide mode requires 'Disease' column.")

    df = df.dropna(subset=["Disease"]).reset_index(drop=True)

    # features are all columns except Disease
    features = [c for c in df.columns if c != "Disease"]
    if not features:
        die("No symptom feature columns found for wide format.")

    print(f"Detected {len(features)} feature columns.")

    # coerce to numeric 0/1
    X_df = df[features].fillna(0)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X_df.values
    y_labels = df["Disease"].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    print("Number of classes:", len(le.classes_))

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("Train/Val split:", X_train.shape[0], X_val.shape[0])

    model = train_model(X_train, y_train, X_val, y_val)

    evaluate_model(model, X_val, y_val, le)

    package = {
        "model": model,
        "label_encoder": le,
        "vectorizer": None,
        "features": features,
        "format": "wide"
    }
    joblib.dump(package, MODEL_OUT)
    print("Saved model package to:", MODEL_OUT)

# -------------------------
# Train model helper
# -------------------------
def train_model(X_train, y_train, X_val, y_val):
    # prefer XGBoost if available
    if XGB_AVAILABLE:
        print("Training XGBoost (hist) ...")
        model = XGBClassifier(**XGB_PARAMS)
        # safe fit without early_stopping arguments (avoid API mismatch)
        model.fit(X_train, y_train)
        return model
    elif HGB_AVAILABLE:
        print("XGBoost not available — training HistGradientBoostingClassifier ...")
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X_train, y_train)
        return model
    else:
        die("No suitable model library found (install xgboost or upgrade scikit-learn).")

# -------------------------
# Evaluate helper
# -------------------------
def evaluate_model(model, X_val, y_val, label_encoder):
    try:
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"Validation Accuracy: {acc:.4f}")
        print("Classification Report (top classes):")
        # print full report (may be large)
        print(classification_report(y_val, preds, target_names=label_encoder.classes_, zero_division=0))
    except Exception as e:
        print("Warning: evaluation failed:", e)

# -------------------------
# Main
# -------------------------
def main():
    ensure_dirs()
    df = load_data(DATA_PATH)
    fmt = detect_format(df)
    if fmt is None:
        die("Unable to detect dataset format. Ensure 'Disease' column exists and either a 'Symptom' text column (text mode) or many symptom columns (wide mode).")
    print("Detected dataset format:", fmt)

    if fmt == "text":
        train_text_pipeline(df)
    else:
        train_wide_pipeline(df)

if __name__ == "__main__":
    main()
