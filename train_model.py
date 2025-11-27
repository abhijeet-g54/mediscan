#!/usr/bin/env python3
"""
train_model.py

Train a one-hot symptom -> disease classifier in "wide" CSV format:

CSV format expected (wide, binary symptom columns):
  Disease, symptom A, symptom B, symptom C, ...
  Malaria, 1,0,1,...
  Flu,     1,1,0,...

What this script does:
- Loads the CSV (default: data/processed/disease_symptom_dataset_small.csv)
- Uses all columns except "Disease" as binary symptom features (exact column names preserved)
- Drops rare disease classes with 1 sample (because stratified split needs >=2 samples per class)
- Trains an XGBoost classifier (falls back to RandomForest if XGBoost isn't available)
- Saves a bundled model package (joblib) to models/symptom_disease_model.pkl containing:
    { "model": model, "label_encoder": le, "symptoms": symptoms_list }
- Also saves models/symptoms_list.json and models/label_encoder.pkl for convenience
- Prints basic metrics

Usage:
    python train_model.py --data data/processed/disease_symptom_dataset.csv

Notes:
- Training uses the binary symptom columns directly. Intensity/duration multipliers are applied
  at inference time in the app (so they are not part of training features here).
- The script is conservative with memory: XGBoost is configured to use single thread by default.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

RANDOM_STATE = 42
MODEL_DIR = Path("models")
DEFAULT_DATA_PATH = Path("data/processed/disease_symptom_dataset_small.csv")


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "Disease" not in df.columns:
        # Some datasets might use lowercase 'disease'
        lowcols = [c.lower() for c in df.columns]
        if "disease" in lowcols:
            # rename that column back to "Disease"
            real = df.columns[lowcols.index("disease")]
            df = df.rename(columns={real: "Disease"})
        else:
            raise ValueError("Dataset must include a 'Disease' column.")

    # Ensure the Disease column is string
    df["Disease"] = df["Disease"].astype(str).str.strip()

    # Symptoms are all other columns
    symptom_cols = [c for c in df.columns if c != "Disease"]
    if len(symptom_cols) == 0:
        raise ValueError("No symptom columns found. Dataset must have columns besides 'Disease'.")

    # Ensure symptom columns are numeric (0/1). Try to coerce.
    df[symptom_cols] = df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    return df, symptom_cols


def drop_rare_classes(df, min_count=2):
    counts = df["Disease"].value_counts()
    rare = counts[counts < min_count].index.tolist()
    if len(rare) > 0:
        before = len(df)
        df = df[~df["Disease"].isin(rare)].reset_index(drop=True)
        after = len(df)
        print(f"⚠️ Dropped {len(rare)} rare classes (count < {min_count}). Rows: {before} -> {after}")
    return df


def train_xgb(X_train, y_train, X_val, y_val):
    try:
        import xgboost as xgb
        # Using conservative settings for low-memory environments
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,            # keep to 1 to reduce memory usage
            tree_method="hist",  # faster & lower memory than exact
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
        return clf
    except Exception as e:
        print("⚠️ XGBoost not available or failed to initialize. Falling back to RandomForest.")
        print("   Reason:", e)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
        clf.fit(X_train, y_train)
        return clf


def main(args):
    ensure_dirs()
    print("Loading data...")
    df, symptom_cols = load_data(args.data)

    print(f"Found {len(symptom_cols)} symptom columns.")
    print("Dropping rare classes (count < 2) to allow stratified split...")
    df = drop_rare_classes(df, min_count=2)

    # Build feature matrix and labels
    X = df[symptom_cols].values.astype(np.float32)
    y = df["Disease"].values

    # Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"Number of classes: {n_classes}")

    # Train/test split with stratify
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
        )
    except ValueError as e:
        # fallback: if stratify fails because of low counts, do a simple split
        print("⚠️ Stratified split failed:", e)
        print("-> Performing non-stratified random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=None
        )

    print("Training model (this may take a minute)...")
    model = train_xgb(X_train, y_train, X_val, y_val)

    # Evaluate
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Classification report (top 10 classes shown):")
    try:
        report = classification_report(y_val, y_pred, target_names=le.inverse_transform(np.unique(y_val)))
        print(report)
    except Exception:
        # fallback printing
        print(classification_report(y_val, y_pred))

    # Save artifacts
    bundle = {
        "model": model,
        "label_encoder": le,
        "symptoms": symptom_cols,
    }

    model_path = MODEL_DIR / "symptom_disease_model.pkl"
    label_path = MODEL_DIR / "label_encoder.pkl"
    symptoms_json = MODEL_DIR / "symptoms_list.json"

    print(f"Saving model bundle to: {model_path}")
    joblib.dump(bundle, model_path, compress=3)
    # also save label encoder and symptom list independently
    joblib.dump(le, label_path)
    with open(symptoms_json, "w", encoding="utf-8") as f:
        json.dump(symptom_cols, f, ensure_ascii=False, indent=2)

    print("All done.")
    print(f"Saved files:\n - {model_path}\n - {label_path}\n - {symptoms_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one-hot symptom -> disease model (wide CSV format).")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to wide-format CSV dataset.")
    args = p.parse_args()
    main(args)
