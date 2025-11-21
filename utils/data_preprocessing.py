"""
utils/data_preprocessing.py

Functions to help convert a row-wise symptom dataset into the wide (one-hot) format,
clean column names, and save a processed CSV ready for model training.
"""

import pandas as pd
import os
from collections import defaultdict

def clean_column_name(s):
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    return s

def rowwise_to_wide(df_rowwise, id_col="id", symptom_col="symptoms", disease_col="Disease", sep=";"):
    """
    Convert a dataset where each row contains a delimited list of symptoms into a wide binary matrix.
    df_rowwise: DataFrame with at least symptom_col and disease_col (and optional id_col)
    symptom_col: column name that contains delimited symptom lists
    sep: delimiter used in the symptom list (e.g., ';' or ',')
    Returns: wide_df with one-hot symptom columns + Disease column
    """
    temp = []
    for _, r in df_rowwise.iterrows():
        sid = r.get(id_col, None) if id_col in df_rowwise.columns else None
        disease = r[disease_col]
        raw = r[symptom_col]
        if pd.isna(raw):
            symptoms = []
        else:
            if isinstance(raw, str):
                symptoms = [s.strip().lower() for s in raw.split(sep) if s.strip()]
            elif isinstance(raw, (list, tuple)):
                symptoms = [s.strip().lower() for s in raw]
            else:
                symptoms = []
        temp.append({"id": sid, "disease": disease, "symptoms": symptoms})
    # collect unique symptoms
    all_symptoms = set()
    for x in temp:
        all_symptoms.update(x["symptoms"])
    all_symptoms = sorted(all_symptoms)
    rows = []
    for x in temp:
        row = {s: 0 for s in all_symptoms}
        for s in x["symptoms"]:
            row[s] = 1
        row["Disease"] = x["disease"]
        rows.append(row)
    wide = pd.DataFrame(rows)
    # normalize column names
    wide.columns = [clean_column_name(c) if c != "Disease" else "Disease" for c in wide.columns]
    return wide

def save_processed(df, out_path="data/processed/disease_symptom_dataset.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")
