import pandas as pd
df = pd.read_csv("data/processed/disease_symptom_dataset.csv")
counts = df["Disease"].value_counts()
print("Total samples:", len(df))
print("Unique classes:", counts.shape[0])
print("Top 10 classes:\n", counts.head(10))
print("Classes with 1 sample:", (counts==1).sum())
print("Classes with <2 samples:", (counts<2).sum())
print("Example rare classes:", counts[counts<2].index[:20].tolist())
