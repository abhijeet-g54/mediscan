# data_sample.py
import pandas as pd
from pathlib import Path

def create_sample(input_path="data/processed/disease_symptom_dataset.csv",
                  output_path="data/processed/disease_symptom_dataset_small.csv",
                  sample_frac=0.10,
                  random_state=42):
    """Create a random sample of the processed dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)

    print(f"🧮 Sampling {sample_frac*100:.0f}% of {len(df):,} rows...")
    sample_df = df.sample(frac=sample_frac, random_state=random_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    print(f"✅ Saved sample to: {output_path} ({len(sample_df):,} rows)")

if __name__ == "__main__":
    create_sample()
