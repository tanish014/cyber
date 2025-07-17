import pandas as pd

def preprocess_data(df):
    print("\n[INFO] Starting preprocessing...")

    # Convert 'Label' to binary: 0 for Normal, 1 for Threat
    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().lower() == 'benign' else 1)

    # Drop missing values
    df = df.dropna()

    # Drop non-numeric columns if any remain
    non_numeric = df.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"[INFO] Dropping non-numeric columns: {non_numeric.tolist()}")
        df = df.drop(columns=non_numeric)

    print("[INFO] Preprocessing complete.")
    if 'Label' in df.columns:
        print("Target value counts:\n", df['Label'].value_counts())

    return df
