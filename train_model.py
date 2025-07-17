import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import os

# --- Step 1: Load dataset ---
def load_dataset():
    df = pd.read_csv("dataset/portscan-friday-no-metadata.csv")
    print(df.head())
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

# --- Step 2: Preprocess data ---
def preprocess_data(df):
    print("\n[INFO] Starting preprocessing...")

    # Convert 'Label' column to binary: 0 = Normal, 1 = Threat
    df['Label'] = df['Label'].apply(lambda x: 0 if str(x).strip().lower() == 'benign' else 1)

    # Drop missing values
    df = df.dropna()

    # Drop non-numeric columns
    non_numeric = df.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"[INFO] Dropping non-numeric columns: {non_numeric.tolist()}")
        df = df.drop(columns=non_numeric)

    print("[INFO] Preprocessing complete.")
    print("New shape:", df.shape)
    print("Target value counts:\n", df['Label'].value_counts())
    return df

# --- Step 3: Undersample majority class + Train ---
def split_and_train(df):
    # ✅ Use only the features used in manual detection
    selected_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Fwd Packet Length Max", "Bwd Packet Length Max",
        "Fwd IAT Mean", "Bwd IAT Mean"
    ]

    # Split Normal & Threat
    df_normal = df[df['Label'] == 0]
    df_threat = df[df['Label'] == 1]

    # Undersample Normal class to match Threat class
    df_normal_sampled = resample(
        df_normal,
        replace=False,
        n_samples=len(df_threat),
        random_state=42
    )

    # Combine balanced data
    df_balanced = pd.concat([df_normal_sampled, df_threat])
    df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

    print("\n[INFO] Balanced data class counts:")
    print(df_balanced['Label'].value_counts())

    X = df_balanced[selected_features]
    y = df_balanced['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Train with class_weight just in case
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Attach feature list to model for prediction use
    model.feature_names_in_ = X.columns

    return model, X_test, y_test

# --- Step 4: Evaluate the model ---
def evaluate_model(model, X_test, y_test):
    print("\n[INFO] Evaluating model...")
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print a sample prediction for verification
    print("\n🔍 Sample Predictions:")
    print("Predicted:", y_pred[:10].tolist())
    print("Actual:   ", y_test[:10].tolist())

# --- Step 5: Main block ---
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = load_dataset()
    df = preprocess_data(df)
    model, X_test, y_test = split_and_train(df)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, "models/cyber_threat_rf_model.pkl")
    print("[✅] Model saved to: models/cyber_threat_rf_model.pkl")
