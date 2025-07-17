import pandas as pd

def predict_on_file(df, model):
    """
    Predicts whether each row in the uploaded DataFrame is a Threat or Normal.
    
    Parameters:
        df (pd.DataFrame): The uploaded CSV data.
        model (sklearn.Model): Trained ML model with feature_names_in_.

    Returns:
        pd.DataFrame: Original data with an added 'Prediction_Label' column.
    """
    # Drop rows with missing values
    df = df.dropna()

    # Keep only numeric columns
    df = df.select_dtypes(include=['number'])

    # Get feature names used during model training
    model_features = model.feature_names_in_

    # Add any missing columns with default value 0
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Align the column order
    df = df[model_features]

    # Predict
    predictions = model.predict(df)

    # Add prediction label column
    df['Prediction_Label'] = ['Threat' if p == 1 else 'Normal' for p in predictions]

    return df
