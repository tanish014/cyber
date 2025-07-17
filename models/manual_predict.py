import pandas as pd

def manual_predict(input_dict, model, feature_order):
    try:
        # Align input_dict with required feature order
        input_data = [input_dict.get(f, 0) for f in feature_order]
        df = pd.DataFrame([input_data], columns=feature_order)

        prediction = model.predict(df)[0]
        label = "Normal" if prediction == 0 else "Threat"
        return label
    except Exception as e:
        raise ValueError(f"[manual_predict] Error: {e}")
    