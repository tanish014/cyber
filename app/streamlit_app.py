import streamlit as st
import pandas as pd
import joblib
from models.predict import predict_on_file
from models.manual_predict import manual_predict
import plotly.express as px

# Set page layout (must be FIRST!)
st.set_page_config(layout="wide", page_title="AI-Based Cyber Threat Detector")

# Load the model once (with feature names)
@st.cache_resource
def load_model():
   model = joblib.load("app/models/cyber_threat_rf_model.pkl")
    feature_names = model.feature_names_in_.tolist()
    return model, feature_names

model, feature_order = load_model()

st.title("🛡️ AI-Based Cyber Threat Detector")

# --- Top Tabs ---
tab1, tab2 = st.tabs(["📂 Upload CSV File", "📝 Manual Detection"])

# ========== TAB 1: UPLOAD CSV ==========
with tab1:
    st.subheader("📁 Upload a network traffic CSV file")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            st.subheader("📊 Uploaded Data Preview:")
            st.dataframe(df, use_container_width=True)

            # Run prediction
            result_df = predict_on_file(df, model)

            st.subheader("📌 Prediction Results (Table):")
            st.dataframe(result_df[['Prediction_Label']].value_counts().reset_index(name='Count'))

            st.subheader("📊 Prediction Results (Bar Chart):")
            chart_data = result_df['Prediction_Label'].value_counts().reset_index()
            chart_data.columns = ['Label', 'Count']

            color_map = {
                'Threat': 'crimson',
                'Normal': 'royalblue'
            }
            fig = px.bar(
                chart_data,
                x='Label',
                y='Count',
                color='Label',
                color_discrete_map=color_map,
                title="Threat vs Normal"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to process the file: {e}")

# ========== TAB 2: MANUAL DETECTION ==========
with tab2:
    st.subheader("✍️ Manually Enter Log Features for Prediction")

    important_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Fwd Packet Length Max", "Bwd Packet Length Max",
        "Fwd IAT Mean", "Bwd IAT Mean"
    ]

    manual_input = {}
    cols = st.columns(3)

    for i, feature in enumerate(important_features):
        with cols[i % 3]:
            manual_input[feature] = st.number_input(f"{feature}", value=0.0, step=1.0)

    if st.button("🔍 Detect"):
        try:
            filled_input = {feat: manual_input.get(feat, 0.0) for feat in feature_order}
            result = manual_predict(filled_input, model, feature_order)

            if result == "Threat":
                st.markdown(
                    "<div style='padding: 12px; background-color: #ffe6e6; border-left: 6px solid red;'>"
                    "<span style='color: red; font-weight: bold; font-size: 18px;'>🚨 Prediction: Threat</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='padding: 12px; background-color: #e6ffe6; border-left: 6px solid green;'>"
                    "<span style='color: green; font-weight: bold; font-size: 18px;'>✅ Prediction: Normal</span>"
                    "</div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"❌ Error in manual prediction: {e}")
