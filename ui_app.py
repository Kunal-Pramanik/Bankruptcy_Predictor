import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("📊 Company Bankruptcy Prediction Dashboard")
st.write("Predict financial bankruptcy risk using machine learning.")

st.divider()

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("bankruptcy_model.pkl")
top_features = pd.read_csv("top_features.csv")["Feature"].tolist()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📌 Dashboard Info")

st.sidebar.info("""
Model: Random Forest  
Features: Top Financial Indicators  
Use sliders to input company financial metrics.
""")

# -----------------------------
# Input Section
# -----------------------------

st.subheader("📥 Enter Financial Indicators")

col1, col2, col3 = st.columns(3)

input_data = {}

for i, feature in enumerate(top_features):

    if i % 3 == 0:
        with col1:
            input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5)

    elif i % 3 == 1:
        with col2:
            input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5)

    else:
        with col3:
            input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5)

st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔎 Predict Bankruptcy Risk"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # -----------------------------
    # Result Card
    # -----------------------------

    if prediction == 1:
        st.error(f"⚠ High Bankruptcy Risk | Probability: {probability:.2f}")
    else:
        st.success(f"✅ Company Financially Stable | Probability: {probability:.2f}")

    st.divider()

    # -----------------------------
    # Risk Gauge
    # -----------------------------

    st.subheader("📉 Bankruptcy Risk Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': "Bankruptcy Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

# -----------------------------
# Feature Importance Section
# -----------------------------

st.divider()
st.subheader("📊 Feature Importance")

try:
    importance = pd.read_csv("feature_importance.csv")

    fig = px.bar(
        importance.head(15),
        x="Importance",
        y="Feature",
        orientation='h',
        title="Top Features Influencing Bankruptcy"
    )

    st.plotly_chart(fig, use_container_width=True)

except:
    st.info("Feature importance file not found.")

# -----------------------------
# CSV Upload Section
# -----------------------------

st.divider()
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload company financial dataset")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data[top_features])
    probabilities = model.predict_proba(data[top_features])[:,1]

    data["Bankruptcy Prediction"] = predictions
    data["Risk Probability"] = probabilities

    st.write("Prediction Results")

    st.dataframe(data)

    csv = data.to_csv(index=False)

    st.download_button(
        "Download Results",
        csv,
        "bankruptcy_predictions.csv",
        "text/csv"
    )