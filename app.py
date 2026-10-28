import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go

# 1. Page Configuration and Layout
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="❤️", layout="wide")

# Directory setup and color palette
BASE_DIR = Path(__file__).parent
TEAL, ORANGE, RED, GREEN = "#0F766E", "#EA580C", "#DC2626", "#16A34A"

# 2. Cached Resource Loaders for Model Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "KNN_heart_model.pkl")
    scaler = joblib.load(BASE_DIR / "heart_scaler.pkl")
    expected_columns = joblib.load(BASE_DIR / "heart_columns.pkl")
    return model, scaler, expected_columns

# 3. Cached Data Loader for Benchmark Comparison
@st.cache_data
def load_dataset():
    csv_path = BASE_DIR / "heart.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    for col in ["Cholesterol", "RestingBP"]:
        mean_val = df.loc[df[col] != 0, col].mean()
        df[col] = df[col].replace(0, mean_val)
    return df

model, scaler, expected_columns = load_artifacts()
dataset = load_dataset()

# Application Header
st.markdown(
    "<h1 style='margin-bottom:0'>❤️ Heart Stroke Prediction by Ujjwal</h1>"
    "<p style='color:#4B6B66;margin-top:0'>Provide the following clinical details to assess heart disease risk</p>",
    unsafe_allow_html=True,
)

# Two-column screen layout: Left for Inputs, Right for Dashboard
col_left, col_right = st.columns([1, 1.3], gap="large")

# ---------------- LEFT COLUMN: Input Form ----------------
with col_left:
    with st.container(border=True):
        st.subheader("Patient Details")
        c1, c2 = st.columns(2)
        
        age = c1.slider("Age", 18, 100, 40)
        sex = c2.selectbox("Sex", ["M", "F"])
        chest_pain_type = c1.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_blood_pressure = c2.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        cholesterol = c1.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        fasting_blood_sugar = c2.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
        restingECG = c1.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_heart_rate = c2.slider("Max Heart Rate", min_value=60, max_value=220, value=150)
        Exercise_induced_angina = c1.selectbox("Exercise-Induced Angina", ["N", "Y"])
        old_peak = c2.slider("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=0.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        
        predict_button = st.button("Check My Risk", type="primary", use_container_width=True)

# Prediction Pipeline Triggered on Button Click
if predict_button:
    raw_input = {
        "Age": age,
        "RestingBP": resting_blood_pressure,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_blood_sugar,
        "MaxHR": max_heart_rate,
        "Oldpeak": old_peak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain_type: 1,
        "RestingECG_" + restingECG: 1,
        "ExerciseAngina_" + Exercise_induced_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Populate missing one-hot encoded categories with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure column alignment matches training feature set
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    pred = int(model.predict(scaled_input)[0])
    
    # Calculate probability score if supported by the model
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(scaled_input)[0][1])
    else:
        prob = 1.0 if pred == 1 else 0.0

    # Persist results in session state across reruns
    st.session_state["result"] = {
        "pred": pred,
        "proba": prob,
        "age": age,
        "bp": resting_blood_pressure,
        "chol": cholesterol,
        "hr": max_heart_rate,
    }

# ---------------- RIGHT COLUMN: Visual Dashboard ----------------
with col_right:
    res = st.session_state.get("result")
    
    if res is None:
        st.info("👈 Enter patient details on the left and click **'Check My Risk'** to generate analysis.")
    else:
        risk_percentage = round(res["proba"] * 100)

        # 1. Primary Risk Outcome Notification
        if res["pred"] == 1:
            st.error(f"⚠️ High Risk of Heart Disease ({risk_percentage}% Probability)")
        else:
            st.success(f"✅ Low Risk of Heart Disease ({risk_percentage}% Probability)")

        # 2. Risk Meter Gauge Chart (Plotly)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                number={"suffix": "%"},
                title={"text": "Heart Disease Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RED if res["pred"] == 1 else GREEN},
                    "steps": [
                        {"range": [0, 35], "color": "#DCFCE7"},
                        {"range": [35, 65], "color": "#FEF3C7"},
                        {"range": [65, 100], "color": "#FEE2E2"},
                    ],
                },
            )
        )
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        # 3. Patient Vitals Comparison vs Dataset Baseline
        if dataset is not None:
            labels = ["Age", "Resting BP", "Cholesterol", "Max HR"]
            user_metrics = [res["age"], res["bp"], res["chol"], res["hr"]]
            avg_metrics = [
                dataset["Age"].mean(),
                dataset["RestingBP"].mean(),
                dataset["Cholesterol"].mean(),
                dataset["MaxHR"].mean(),
            ]

            bar_chart = go.Figure()
            bar_chart.add_bar(name="You", x=labels, y=user_metrics, marker_color=TEAL)
            bar_chart.add_bar(name="Average Patient", x=labels, y=avg_metrics, marker_color="#9CA3AF")
            bar_chart.update_layout(
                barmode="group",
                height=280,
                title="Your Vitals vs Dataset Average",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(bar_chart, use_container_width=True)

            # 4. 3D Cluster Visualization
            cloud = go.Figure()
            for flag, name, color in [(0, "Healthy (No Disease)", TEAL), (1, "Heart Disease", RED)]:
                subset = dataset[dataset["HeartDisease"] == flag]
                cloud.add_trace(
                    go.Scatter3d(
                        x=subset["Age"],
                        y=subset["MaxHR"],
                        z=subset["Cholesterol"],
                        mode="markers",
                        name=name,
                        marker=dict(size=3, color=color, opacity=0.5),
                    )
                )
            cloud.add_trace(
                go.Scatter3d(
                    x=[res["age"]],
                    y=[res["hr"]],
                    z=[res["chol"]],
                    mode="markers",
                    name="Your Position",
                    marker=dict(size=8, color=ORANGE, symbol="diamond"),
                )
            )
            cloud.update_layout(
                height=380,
                title="3D Patient Distribution (Rotate to inspect positioning)",
                scene=dict(xaxis_title="Age", yaxis_title="Max HR", zaxis_title="Cholesterol"),
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(cloud, use_container_width=True)
        else:
            st.warning("ℹ️ Place 'heart.csv' in the project directory to enable comparative and 3D visual charts.")

# Model Metadata Details
with st.expander("About This Model"):
    st.markdown(
        "Model: **K-Nearest Neighbors (KNN)** trained on the Heart Failure Prediction dataset.\n\n"
        "| Metric | Value |\n|---|---|\n| Accuracy | ~84% |\n| Preprocessing | StandardScaler + One-Hot Encoding |"
    )

st.caption("Disclaimer: This application is for educational purposes only and is not a substitute for professional medical advice.")