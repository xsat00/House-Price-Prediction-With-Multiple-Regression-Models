import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import locale

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI House Price Predictor - Made By XSAT",
    page_icon="house.png",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

.main {
    background-color: #0E2D78;
    color: white;
}

.hero {
    padding: 3rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #334155;
}

.hero h1 {
    font-size: 3rem;
    color: white;
}

.hero p {
    font-size: 1.2rem;
    color: #cbd5e1;
}

.metric-card {
    padding: 1rem;
    border-radius: 15px;
    background-color: #1e293b;
    border: 1px solid #334155;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
    <h1>🏠 AI House Price Predictor - SATHWIK</h1>
    <p>
        Predict House Prices using Multiple Machine Learning Models
        with Interactive Analytics and Model Comparison _ Made By XSAT
    </p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():

    model_names = [
        'Linear Regression',
        'Ridge Regression',
        'Lasso Regression',
        'ElasticNet Regression',
        'Polynomial Regression',
        'SGDRegressor',
        'ANN',
        'Random Forest Regressor',
        'Support Vector Regressor',
        'LGBM',
        'XGBoost',
        'KNN',
        'Huber Regressor'
    ]

    models = {}

    for name in model_names:
        filename = f"{name}.pkl"

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                models[name] = pickle.load(file)

    return models


models = load_models()

# ================= LOAD RESULTS =================
@st.cache_data
def load_results():
    if os.path.exists("model_evaluation_results.csv"):
        return pd.read_csv("model_evaluation_results.csv")
    return None


results_df = load_results()

# ================= BEST MODEL =================
best_model_name = None

if results_df is not None and 'R2 Score' in results_df.columns:
    best_model_name = results_df.loc[results_df['R2 Score'].idxmax()]['Model']

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Prediction Settings")

if best_model_name:
    st.sidebar.success(f"Best Model Selected:\n{best_model_name}")

selected_model = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(models.keys()),
    index=list(models.keys()).index(best_model_name)
    if best_model_name in models else 0
)

# ================= MAIN INPUT SECTION =================
st.subheader("📥 Enter Property Details")

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Average Area Income",
        min_value=0.0,
        value=65000.0
    )

    house_age = st.number_input(
        "Average House Age",
        min_value=0.0,
        value=6.0
    )

    rooms = st.number_input(
        "Average Number of Rooms",
        min_value=0,
        value=0
    )

with col2:

    bedrooms = st.number_input(
        "Average Number of Bedrooms",
        min_value=0,
        value=0
    )

    population = st.number_input(
        "Area Population",
        min_value=0,
        value=35000
    )

# ================= PREDICT =================
if st.button("Predict House Price"):

    input_df = pd.DataFrame({
        'Avg. Area Income': [income],
        'Avg. Area House Age': [house_age],
        'Avg. Area Number of Rooms': [rooms],
        'Avg. Area Number of Bedrooms': [bedrooms],
        'Area Population': [population]
    })

    model = models[selected_model]

    prediction_usd = model.predict(input_df)[0]

    # USD TO INR CONVERSION
    usd_to_inr = 83

    prediction_inr = prediction_usd * usd_to_inr

    st.markdown("---")

    # ================= RESULT CARDS =================
    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Predicted Price (USD)",
            f"${prediction_usd:,.2f}"
        )

    with c2:
        
        st.metric(
            "Predicted Price (INR)",
            f"₹ {prediction_inr:,.2f}"
        )

    st.success(f"Prediction generated using {selected_model}")
    
# ================= MODEL COMPARISON =================
st.markdown("---")

st.subheader("📊 Model Evaluation")

if results_df is not None:

    st.dataframe(
        results_df,
        use_container_width=True
    )

    if st.button("Show Model Comparison Graph"):

        metric_column = None

        possible_metrics = [
            'R2 Score',
            'MAE',
            'RMSE',
            'MSE'
        ]

        for metric in possible_metrics:
            if metric in results_df.columns:
                metric_column = metric
                break

        if metric_column:

            fig3, ax3 = plt.subplots(figsize=(12, 6))

            ax3.bar(
                results_df['Model'],
                results_df[metric_column]
            )

            ax3.set_title(
                f"Models Comparison using {metric_column}"
            )

            ax3.set_ylabel(metric_column)

            plt.xticks(rotation=45)

            st.pyplot(fig3)

        else:
            st.warning("No valid metric columns found.")

else:
    st.warning("model_evaluation_results.csv not found.")

# ================= FOOTER =================
st.markdown("---")
st.caption("Built with Streamlit and Machine Learning")