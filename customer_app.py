import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

#Page Configuration
st.set_page_config(page_title="Customer Segmentation System", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f5f7fb;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
model_bundle = joblib.load(APP_DIR / "customer-segmentation.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]
model_features = model_bundle["features"]
df = pd.read_csv(APP_DIR / "Mall_Customers.csv")

st.title("Customer Segmentation System")
st.markdown("### AI powered Marketing Intelligence")

st.subheader("Cluster info")
cluster_info = {
    0: ("Careful customers", "#457b9d"),
    1: ("Standard Customers", "#6a4c93"),
    2: ("Target", "#2a9d8f"),
    3: ("Careless", "#e76f51"),
    4: ("Sensible", "#e9a820"),
}

cluster_columns = st.columns(len(cluster_info))
for column, (cluster_id, (cluster_name, cluster_color)) in zip(
    cluster_columns, cluster_info.items()
):
    with column:
        st.markdown(
            f"""
            <div style="
                background-color: {cluster_color};
                color: white;
                padding: 1rem;
                border-radius: 0.75rem;
                text-align: center;
                min-height: 5rem;
            ">
                <strong>Cluster {cluster_id}</strong><br>
                {cluster_name}
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.sidebar:
    st.title("Customer Profile")
    st.subheader("Enter Customer Details")
    annual_income = st.number_input(
        "Annual income (k$)",
        min_value=0.0,
        value=70.0,
        step=1.0,
    )
    spending_score = st.number_input(
        "Spending score (1-100)",
        min_value=1,
        max_value=100,
        value=80,
        step=1,
    )
    find_segment = st.button("Find customer segment", type="primary")

if find_segment:
    customer_data = pd.DataFrame(
        [[annual_income, spending_score]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"],
    )
    customer_data = customer_data[model_features]
    customer_scaled = scaler.transform(customer_data)
    predicted_cluster = int(model.predict(customer_scaled)[0])

    if predicted_cluster not in cluster_info:
        st.error(f"No display information is configured for cluster {predicted_cluster}.")
    else:
        cluster_name, cluster_color = cluster_info[predicted_cluster]
        st.subheader("Prediction Result")
        result_columns = st.columns(3)
        result_columns[0].metric("Annual income", f"${annual_income:.0f}k")
        result_columns[1].metric("Spending score", str(spending_score))
        result_columns[2].metric("Predicted cluster", f"Cluster {predicted_cluster}")
        st.markdown(
            f"""
            <div style="
                background-color: {cluster_color};
                color: white;
                padding: 1.25rem;
                border-radius: 0.75rem;
                margin-top: 0.5rem;
                font-size: 1.1rem;
                text-align: center;
            ">
                <strong>Customer segment: {cluster_name}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )