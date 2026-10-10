import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
MODEL_PATH = APP_DIR / "customer-segmentation.pkl"
DATA_PATH = APP_DIR / "Mall_Customers.csv"


@st.cache_resource
def load_model_bundle(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_customer_data(data_path):
    return pd.read_csv(data_path)


# Cluster information
cluster_info = {
    0: {
        'name': 'Careful Customers',
        'description': 'Low Income, Low Spending',
        'strategy': 'Budget-friendly products, discount offers',
        'color': '#2ecc71'
    },
    1: {
        'name': 'Standard Customers',
        'description': 'Moderate Income, Moderate Spending',
        'strategy': 'Balanced product range, loyalty programs',
        'color': '#e74c3c'
    },
    2: {
        'name': 'Target Customers',
        'description': 'High Income, High Spending',
        'strategy': 'Premium products, VIP services - PRIORITY!',
        'color': '#f39c12'
    },
    3: {
        'name': 'Careless Customers',
        'description': 'Low Income, High Spending',
        'strategy': 'Installment plans, credit options',
        'color': '#9b59b6'
    },
    4: {
        'name': 'Sensible Customers',
        'description': 'High Income, Low Spending',
        'strategy': 'Quality emphasis, investment products',
        'color': '#3498db'
    }
}

# Header
st.title("Customer Segmentation System")
st.markdown("### AI-Powered Marketing Intelligence")

model_bundle = load_model_bundle(str(MODEL_PATH))
model = model_bundle["model"]
scaler = model_bundle["scaler"]
model_features = model_bundle["features"]
data = load_customer_data(str(DATA_PATH))
data["Cluster"] = model.predict(scaler.transform(data[model_features]))

# Sidebar
st.sidebar.title("Customer Profile")

st.sidebar.subheader("Enter Customer Details")
annual_income = st.sidebar.slider('Annual Income (in thousands $)', 10, 150, 50, 1)
spending_score = st.sidebar.slider('Spending Score (1-100)', 1, 100, 50, 1)

st.sidebar.markdown("---")
segment_btn = st.sidebar.button("Find Customer Segment", type="primary", use_container_width=True)

# Main content
if segment_btn:
    # Predict cluster
    input_data = pd.DataFrame(
        [[annual_income, spending_score]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"],
    )[model_features]
    predicted_cluster = int(model.predict(scaler.transform(input_data))[0])
    cluster_details = cluster_info[predicted_cluster]
    
    # Display results
    st.markdown("---")
    st.header("Segmentation Results")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        # Cluster information
        st.markdown(f"""
        <div style='background-color: {cluster_details['color']}; padding: 2rem; border-radius: 1rem; color: white;'>
            <h2 style='color: white; margin: 0;'>
{cluster_details['name']}</h2>
            <h4 style='color: white; margin: 10px 0;'>{cluster_details['description']}</h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Customer Characteristics")

        # Find similar customers in this cluster
        cluster_customers = data[data['Cluster'] == predicted_cluster]

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric(
                "Average Age",
                f"{cluster_customers['Age'].mean():.0f} years"
            )

        with metrics_col2:
            st.metric(
                "Average Income",
                f"${cluster_customers['Annual Income (k$)'].mean():.0f}k"
            )

        with metrics_col3:
            st.metric(
                "Average Spending",
                f"{cluster_customers['Spending Score (1-100)'].mean():.0f}/100"
            )

        # Marketing strategy
        st.markdown("### Marketing Strategy")
        st.success(f"**Strategy:** {cluster_details['strategy']}")

        # Cluster size
        cluster_size = len(cluster_customers)
        total_customers = len(data)
        st.info(f"**Segment Size:** {cluster_size} customers ({cluster_size/total_customers*100:.1f}% of total)")

    with col2:
        # Position on scatter plot
        st.markdown("### Your Position")

        fig = go.Figure()

        # Plot all clusters
        for cluster_id in range(5):
            cluster_data = data[data['Cluster'] == cluster_id]
            info = cluster_info[cluster_id]

            fig.add_trace(go.Scatter(
                x=cluster_data['Annual Income (k$)'],
                y=cluster_data['Spending Score (1-100)'],
                mode='markers',
                name=info['name'],
                marker=dict(
                    size=8,
                    color=info['color'],
                    opacity=0.3
                )
            ))

        # Highlight the input customer
        fig.add_trace(go.Scatter(
            x=[annual_income],
            y=[spending_score],
            mode='markers',
            name='Your Customer',
            marker=dict(
                size=20,
                color='black',
                symbol='star',
                line=dict(color='yellow', width=2)
            )
        ))

        fig.update_layout(
            title="Customer Position",
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # All clusters overview
    st.markdown("---")
    st.header("All Customer Segments")

    cols = st.columns(5)
    
    for idx, (cluster_id, info) in enumerate(cluster_info.items()):
        with cols[idx]:
            cluster_customers = data[data['Cluster'] == cluster_id]
            cluster_count = len(cluster_customers)
            
            st.markdown(f"""
            <div style='background-color: {info['color']}; padding: 1rem; border-radius: 0.5rem; 
                        color: white; text-align: center; height: 200px;'>
                <h4 style='color: white; margin: 0;'>{info['name']}</h4>
                <h2 style='color: white; margin: 10px 0;'>{cluster_count}</h2>
                <p style='color: white; font-size: 0.9em; margin: 0;'>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed comparison
    st.markdown("---")
    st.subheader("Segment Comparison")
    
    # Create comparison data
    comparison_data = []
    for cluster_id in range(5):
        cluster_customers = data[data['Cluster'] == cluster_id]
        comparison_data.append({
            'Segment': cluster_info[cluster_id]['name'],
            'Count': len(cluster_customers),
            'Avg Income': f"${cluster_customers['Annual Income (k$)'].mean():.0f}k",
            'Avg Spending': f"{cluster_customers['Spending Score (1-100)'].mean():.0f}/100",
            'Avg Age': f"{cluster_customers['Age'].mean():.0f}",
            'Strategy': cluster_info[cluster_id]['strategy']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

else:
    # Initial page
    st.markdown("---")
    st.info("Enter customer income and spending score in the sidebar to find their segment")
    
    # Show overview
    st.subheader("Customer Segments Overview")
    
    # Scatter plot of all customers
    fig = px.scatter(
        data, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)',
        color='Cluster',
        color_continuous_scale=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db'],
        title='Customer Segmentation Map',
        height=500
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(coloraxis_showscale=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment cards
    st.markdown("### Five Customer Segments")
    
    cols = st.columns(5)
    
    for idx, (cluster_id, info) in enumerate(cluster_info.items()):
        with cols[idx]:
            cluster_customers = data[data['Cluster'] == cluster_id]
            
            st.markdown(f"""
            <div style='background-color: {info['color']}; padding: 1.5rem; border-radius: 0.5rem; 
                        color: white; min-height: 250px;'>
                <h4 style='color: white; margin-bottom: 1rem;'>{info['name']}</h4>
                <p style='color: white; font-size: 0.9em;'><strong>{info['description']}</strong></p>
                <p style='color: white; font-size: 0.85em; margin-top: 1rem;'>{len(cluster_customers)} customers</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Target Segment**\n\nCluster 2: High-value customers. Focus premium marketing here!")
    
    with col2:
        st.success("**Growth Opportunity**\n\nCluster 4: High income, low spending. Show them value!")
    
    with col3:
        st.warning("** Volume Play**\n\nCluster 0: Volume sales with affordable products!")
    
    # Model info
    st.markdown("---")
    st.subheader("Model Info")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm", "K-Means")
    col2.metric("Clusters", "5")
    col3.metric("Features", "2")
    col4.metric("Data Size", len(data))
