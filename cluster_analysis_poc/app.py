import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator # A handy library to find the "elbow" point

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Cluster Analysis Methods",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Determining the Optimal Number of Clusters")
st.markdown("This app demonstrates two common methods, the Elbow Method and the Silhouette Score, for finding the optimal 'k' in a K-Means clustering problem.")

# --- 2. Data Loading and Processing (Cached for performance) ---
@st.cache_data
def load_and_process_data():
    """Loads the mock data and prepares it for clustering."""
    try:
        df = pd.read_csv('../data/mock_applications.csv')
    except FileNotFoundError:
        return None

    numerical_features = ['Income', 'CreditScore', 'LTV', 'DebtToIncomeRatio']
    categorical_features = ['EmploymentStatus']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(df)
    return X_processed

# --- 3. K-Means Calculation (Cached for performance) ---
@st.cache_data
def calculate_kmeans_metrics(_processed_data, max_k=10):
    """
    Runs K-Means for a range of k values and calculates WCSS (inertia)
    and Silhouette Scores for each.
    """
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(_processed_data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(_processed_data, kmeans.labels_))
    
    return k_range, wcss, silhouette_scores

# --- 4. Main Application Logic ---
X_processed = load_and_process_data()

if X_processed is None:
    st.error("Error: The mock_applications.csv file was not found in the 'data' folder. Please run `generate_data.py` from the root directory first.")
else:
    # A slider to control the maximum k to test
    max_clusters = st.slider("Select the maximum number of clusters (k) to test:", 2, 15, 10)
    
    # Calculate the metrics
    k_range, wcss, silhouette_scores = calculate_kmeans_metrics(X_processed, max_clusters)

    st.markdown("---")
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)

    with col1:
        st.header("The Elbow Method (WCSS)")
        
        # Find the elbow point automatically
        kn = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
        elbow_point = kn.elbow if kn.elbow else "Not found"

        # Create the plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(k_range), y=wcss, mode='lines+markers', name='WCSS'))
        if elbow_point != "Not found":
            fig1.add_vline(x=elbow_point, line_width=2, line_dash="dash", line_color="red", 
                          annotation_text=f"Elbow at k={elbow_point}", annotation_position="top left")
        
        fig1.update_layout(
            title="WCSS vs. Number of Clusters (k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Within-Cluster Sum of Squares (WCSS)"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"The 'elbow' represents the point of diminishing returns. Based on this plot, the optimal number of clusters is likely **{elbow_point}**.")

    with col2:
        st.header("The Silhouette Score")

        # Find the max silhouette score
        max_score = max(silhouette_scores)
        optimal_k_silhouette = k_range[silhouette_scores.index(max_score)]

        # Create the plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'))
        fig2.add_vline(x=optimal_k_silhouette, line_width=2, line_dash="dash", line_color="green",
                       annotation_text=f"Peak at k={optimal_k_silhouette}", annotation_position="top right")

        fig2.update_layout(
            title="Average Silhouette Score vs. Number of Clusters (k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Average Silhouette Score"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"The Silhouette Score measures how well-separated the clusters are. The highest score is preferred. Based on this plot, the optimal number of clusters is **{optimal_k_silhouette}**.")