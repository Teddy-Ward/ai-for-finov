import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from umap import UMAP

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Market Opportunity Finder",
    page_icon="🗺️",
    layout="wide"
)
st.title("🗺️ Market Opportunity Finder")
st.markdown("Discovering hidden customer segments in rejected application data using unsupervised learning.")

# --- 2. Data Loading and Caching ---
@st.cache_data
def load_and_process_data():
    """Loads, preprocesses, and clusters the application data."""
    try:
        df = pd.read_csv('../data/mock_applications.csv')
    except FileNotFoundError:
        return None

    # Define features to be used for clustering
    numerical_features = ['Income', 'CreditScore', 'LTV', 'DebtToIncomeRatio']
    categorical_features = ['EmploymentStatus']

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Preprocess the data
    X_processed = preprocessor.fit_transform(df)

    # --- Run Machine Learning ---
    # 1. K-Means Clustering to find the groups
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_processed)

    # 2. UMAP for 2D visualisation
    umap = UMAP(n_components=2, random_state=42)
    X_reduced = umap.fit_transform(X_processed)
    df['umap_x'] = X_reduced[:, 0]
    df['umap_y'] = X_reduced[:, 1]
    
    return df

# --- 3. Persona Interpretation ---
def get_persona_details(df):
    """Analyzes the clustered data to create human-readable personas."""
    personas = {}
    for i in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == i]
        # Determine the most common rejection reason for the cluster
        dominant_reason = cluster_df['RejectionReason'].mode()[0]
        
        # Create a persona based on the dominant reason
        if "trading history" in dominant_reason:
            title = "The Stable Gig-Worker"
        elif "LTV" in dominant_reason:
            title = "The Nearly-There First-Time Buyer"
        elif "Adverse credit" in dominant_reason:
            title = "The Minor Credit Blip"
        elif "Affordability" in dominant_reason:
            title = "The Slightly Overextended"
        else:
            title = f"Cluster {i}"

        # Create a summary of the cluster's characteristics
        details = (
            f"**Description:** {title}\n"
            f"- **Dominant Rejection Reason:** {dominant_reason}\n"
            f"- **Avg. Income:** £{cluster_df['Income'].mean():,.0f}\n"
            f"- **Avg. Credit Score:** {cluster_df['CreditScore'].mean():.0f}\n"
            f"- **Avg. LTV:** {cluster_df['LTV'].mean():.1f}%\n"
        )
        personas[i] = {"title": title, "details": details}
    return personas

# --- 4. Build the Streamlit UI ---

data_df = load_and_process_data()

if data_df is None:
    st.error("Error: The mock_applications.csv file was not found in the 'data' folder. Please run `generate_data.py` first.")
else:
    personas = get_persona_details(data_df)
    
    # Add cluster titles to the DataFrame for the plot legend
    data_df['persona_title'] = data_df['cluster'].map(lambda c: personas[c]['title'])

    st.header("Interactive Market Map")
    st.markdown("Each dot represents a rejected applicant. Dots are coloured by the market segment they belong to. Hover over a dot to see applicant details.")

    # Create the interactive scatter plot
    fig = px.scatter(
        data_df,
        x='umap_x',
        y='umap_y',
        color='persona_title',
        hover_data=['ApplicantID', 'Income', 'CreditScore', 'LTV', 'RejectionReason'],
        title="Visualisation of Underserved Applicant Segments"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.header("Discovered Market Segments")
    st.markdown("The algorithm automatically discovered these four distinct groups in the data, each representing a potential market opportunity.")
    
    # Display the persona descriptions
    cols = st.columns(len(personas))
    for i, col in enumerate(cols):
        with col:
            st.subheader(personas[i]['title'])
            st.markdown(personas[i]['details'])