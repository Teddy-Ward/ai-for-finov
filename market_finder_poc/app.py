import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from umap import UMAP
import json
from datetime import datetime
import os

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

# --- 3. Training Data Export ---
def save_clustering_training_data(df, preprocessor, kmeans, personas):
    """Exports clustering results and features as JSON for model training."""
    # Create training_data directory if it doesn't exist
    training_dir = "training_data"
    os.makedirs(training_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{training_dir}/market_clustering_{timestamp}.json"
    
    # Prepare training data structure
    training_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_applicants": len(df),
            "num_clusters": len(df['cluster'].unique()),
            "algorithm": "KMeans",
            "features_used": {
                "numerical": ['Income', 'CreditScore', 'LTV', 'DebtToIncomeRatio'],
                "categorical": ['EmploymentStatus']
            }
        },
        "cluster_profiles": {},
        "applicant_records": []
    }
    
    # Add cluster-level statistics
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        training_data["cluster_profiles"][int(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "persona_title": personas[cluster_id]['title'],
            "size": len(cluster_df),
            "dominant_rejection_reason": cluster_df['RejectionReason'].mode()[0],
            "statistics": {
                "income_mean": float(cluster_df['Income'].mean()),
                "income_std": float(cluster_df['Income'].std()),
                "credit_score_mean": float(cluster_df['CreditScore'].mean()),
                "credit_score_std": float(cluster_df['CreditScore'].std()),
                "ltv_mean": float(cluster_df['LTV'].mean()),
                "ltv_std": float(cluster_df['LTV'].std()),
                "dti_mean": float(cluster_df['DebtToIncomeRatio'].mean()),
                "dti_std": float(cluster_df['DebtToIncomeRatio'].std())
            },
            "employment_distribution": cluster_df['EmploymentStatus'].value_counts().to_dict(),
            "rejection_reasons": cluster_df['RejectionReason'].value_counts().to_dict()
        }
    
    # Add individual applicant records with features and cluster assignments
    for _, row in df.iterrows():
        applicant_record = {
            "applicant_id": row['ApplicantID'],
            "features": {
                "income": float(row['Income']),
                "credit_score": int(row['CreditScore']),
                "ltv": float(row['LTV']),
                "debt_to_income_ratio": float(row['DebtToIncomeRatio']),
                "employment_status": row['EmploymentStatus']
            },
            "rejection_reason": row['RejectionReason'],
            "cluster_assignment": int(row['cluster']),
            "persona_title": personas[int(row['cluster'])]['title'],
            "umap_coordinates": {
                "x": float(row['umap_x']),
                "y": float(row['umap_y'])
            }
        }
        training_data["applicant_records"].append(applicant_record)
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    return filename

# --- 4. Persona Interpretation ---
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

# --- 5. Build the Streamlit UI ---

data_df = load_and_process_data()

if data_df is None:
    st.error("Error: The mock_applications.csv file was not found in the 'data' folder. Please run `generate_data.py` first.")
else:
    personas = get_persona_details(data_df)
    
    # Add cluster titles to the DataFrame for the plot legend
    data_df['persona_title'] = data_df['cluster'].map(lambda c: personas[c]['title'])

    # Training Data Export Section
    st.sidebar.header("🎯 Training Data Export")
    st.sidebar.markdown("""
    **Building Proprietary Models**
    
    Export clustering results as JSON to build:
    - Lender acceptance prediction models
    - Smart submission recommendations
    - Market trend analysis
    """)
    
    if st.sidebar.button("📥 Export Training Data", type="primary"):
        # Note: We need to recreate preprocessor and kmeans for export
        # This is a simplified approach - in production, save these objects
        numerical_features = ['Income', 'CreditScore', 'LTV', 'DebtToIncomeRatio']
        categorical_features = ['EmploymentStatus']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        X_processed = preprocessor.fit_transform(data_df)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_processed)
        
        filename = save_clustering_training_data(data_df, preprocessor, kmeans, personas)
        st.sidebar.success(f"✅ Training data exported!\n\n`{filename}`")
        st.sidebar.info(f"📊 Captured {len(data_df)} applicant records with cluster assignments and features.")

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
    
    # Training Data Information
    st.header("📈 Path to Proprietary Models")
    st.markdown("""
    ### From Clustering to Prediction
    
    **Current State:** Unsupervised clustering reveals hidden market segments for strategic insights.
    
    **Future Vision:** The exported training data enables building proprietary predictive models:
    
    1. **Lender Acceptance Prediction**
       - Train models on our cross-lender rejection/acceptance dataset
       - Predict which lenders will accept specific applicant profiles
       - Build a "Smart Submission Engine" unique to Finova
    
    2. **Market Trend Analysis**
       - Track how cluster compositions change over time
       - Identify emerging underserved segments
       - Predict market shifts before competitors
    
    3. **Risk Scoring Models**
       - Develop custom risk metrics based on rejection patterns
       - Fine-tune models specific to UK mortgage market
       - Own the IP and eliminate API dependencies
    
    **Training Data Captured:**
    - Applicant feature vectors (income, credit, LTV, DTI, employment)
    - Cluster assignments and persona labels
    - Rejection reasons and patterns
    - Statistical profiles for each market segment
    
    **Competitive Advantage:** This cross-lender market view is impossible to replicate. 
    Individual lenders only see their own data - we see the entire market.
    """)
    
    # Display sample training data schema
    with st.expander("📋 View Training Data Schema"):
        st.json({
            "metadata": {
                "timestamp": "2026-01-05T10:30:00",
                "total_applicants": 1000,
                "num_clusters": 4,
                "algorithm": "KMeans"
            },
            "cluster_profiles": {
                "0": {
                    "persona_title": "The Stable Gig-Worker",
                    "size": 250,
                    "dominant_rejection_reason": "Insufficient trading history",
                    "statistics": {
                        "income_mean": 55000.0,
                        "credit_score_mean": 720.0
                    }
                }
            },
            "applicant_records": [
                {
                    "applicant_id": "A001",
                    "features": {
                        "income": 55000.0,
                        "credit_score": 720,
                        "ltv": 85.0,
                        "debt_to_income_ratio": 3.5,
                        "employment_status": "Self-employed"
                    },
                    "rejection_reason": "Insufficient trading history",
                    "cluster_assignment": 0,
                    "persona_title": "The Stable Gig-Worker"
                }
            ]
        })
