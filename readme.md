# Finova AI Proofs of Concept (PoCs)

This repository contains multiple proofs of concept demonstrating how modern AI can be used to add operational and strategic value to the Finova platform. **Now includes human-in-the-loop training data collection for building proprietary models.**

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.9 or higher
- Git for cloning the repository

## Setup Instructions

Follow these steps to get the project and its dependencies set up locally.

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone https://github.com/Teddy-Ward/ai-for-finov.git
cd finova_ai_poc
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

With your virtual environment active, install all the required Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

The Auditor Agent requires an API key for a Large Language Model (e.g., OpenAI).

1. Create a new file in the root directory named `.env`.
2. Add your API key to this file:

```
OPENAI_API_KEY="sk-..."
```

**Note**: You can get an API key from [OpenAI](https://platform.openai.com/api-keys). The Auditor Agent uses GPT-4-turbo for optimal performance.

### 5. Verify Data Files

The required data files are already included in this repository:

- `data/mock_applications.csv` - Contains 1,000 synthetic mortgage applications across 4 different customer personas
- `auditor_agent_poc/knowledge_base/faiss_index/` - Pre-built knowledge base for the Auditor Agent

**Note**: If you need to regenerate the data files for any reason, you can run:

```bash
# Generate mock data (optional - already included)
python generate_data.py

# Rebuild knowledge base (optional - already included)
cd auditor_agent_poc
python ingest.py
```

## Running the Proofs of Concept (PoCs)

Each PoC is a self-contained Streamlit application.

### Auditor Agent (with Human-in-the-Loop Training)

This AI co-pilot audits mortgage applications for errors and compliance issues using a RAG (Retrieval-Augmented Generation) system. **Now includes human review capability that captures expert feedback for training proprietary models.**

**Prerequisites:**

- OpenAI API key (see step 4 above)
- Data files are already included (see step 5 above)

**Setup Steps:**

1. **Ensure your virtual environment is activated** (you should see `(venv)` in your terminal prompt)

2. Navigate into the agent's directory:

```bash
cd auditor_agent_poc
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

**How to Use:**

- Select an Applicant ID from the dropdown (10001-11000)
- Click "Run Audit" to start the AI audit process
- The agent will automatically fetch application data and check compliance against the lender policy
- Review the structured audit report with internal consistency checks and compliance status
- **NEW**: Review and edit the AI's audit if needed
- **NEW**: Provide feedback on why you made changes
- **NEW**: Save corrections as training data for future model improvement

**Features:**

- Automated data consistency validation
- Policy compliance checking against lender guidelines
- Structured audit reports with clear recommendations
- Real-time AI reasoning display
- **Human-in-the-Loop Review**: Side-by-side comparison of AI vs human assessment
- **AI-Powered Change Analysis**: System asks "Why did you change this?" and analyzes corrections
- **Training Data Collection**: Automatically saves expert feedback in JSON format
- **Error Categorization**: Classifies AI errors (factual, logic, policy, completeness, etc.)
- **Learning Priority Scoring**: Flags high-priority corrections for model retraining

**Training Data Output:**

- Saves to `auditor_agent_poc/training_data/`
- Includes original AI output, human corrections, and reasoning analysis
- JSON format ready for model fine-tuning
- Tracks agreement levels, confidence scores, and learning points

### Market Opportunity Finder

This dashboard uses unsupervised learning to discover hidden customer segments in rejected application data.

**Prerequisites:**

- Data files are already included (see step 5 above)
- No API keys required (runs entirely locally)

**Setup Steps:**

1. **Ensure your virtual environment is activated** (you should see `(venv)` in your terminal prompt)

2. Navigate into the dashboard's directory:

```bash
cd market_finder_poc
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

**How to Use:**

- The application automatically loads and processes the mock application data
- View the interactive market map showing customer segments as colored clusters
- Hover over data points to see individual applicant details
- Review the discovered market segments and their characteristics

**Features:**

- **K-Means Clustering**: Automatically identifies 4 distinct customer personas
- **UMAP Visualization**: 2D projection of high-dimensional customer data
- **Interactive Plotly Charts**: Hover to explore individual applications
- **Persona Analysis**: Detailed breakdown of each discovered segment
- **Real-time Processing**: No pre-computation required

**Discovered Segments:**

- **Stable Gig-Worker**: Self-employed with insufficient trading history
- **Nearly-There First-Time Buyer**: Good credit but LTV exceeds limits
- **Minor Credit Blip**: High income but recent credit issues
- **Slightly Overextended**: Good credit but high debt-to-income ratio

**Technical Details:**

- Uses scikit-learn for clustering and preprocessing
- UMAP for dimensionality reduction and visualization
- Plotly for interactive charts
- Automatic feature scaling and categorical encoding

### Visual Forensic Scanner Pro (Document Tampering Detection)

A comprehensive AI-powered document verification system that detects tampering in financial documents using multi-layer forensic analysis. **Automatically generates structured training data for building proprietary fraud detection models.**

**Prerequisites:**

- OpenAI API key (see step 4 above) - requires GPT-4o for vision analysis
- Gold standard document in `visual_forensic_poc/docs/` folder
- No pre-training required - ready to use immediately

**Setup Steps:**

1. **Ensure your virtual environment is activated** (you should see `(venv)` in your terminal prompt)

2. Navigate into the forensic scanner directory:

```bash
cd visual_forensic_poc
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

**How to Use:**

- Upload a payslip PDF for comprehensive forensic analysis
- System automatically compares against gold standard document
- Review 5-module analysis results (Data Extraction, Math Validation, Visual Forensics, Digital Forensics, Risk Scoring)
- View AI-generated comprehensive forensic report
- **Optional**: Provide human expert review and feedback
- **Automatic**: Every analysis saves training data in JSON format

**Features:**

**Module 1: Data Extraction & Cross-Validation (D2D)**

- LLM Vision-powered OCR (GPT-4o) extracts all document fields
- Intelligent field recognition (salary, tax, NI, deductions, etc.)
- Cross-validation between gold standard and uploaded document
- Field-level discrepancy detection with severity ratings

**Module 2: Mathematical & Logical Integrity**

- Penny-perfect calculation audit (Gross - Deductions = Net)
- Tax code format validation
- NI category verification against UK standards
- Statistical improbability detection (suspicious rounding patterns)

**Module 3: Visual Forensic Analysis**

- Spatial fingerprinting using SSIM (Structural Similarity Index)
- Pixel-level layout comparison with deviation mapping
- Font weight and style profiling via edge detection
- Character kerning analysis for manual tampering detection

**Module 4: Deep Digital Forensics**

- Error Level Analysis (ELA) for image manipulation detection
- Metadata X-Ray scanning for editing software signatures
- Linguistic hallucination detection (AI-generated forgery terms)
- Compression inconsistency analysis

**Module 5: Comprehensive Risk Scoring**

- Weighted scoring across all forensic modules
- AI-powered analysis synthesis
- Fraud probability assessment
- Actionable recommendations for underwriters

**Human-in-the-Loop Training:**

- Review and edit AI analysis results
- Provide expert corrections and reasoning
- System captures feedback for model improvement
- Tracks agreement levels and learning priorities

**Automatic Training Data Collection:**

- **Raw Analysis Outputs**: Every analysis auto-saved as structured JSON
- **Pre-computed Feature Vectors**: ML-ready numerical features
- **Training Labels**: Automated classification (authentic/tampered)
- **Human Reviews**: Expert-validated corrections when provided
- Saves to `visual_forensic_poc/training_data/` folder

**Training Data Schema:**

- Complete forensic analysis results from all 5 modules
- Extracted features ready for ML model training:
  - `data_match_rate`, `ssim_score`, `font_consistency`
  - `ela_manipulation_score`, `discrepancy_count`
  - `has_calculation_errors`, `has_metadata_issues`
- Training labels: `is_tampered`, `is_authentic`, `confidence_band`
- See `TRAINING_DATA_SCHEMA.md` for full documentation

**Use Cases for Training Data:**

1. **Binary Classification**: Train model to classify authentic vs tampered
2. **Risk Score Prediction**: Regression model for trust scores
3. **Multi-class Classification**: Predict risk levels (LOW/MODERATE/HIGH/CRITICAL)
4. **LLM Fine-tuning**: Use human corrections to fine-tune reasoning
5. **Anomaly Detection**: Unsupervised learning on authentic documents

**Replace LLM with Your Own Model:**

- Collect 100+ diverse samples (authentic + tampered)
- Train classifier using pre-computed feature vectors
- Deploy your proprietary model for:
  - ⚡ Faster inference (no API calls)
  - 💰 Lower costs (no per-document charges)
  - 🔒 Data privacy (on-premise processing)
  - 🎯 Specialized for your fraud patterns

**Technical Stack:**

- GPT-4o for vision-based data extraction and analysis
- OpenCV & scikit-image for visual forensics
- PyPDF2 & pdf2image for document processing
- Pillow for image manipulation
- SSIM for structural similarity analysis
- Human feedback loop for continuous improvement

### Cluster Analysis Methods

This application demonstrates how to determine the optimal number of clusters for K-Means analysis using statistical methods.

**Prerequisites:**

- Data files are already included (see step 5 above)
- No API keys required (runs entirely locally)

**Setup Steps:**

1. **Ensure your virtual environment is activated** (you should see `(venv)` in your terminal prompt)

2. Navigate into the cluster analysis directory:

```bash
cd cluster_analysis_poc
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

**How to Use:**

- Use the slider to select the maximum number of clusters (k) to test (2-15)
- View the Elbow Method plot showing WCSS (Within-Cluster Sum of Squares) vs. number of clusters
- View the Silhouette Score plot showing cluster separation quality
- The application automatically identifies the optimal k using both methods
- Compare results from both statistical approaches

**Features:**

- **Elbow Method**: Uses WCSS to find the point of diminishing returns
- **Silhouette Analysis**: Measures cluster separation quality (range: -1 to 1, higher is better)
- **Automatic Detection**: Uses KneeLocator library to automatically find the elbow point
- **Interactive Controls**: Adjustable maximum k value for testing
- **Real-time Calculation**: Processes clustering metrics on demand

**Technical Details:**

- Uses scikit-learn for K-Means clustering and silhouette score calculation
- KneeLocator library for automatic elbow point detection
- Plotly for interactive visualization
- StandardScaler and OneHotEncoder for data preprocessing
- Cached calculations for performance optimization

**Statistical Methods Explained:**

- **WCSS (Within-Cluster Sum of Squares)**: Measures the compactness of clusters
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters
- **Elbow Point**: The optimal balance between model complexity and performance
- **Optimal k Selection**: Combines both methods for robust cluster number determination

## Troubleshooting

### Common Issues

**"ModuleNotFoundError" when running applications:**

- **Most common cause**: Virtual environment is not activated
- Ensure you've activated your virtual environment (look for `(venv)` in your terminal prompt)
- Run `pip install -r requirements.txt` to install all dependencies
- If still having issues, try: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)

**"mock_applications.csv file not found" error:**

- The data file should already be included in the repository
- If missing, run `python generate_data.py` from the root directory to regenerate it

**Auditor Agent shows "No PDF documents found":**

- The knowledge base should already be included in the repository
- If missing, run `python ingest.py` from the `auditor_agent_poc/` directory to rebuild it
- Ensure PDF documents are placed in `auditor_agent_poc/docs/` folder

**OpenAI API errors:**

- Verify your API key is correctly set in the `.env` file
- Ensure you have sufficient API credits
- Check that the API key has access to GPT-4-turbo and GPT-4o (for Visual Forensic Scanner)
- Visual Forensic Scanner requires GPT-4o with vision capabilities

**Streamlit app won't start:**

- Make sure you're in the correct directory (`auditor_agent_poc/`, `market_finder_poc/`, `visual_forensic_poc/`, etc.)
- Try `streamlit run app.py --server.port 8501` if port conflicts occur

**Training data not saving:**

- Check that `training_data/` folder exists in the respective POC directory
- Ensure you have write permissions to the directory
- Verify OpenAI API is working for change analysis features

**Visual Forensic Scanner errors:**

- Ensure you upload a PDF file (not image formats)
- Gold standard document must exist in `visual_forensic_poc/docs/` folder
- Check that all required libraries are installed: `opencv-python`, `scikit-image`, `pdf2image`
- On Windows, you may need to install Poppler for pdf2image (see library documentation)

### Performance Notes

- **Ready to run**: All applications are ready to run immediately since data files are pre-included
- **Memory usage**: The Market Finder processes 1,000 records in real-time
- **API costs**:
  - Auditor Agent: ~$0.01-0.03 per audit (GPT-4-turbo)
  - Visual Forensic Scanner: ~$0.05-0.15 per document (GPT-4o vision + analysis)
- **Regeneration**: If you need to rebuild data files, the ingestion process may take 2-5 minutes to download the embedding model
- **Training Data**: Accumulates automatically with each use - no manual export needed
- **Storage**: JSON training files are typically 10-50KB each

## Project Structure

```
/
├── data/                          # Generated mock data
│   └── mock_applications.csv     # 1,000 synthetic applications
├── auditor_agent_poc/            # AI Agent application with HITL
│   ├── app.py                    # Main Streamlit application
│   ├── tools.py                  # Custom agent tools
│   ├── ingest.py                 # Knowledge base creation
│   ├── docs/                     # PDF documents for RAG
│   ├── knowledge_base/           # Generated vector store
│   └── training_data/            # Expert feedback & corrections (JSON)
├── visual_forensic_poc/          # Document forensics with ML training
│   ├── app.py                    # Main Streamlit application
│   ├── forensic_logic.py         # 5-module forensic analysis engine
│   ├── template_library.py       # Gold standard configuration
│   ├── docs/                     # Gold standard documents
│   ├── training_data/            # Auto-saved analysis outputs (JSON)
│   └── TRAINING_DATA_SCHEMA.md   # Training data documentation
├── market_finder_poc/            # Data Science dashboard
│   └── app.py                    # Main Streamlit application
├── cluster_analysis_poc/         # Cluster analysis methods
│   └── app.py                    # Main Streamlit application
├── generate_data.py              # Mock data generator
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
└── readme.md                     # This file
```

## Training Data & Model Development

### Human-in-the-Loop Workflow

Both the **Auditor Agent** and **Visual Forensic Scanner** now collect training data through human review:

1. **AI Analysis**: System performs automated analysis
2. **Human Review**: Expert reviews and optionally edits results
3. **Change Analysis**: AI asks "Why did you change this?" and analyzes reasoning
4. **Data Storage**: Saves to JSON with original output + human corrections + analysis
5. **Model Training**: Use accumulated data to train proprietary models

### Automatic Data Collection

The **Visual Forensic Scanner** automatically saves every analysis as structured JSON:

- **No human interaction required** - saves immediately after analysis
- **Pre-computed features** ready for ML training
- **Training labels** automatically generated
- **Feature vectors** include: similarity scores, error counts, risk metrics
- **100% coverage** - every document analyzed becomes training data

### Building Your Own Models

Replace expensive LLM API calls with your proprietary models:

**Step 1: Data Collection** (Already implemented)

- Run analyses on diverse documents
- Collect both authentic and tampered samples
- Ensure balanced dataset (50/50 split recommended)

**Step 2: Feature Engineering**

```python
import json
import pandas as pd
from glob import glob

# Load training data
files = glob("visual_forensic_poc/training_data/raw_analysis_*.json")
data = [json.load(open(f)) for f in files]

# Extract features
df = pd.DataFrame([d["feature_vectors"] for d in data])
labels = [d["training_labels"]["is_tampered"] for d in data]
```

**Step 3: Model Training**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

**Step 4: Deployment**

- Replace LLM analysis with your trained model
- Integrate into existing workflow
- Monitor performance and retrain periodically

### Benefits of Proprietary Models

- **Cost Savings**: No per-document API charges
- **Speed**: Faster inference without network calls
- **Privacy**: All data stays on-premise
- **Customization**: Optimized for your specific fraud patterns
- **Independence**: No reliance on external AI providers

See `visual_forensic_poc/TRAINING_DATA_SCHEMA.md` for complete documentation on training data structure and ML use cases.
