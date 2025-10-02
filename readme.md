# Finova AI Proofs of Concept (PoCs)

This repository contains two proofs of concept demonstrating how modern AI can be used to add operational and strategic value to the Finova platform.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.9 or higher
- Git for cloning the repository

## Setup Instructions

Follow these steps to get the project and its dependencies set up locally.

### 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone <your-repository-url>
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

### Auditor Agent

This AI co-pilot audits mortgage applications for errors and compliance issues using a RAG (Retrieval-Augmented Generation) system.

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

**Features:**
- Automated data consistency validation
- Policy compliance checking against lender guidelines
- Structured audit reports with clear recommendations
- Real-time AI reasoning display

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
- Check that the API key has access to GPT-4-turbo

**Streamlit app won't start:**
- Make sure you're in the correct directory (`auditor_agent_poc/` or `market_finder_poc/`)
- Try `streamlit run app.py --server.port 8501` if port conflicts occur

### Performance Notes

- **Ready to run**: Both applications are ready to run immediately since data files are pre-included
- **Memory usage**: The Market Finder processes 1,000 records in real-time
- **API costs**: The Auditor Agent makes API calls for each audit (approximately $0.01-0.03 per audit)
- **Regeneration**: If you need to rebuild data files, the ingestion process may take 2-5 minutes to download the embedding model

## Project Structure

```
/
├── data/                          # Generated mock data
│   └── mock_applications.csv     # 1,000 synthetic applications
├── auditor_agent_poc/            # AI Agent application
│   ├── app.py                    # Main Streamlit application
│   ├── tools.py                  # Custom agent tools
│   ├── ingest.py                 # Knowledge base creation
│   ├── docs/                     # PDF documents for RAG
│   └── knowledge_base/           # Generated vector store
├── market_finder_poc/            # Data Science dashboard
│   └── app.py                    # Main Streamlit application
├── generate_data.py              # Mock data generator
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
└── readme.md                     # This file
```