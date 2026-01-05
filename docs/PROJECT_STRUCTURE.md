# Project Structure

```
finova_ai_poc/
├── data/
│   └── mock_applications.csv     # 1,000 synthetic mortgage applications
├── auditor_agent_poc/            # AI Agent application with HITL
│   ├── app.py                    # Main Streamlit application
│   ├── tools.py                  # Custom agent tools
│   ├── ingest.py                 # Knowledge base creation
│   ├── docs/                     # PDF documents for RAG
│   ├── knowledge_base/           # Generated vector store
│   ├── training_data/            # Expert feedback & corrections (JSON)
│   └── TRAINING_DATA_SCHEMA.md   # Training data documentation
├── visual_forensic_poc/          # Document forensics with ML training
│   ├── app.py                    # Main Streamlit application
│   ├── forensic_logic.py         # 5-module forensic analysis engine
│   ├── template_library.py       # Gold standard configuration
│   ├── docs/                     # Gold standard documents
│   ├── training_data/            # Auto-saved analysis outputs (JSON)
│   └── TRAINING_DATA_SCHEMA.md   # Training data documentation
├── market_finder_poc/            # Data Science dashboard
│   ├── app.py                    # Main Streamlit application
│   ├── training_data/            # Clustering results (JSON)
│   └── TRAINING_DATA_SCHEMA.md   # Training data documentation
├── cluster_analysis_poc/         # Cluster analysis methods
│   └── app.py                    # Main Streamlit application
├── docs/                         # Documentation
│   ├── SETUP.md                  # Setup instructions
│   ├── RUNNING_POCS.md           # How to run each POC
│   ├── TRAINING_DATA.md          # Model training guide
│   ├── TROUBLESHOOTING.md        # Common issues & solutions
│   └── PROJECT_STRUCTURE.md      # This file
├── generate_data.py              # Mock data generator
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
├── index.html                    # Presentation deck
└── readme.md                     # Main documentation
```

## Key Directories

### `/data/`

Contains synthetic mortgage application data used by the Market Finder and Cluster Analysis POCs.

### `/auditor_agent_poc/`

AI-powered mortgage auditing system with RAG capabilities and human-in-the-loop training data collection.

### `/visual_forensic_poc/`

Multi-module document forensic analysis system with automatic training data generation.

### `/market_finder_poc/`

Unsupervised learning dashboard for discovering hidden customer segments in rejected applications.

### `/cluster_analysis_poc/`

Statistical methods for determining optimal cluster numbers (Elbow Method & Silhouette Analysis).

### `/docs/`

Comprehensive documentation split into focused guides for easier navigation.

## Training Data Organisation

Each POC that collects training data has its own `training_data/` directory:

- **Auditor Agent**: `auditor_agent_poc/training_data/` - Expert corrections and reasoning
- **Visual Forensic**: `visual_forensic_poc/training_data/` - Automatic analysis outputs
- **Market Finder**: `market_finder_poc/training_data/` - Clustering results

Each also includes a `TRAINING_DATA_SCHEMA.md` file documenting the JSON structure and use cases.
