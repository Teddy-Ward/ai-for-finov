# Finova AI Proofs of Concept (PoCs)

This repository contains multiple proofs of concept demonstrating how modern AI can be used to add operational and strategic value to the Finova platform. **Now includes human-in-the-loop training data collection for building proprietary models.**

## Quick Start

1. **[Setup Instructions](docs/SETUP.md)** - Install dependencies and configure environment
2. **[Running the POCs](docs/RUNNING_POCS.md)** - Launch and use each application
3. **[Training Data Guide](docs/TRAINING_DATA.md)** - Build proprietary models from collected data
4. **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
5. **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Directory layout and organisation

## What''s Included

### 1. Auditor Agent (AI Co-Pilot)

AI-powered mortgage auditing with RAG capabilities and human-in-the-loop training.

- **Tech**: LangGraph, FAISS, GPT-4-turbo
- **Features**: Compliance checking, consistency validation, expert feedback collection
- **Training Data**: Automatic capture of human corrections and reasoning

### 2. Market Opportunity Finder

Unsupervised learning dashboard discovering hidden customer segments.

- **Tech**: K-Means, UMAP, Plotly
- **Features**: Interactive clustering visualisation, persona analysis
- **Training Data**: Automatic export of clustering results

### 3. Visual Forensic Scanner Pro

Multi-module document tampering detection with automatic training data generation.

- **Tech**: GPT-4o Vision, OpenCV, SSIM, ELA
- **Features**: 5-module forensic analysis, human review, automatic data collection
- **Training Data**: Pre-computed feature vectors ready for ML training

### 4. Cluster Analysis Methods

Statistical methods for determining optimal cluster numbers.

- **Tech**: Elbow Method, Silhouette Analysis, KneeLocator
- **Features**: Interactive k-selection, automatic optimal k detection

## Key Value Propositions

### Build Your Data Moat

Every interaction becomes training data. Replace expensive LLM API calls with proprietary models:

- **Cost Savings**: No per-request API charges
- **Performance**: Faster inference without network calls
- **Privacy**: All data stays on-premise
- **Specialisation**: Optimised for your specific use cases

### Human-in-the-Loop Learning

Capture expert knowledge systematically:

- AI performs initial analysis
- Experts review and correct
- System asks "Why did you change this?"
- Saves corrections as structured training data

### Automatic Data Collection

The Visual Forensic Scanner saves every analysis automatically:

- No manual export needed
- Pre-computed features ready for ML
- 100% coverage - every document analysed becomes training data

## Documentation

All documentation has been organised into focused guides in the [`docs/`](docs/) folder:

- **[SETUP.md](docs/SETUP.md)** - Prerequisites, installation, environment configuration
- **[RUNNING_POCS.md](docs/RUNNING_POCS.md)** - Detailed usage instructions for each POC
- **[TRAINING_DATA.md](docs/TRAINING_DATA.md)** - Building and deploying proprietary models
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues, solutions, performance notes
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Directory layout and file organisation

## Technical Stack

**AI/ML**

- LangChain/LangGraph for agent orchestration
- OpenAI GPT-4-turbo & GPT-4o for reasoning and vision
- FAISS for vector similarity search
- scikit-learn for clustering and preprocessing

**Data Science**

- UMAP for dimensionality reduction
- K-Means clustering
- Elbow Method & Silhouette Analysis

**Computer Vision**

- OpenCV for image processing
- scikit-image for forensic analysis
- SSIM for structural similarity
- ELA for manipulation detection

**Development**

- Streamlit for rapid prototyping
- Plotly for interactive visualisations
- PyMuPDF for document processing
- Pandas for data manipulation

## Training Data Schemas

Each POC includes comprehensive training data documentation:

- [`auditor_agent_poc/TRAINING_DATA_SCHEMA.md`](auditor_agent_poc/TRAINING_DATA_SCHEMA.md)
- [`visual_forensic_poc/TRAINING_DATA_SCHEMA.md`](visual_forensic_poc/TRAINING_DATA_SCHEMA.md)
- [`market_finder_poc/TRAINING_DATA_SCHEMA.md`](market_finder_poc/TRAINING_DATA_SCHEMA.md)

## Next Steps

1. Complete [setup](docs/SETUP.md)
2. Run the [POCs](docs/RUNNING_POCS.md) with your data
3. Collect training samples (target: 100-1000+ per use case)
4. Follow the [training guide](docs/TRAINING_DATA.md) to build proprietary models
5. Deploy models to replace expensive API calls

## Presentation

View the full strategic presentation: [index.html](index.html)

- Open in browser to see the complete "Beyond the Wrapper" narrative
- Explains data moat strategy and integration approach
