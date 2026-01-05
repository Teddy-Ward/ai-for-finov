# Running the POCs

Each PoC is a self-contained Streamlit application. Ensure you've completed [setup](SETUP.md) before proceeding.

## Auditor Agent (with Human-in-the-Loop Training)

This AI co-pilot audits mortgage applications for errors and compliance issues using a RAG (Retrieval-Augmented Generation) system. **Now includes human review capability that captures expert feedback for training proprietary models.**

**Prerequisites:**

- OpenAI API key configured in `.env`
- Data files are already included

**Launch:**

```bash
cd auditor_agent_poc
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

---

## Market Opportunity Finder

This dashboard uses unsupervised learning to discover hidden customer segments in rejected application data.

**Prerequisites:**

- No API keys required (runs entirely locally)

**Launch:**

```bash
cd market_finder_poc
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

---

## Visual Forensic Scanner Pro (Document Tampering Detection)

A comprehensive AI-powered document verification system that detects tampering in financial documents using multi-layer forensic analysis. **Automatically generates structured training data for building proprietary fraud detection models.**

**Prerequisites:**

- OpenAI API key configured in `.env` (requires GPT-4o for vision analysis)
- Gold standard document in `visual_forensic_poc/docs/` folder

**Launch:**

```bash
cd visual_forensic_poc
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
  - 🎯 Specialised for your fraud patterns

**Technical Stack:**

- GPT-4o for vision-based data extraction and analysis
- OpenCV & scikit-image for visual forensics
- PyMuPDF for document processing
- Pillow for image manipulation
- SSIM for structural similarity analysis
- Human feedback loop for continuous improvement

---

## Cluster Analysis Methods

This application demonstrates how to determine the optimal number of clusters for K-Means analysis using statistical methods.

**Prerequisites:**

- No API keys required (runs entirely locally)

**Launch:**

```bash
cd cluster_analysis_poc
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
- Plotly for interactive visualisation
- StandardScaler and OneHotEncoder for data preprocessing
- Cached calculations for performance optimisation

**Statistical Methods Explained:**

- **WCSS (Within-Cluster Sum of Squares)**: Measures the compactness of clusters
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters
- **Elbow Point**: The optimal balance between model complexity and performance
- **Optimal k Selection**: Combines both methods for robust cluster number determination
