# Training Data Schema for Model Development

## Overview

Every forensic analysis automatically generates structured JSON training data. This enables you to build proprietary fraud detection models without relying on RAG (Retrieval-Augmented Generation).

## File Types

### 1. Raw Analysis Outputs (`raw_analysis_*.json`)

**Purpose**: Automatic capture of every analysis for model training
**Generated**: Automatically after each document analysis
**Format**: `raw_analysis_{document_name}_{timestamp}.json`

#### Structure:

```json
{
  "metadata": {
    "timestamp": "ISO timestamp",
    "analysis_version": "2.0_comprehensive",
    "document_name": "filename.pdf",
    "auto_saved": true,
    "requires_human_review": boolean
  },

  "analysis_outputs": {
    "data_extraction": {
      "gold_standard": {...},
      "uploaded_document": {...},
      "cross_validation": {...}
    },
    "mathematical_integrity": {...},
    "visual_forensics": {...},
    "digital_forensics": {...},
    "risk_assessment": {...},
    "llm_comprehensive_analysis": {...}
  },

  "training_labels": {
    "is_tampered": boolean,
    "is_authentic": boolean,
    "requires_manual_review": boolean,
    "confidence_band": "high|medium|low"
  },

  "feature_vectors": {
    "data_match_rate": float,
    "ssim_score": float,
    "font_consistency": float,
    "ela_manipulation_score": float,
    "has_calculation_errors": boolean,
    "has_metadata_issues": boolean,
    "discrepancy_count": int,
    "statistical_anomaly_count": int,
    "visual_deviation_count": int
  }
}
```

### 2. Human-Reviewed Training Data (`forensic_analysis_*.json`)

**Purpose**: Expert-validated corrections and feedback
**Generated**: When broker saves human review
**Format**: `forensic_analysis_{timestamp}.json`

#### Structure:

```json
{
  "metadata": {...},
  "automated_analysis": {
    "data_extraction": {...},
    "mathematical_validation": {...},
    "visual_forensics": {...},
    "digital_forensics": {...},
    "risk_assessment": {...},
    "llm_analysis": {...}
  },
  "human_expert_review": {
    "edited_audit": "...",
    "verdict": "...",
    "confidence_level": int,
    "ai_agreement": "...",
    "key_changes_noted": "...",
    "additional_notes": "...",
    "training_notes": [...]
  },
  "training_value": {
    "data_quality": "high",
    "use_cases": [...],
    "model_feedback_signals": {
      "correct_prediction": boolean,
      "human_override": boolean,
      "key_learning_points": [...]
    }
  }
}
```

## Training Data Use Cases

### 1. Supervised Learning - Binary Classification

**Task**: Classify documents as authentic or tampered
**Features**: Use `feature_vectors` from raw outputs
**Labels**: Use `training_labels.is_tampered`
**Model Types**: Random Forest, XGBoost, Neural Networks

```python
# Example training approach
X = [
    sample["feature_vectors"] for sample in training_data
]
y = [
    sample["training_labels"]["is_tampered"] for sample in training_data
]
```

### 2. Multi-Class Classification

**Task**: Classify risk level (LOW/MODERATE/HIGH/CRITICAL)
**Features**: Feature vectors + component scores
**Labels**: `risk_assessment.risk_level`

### 3. Regression Models

**Task**: Predict trust score (0-100)
**Features**: Feature vectors
**Target**: `risk_assessment.final_trust_score`

### 4. Anomaly Detection

**Task**: Identify unusual patterns
**Features**: All forensic module outputs
**Approach**: Unsupervised learning on authentic documents

### 5. LLM Fine-Tuning

**Task**: Improve reasoning and analysis quality
**Format**: Use human-reviewed data with corrections
**Input**: Raw analysis outputs
**Expected Output**: Human-corrected analysis

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a document forensic expert..."
    },
    {
      "role": "user",
      "content": "[All analysis data as prompt]"
    },
    {
      "role": "assistant",
      "content": "[Human-corrected audit report]"
    }
  ]
}
```

## Pre-computed Features

### Numerical Features (Ready for ML)

- `data_match_rate`: 0.0-1.0 (cross-validation score)
- `ssim_score`: 0.0-1.0 (visual similarity)
- `font_consistency`: 0.0-1.0 (font analysis)
- `ela_manipulation_score`: 0.0-1.0 (image tampering)
- `discrepancy_count`: int (field differences)
- `statistical_anomaly_count`: int (math flags)
- `visual_deviation_count`: int (layout changes)

### Boolean Features

- `has_calculation_errors`: Mathematical inconsistencies
- `has_metadata_issues`: Suspicious editing software

### Categorical Features

- `verdict`: AUTHENTIC, SUSPICIOUS, LIKELY_TAMPERED, TAMPERED
- `risk_level`: LOW, MODERATE, HIGH, CRITICAL
- `confidence_band`: high, medium, low

## Model Training Pipeline

### Phase 1: Data Collection (Current)

1. Run analyses on various documents
2. Collect both authentic and tampered samples
3. Ensure balanced dataset

### Phase 2: Feature Engineering

```python
import json
import pandas as pd

# Load all training data
training_files = glob.glob("training_data/raw_analysis_*.json")
data = []

for file in training_files:
    with open(file) as f:
        sample = json.load(f)
        data.append({
            **sample["feature_vectors"],
            "label": sample["training_labels"]["is_tampered"],
            "trust_score": sample["analysis_outputs"]["risk_assessment"]["final_trust_score"]
        })

df = pd.DataFrame(data)
```

### Phase 3: Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop(["label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
```

### Phase 4: Deployment

Replace LLM-based analysis with your trained model for:

- Faster inference
- Lower cost
- No API dependencies
- Proprietary IP

## Data Quality Metrics

Track these metrics in your training data:

- **Coverage**: Do you have samples for all fraud types?
- **Balance**: Similar count of authentic vs tampered?
- **Human Review Rate**: % of samples validated by experts
- **Agreement Rate**: AI vs human verdict match percentage

## Best Practices

1. **Collect Diverse Samples**: Various document types, fraud techniques
2. **Regular Human Review**: Validate AI predictions periodically
3. **Version Control**: Track analysis_version for model compatibility
4. **Data Augmentation**: Use minor variations to increase dataset size
5. **Continuous Learning**: Retrain models as new patterns emerge

## Privacy & Security

- Ensure PII is handled according to GDPR/data protection laws
- Consider anonymising extracted personal data
- Secure storage for training data
- Access controls on training_data folder

## Next Steps

1. **Collect 100+ samples** (mix of authentic and tampered)
2. **Train baseline model** using feature vectors
3. **Evaluate performance** vs LLM-based analysis
4. **Iterate**: Add features, tune hyperparameters
5. **Deploy**: Replace expensive LLM calls with your model
6. **Monitor**: Track real-world performance and retrain

---

**Goal**: Build a proprietary fraud detection model that's faster, cheaper, and doesn't rely on external APIs while maintaining or exceeding current accuracy.
