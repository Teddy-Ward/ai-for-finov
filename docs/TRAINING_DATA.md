# Training Data & Model Development

## Human-in-the-Loop Workflow

Both the **Auditor Agent** and **Visual Forensic Scanner** collect training data through human review:

1. **AI Analysis**: System performs automated analysis
2. **Human Review**: Expert reviews and optionally edits results
3. **Change Analysis**: AI asks "Why did you change this?" and analyses reasoning
4. **Data Storage**: Saves to JSON with original output + human corrections + analysis
5. **Model Training**: Use accumulated data to train proprietary models

## Automatic Data Collection

The **Visual Forensic Scanner** automatically saves every analysis as structured JSON:

- **No human interaction required** - saves immediately after analysis
- **Pre-computed features** ready for ML training
- **Training labels** automatically generated
- **Feature vectors** include: similarity scores, error counts, risk metrics
- **100% coverage** - every document analysed becomes training data

## Building Your Own Models

Replace expensive LLM API calls with your proprietary models:

### Step 1: Data Collection

- Run analyses on diverse documents
- Collect both authentic and tampered samples
- Ensure balanced dataset (50/50 split recommended)

### Step 2: Feature Engineering

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

### Step 3: Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

### Step 4: Deployment

- Replace LLM analysis with your trained model
- Integrate into existing workflow
- Monitor performance and retrain periodically

## Benefits of Proprietary Models

- **Cost Savings**: No per-document API charges
- **Speed**: Faster inference without network calls
- **Privacy**: All data stays on-premise
- **Customisation**: Optimised for your specific fraud patterns
- **Independence**: No reliance on external AI providers

## Training Data Documentation

See the following files for complete documentation on training data structure and ML use cases:

- [`visual_forensic_poc/TRAINING_DATA_SCHEMA.md`](visual_forensic_poc/TRAINING_DATA_SCHEMA.md) - Visual Forensic Scanner training data
- [`auditor_agent_poc/TRAINING_DATA_SCHEMA.md`](auditor_agent_poc/TRAINING_DATA_SCHEMA.md) - Auditor Agent training data
- [`market_finder_poc/TRAINING_DATA_SCHEMA.md`](market_finder_poc/TRAINING_DATA_SCHEMA.md) - Market Finder training data
