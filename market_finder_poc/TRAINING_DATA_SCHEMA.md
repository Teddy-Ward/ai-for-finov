# Market Finder Training Data Schema

## Overview

The Market Finder automatically exports clustering results and applicant features as JSON files for training proprietary predictive models. This data captures cross-lender market intelligence that individual lenders cannot access.

## File Location

Training data is saved to: `market_finder_poc/training_data/market_clustering_YYYYMMDD_HHMMSS.json`

## JSON Structure

```json
{
  "metadata": {
    "timestamp": "ISO 8601 datetime",
    "total_applicants": "int - number of applicants analyzed",
    "num_clusters": "int - number of market segments discovered",
    "algorithm": "string - clustering algorithm used (KMeans)",
    "features_used": {
      "numerical": ["Income", "CreditScore", "LTV", "DebtToIncomeRatio"],
      "categorical": ["EmploymentStatus"]
    }
  },
  "cluster_profiles": {
    "0": {
      "cluster_id": "int - cluster identifier",
      "persona_title": "string - human-readable segment name",
      "size": "int - number of applicants in this cluster",
      "dominant_rejection_reason": "string - most common rejection reason",
      "statistics": {
        "income_mean": "float",
        "income_std": "float",
        "credit_score_mean": "float",
        "credit_score_std": "float",
        "ltv_mean": "float",
        "ltv_std": "float",
        "dti_mean": "float",
        "dti_std": "float"
      },
      "employment_distribution": {
        "Employed": "int - count",
        "Self-employed": "int - count"
      },
      "rejection_reasons": {
        "reason_string": "int - count of this rejection reason"
      }
    }
  },
  "applicant_records": [
    {
      "applicant_id": "string - unique identifier",
      "features": {
        "income": "float - annual income",
        "credit_score": "int - credit score",
        "ltv": "float - loan-to-value ratio %",
        "debt_to_income_ratio": "float - debt to income ratio",
        "employment_status": "string - Employed/Self-employed"
      },
      "rejection_reason": "string - why application was rejected",
      "cluster_assignment": "int - which cluster this applicant belongs to",
      "persona_title": "string - cluster persona name",
      "umap_coordinates": {
        "x": "float - 2D visualization x coordinate",
        "y": "float - 2D visualization y coordinate"
      }
    }
  ]
}
```

## Training Use Cases

### 1. Lender Acceptance Prediction Model

**Goal:** Predict which lenders will accept specific applicant profiles

**Training Approach:**

- Combine rejection data with historical acceptance data (when available)
- Features: applicant characteristics + lender + cluster assignment
- Target: binary acceptance/rejection or probability score
- Model: Gradient Boosting, Random Forest, or Neural Network

**Value:** "Smart Submission Engine" that recommends optimal lenders for each applicant

### 2. Market Trend Analysis

**Goal:** Track how market segments evolve over time

**Training Approach:**

- Collect clustering data over multiple time periods
- Analyze cluster drift and membership changes
- Identify emerging or shrinking segments
- Model: Time series analysis, sequential clustering

**Value:** Predictive insights into market opportunities before competitors

### 3. Risk Scoring Models

**Goal:** Develop custom risk metrics based on market-wide patterns

**Training Approach:**

- Use cluster profiles as features for risk modeling
- Combine with rejection reasons and patterns
- Train specialized models for UK mortgage market
- Model: Logistic Regression, XGBoost, Custom Ensemble

**Value:** Proprietary risk scoring that replaces expensive credit bureau APIs

## Key Advantages

1. **Cross-Lender Intelligence**: Individual lenders only see their own rejections. We see market-wide patterns.

2. **Unique Dataset**: This data cannot be replicated by competitors or purchased from third parties.

3. **Continuous Improvement**: Dataset grows with every application, improving model accuracy over time.

4. **Multiple Model Types**: Same data supports various prediction tasks (acceptance, risk, trends).

5. **Feature Engineering Ready**: Pre-computed statistics and cluster assignments accelerate model development.

## Example Training Pipeline

```python
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load training data
with open('training_data/market_clustering_20260105_103000.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['applicant_records'])

# Extract features
features_df = pd.json_normalize(df['features'])
features_df['cluster'] = df['cluster_assignment']

# For acceptance prediction (if you have acceptance labels):
# X = features_df
# y = acceptance_labels  # Would come from actual submission data
# model = GradientBoostingClassifier()
# model.fit(X, y)

# For market trend analysis:
# Track cluster_profiles across multiple time periods
# Analyze changes in cluster sizes and characteristics
```

## Integration with Other POCs

The Market Finder training data complements data from other POCs:

- **Auditor Agent**: Provides compliance decision patterns
- **Visual Forensic**: Provides document fraud patterns
- **Market Finder**: Provides cross-lender acceptance patterns

Together, these datasets enable comprehensive predictive models for the entire mortgage application lifecycle.

## Future Enhancements

1. **Lender-Specific Models**: Train separate models for each lender's acceptance criteria
2. **Temporal Features**: Add time-based features to capture market seasonality
3. **Multi-Modal Learning**: Combine structured data with document analysis
4. **Active Learning**: Use model predictions to guide manual review priorities
5. **A/B Testing Framework**: Compare proprietary models vs API-based approaches

## Data Privacy & Compliance

- All personally identifiable information (PII) should be anonymised
- Applicant IDs should be hashed or pseudonymized
- Ensure GDPR compliance for data storage and processing
- Implement access controls for training data files
- Regular audits of model fairness and bias
