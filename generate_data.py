import pandas as pd
import numpy as np
import os

# --- Configuration ---
NUM_SAMPLES_PER_PERSONA = 250
OUTPUT_FOLDER = 'data'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'mock_applications.csv')

def generate_persona_data(size, persona_params):
    """Generates a DataFrame for a single persona based on defined parameters."""
    data = {
        # Generate income with a normal distribution and clip at a minimum of 20k
        'Income': np.clip(np.random.normal(
            loc=persona_params['income_mean'],
            scale=persona_params['income_std'],
            size=size
        ), a_min=20000, a_max=None).round(2),

        # Generate CreditScore and ensure it's an integer
        'CreditScore': np.clip(np.random.normal(
            loc=persona_params['credit_score_mean'],
            scale=persona_params['credit_score_std'],
            size=size
        ), a_min=300, a_max=900).astype(int),

        # Generate LTV and clip between 50 and 100
        'LTV': np.clip(np.random.normal(
            loc=persona_params['ltv_mean'],
            scale=persona_params['ltv_std'],
            size=size
        ), a_min=50, a_max=100).round(2),

        # Generate DTI and clip at a minimum of 10
        'DebtToIncomeRatio': np.clip(np.random.normal(
            loc=persona_params['dti_mean'],
            scale=persona_params['dti_std'],
            size=size
        ), a_min=10, a_max=None).round(2),

        'EmploymentStatus': persona_params['status'],
        'RejectionReason': persona_params['reason']
    }
    return pd.DataFrame(data)

def main():
    """Main function to define personas, generate data, and save to CSV."""
    print("Starting mock data generation...")

    # --- Persona Definitions ---
    personas = {
        "Stable Gig-Worker": {
            'income_mean': 75000, 'income_std': 15000,
            'credit_score_mean': 750, 'credit_score_std': 50,
            'ltv_mean': 75, 'ltv_std': 5,
            'dti_mean': 25, 'dti_std': 5,
            'status': 'Self-Employed < 2yrs',
            'reason': 'Insufficient trading history'
        },
        "Nearly-There First-Time Buyer": {
            'income_mean': 60000, 'income_std': 8000,
            'credit_score_mean': 780, 'credit_score_std': 40,
            'ltv_mean': 92, 'ltv_std': 1.5,
            'dti_mean': 30, 'dti_std': 5,
            'status': 'PAYE',
            'reason': 'LTV exceeds product maximum'
        },
        "Minor Credit Blip": {
            'income_mean': 90000, 'income_std': 20000,
            'credit_score_mean': 600, 'credit_score_std': 25,
            'ltv_mean': 65, 'ltv_std': 8,
            'dti_mean': 20, 'dti_std': 5,
            'status': 'PAYE',
            'reason': 'Adverse credit history'
        },
        "Slightly Overextended": {
            'income_mean': 65000, 'income_std': 10000,
            'credit_score_mean': 720, 'credit_score_std': 50,
            'ltv_mean': 85, 'ltv_std': 3,
            'dti_mean': 50, 'dti_std': 3,
            'status': 'PAYE',
            'reason': 'Affordability / High DTI'
        }
    }

    # Generate data for each persona
    all_dataframes = []
    for name, params in personas.items():
        print(f"Generating {NUM_SAMPLES_PER_PERSONA} samples for persona: {name}")
        df = generate_persona_data(NUM_SAMPLES_PER_PERSONA, params)
        all_dataframes.append(df)

    # Combine all data into a single DataFrame
    synthetic_data = pd.concat(all_dataframes)

    # Shuffle the data to mix the personas
    synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)

    # Add a realistic ApplicantID
    synthetic_data.insert(0, 'ApplicantID', range(10001, 10001 + len(synthetic_data)))

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Save the final dataset to a CSV file
    synthetic_data.to_csv(OUTPUT_FILE, index=False)

    print("-" * 30)
    print(f"Successfully created mock dataset with {len(synthetic_data)} rows.")
    print(f"File saved to: {OUTPUT_FILE}")
    print("Here's a preview of the first 5 rows:")
    print(synthetic_data.head())
    print("-" * 30)


if __name__ == "__main__":
    main()