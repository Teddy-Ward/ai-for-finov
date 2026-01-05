import os
from pathlib import Path

# Get the absolute path to the docs folder
DOCS_FOLDER = Path(__file__).parent / "docs"
GOLD_STANDARD_PDF = DOCS_FOLDER / "NLC-Payslip-Example-V1.pdf"

# Training data storage
TRAINING_DATA_FOLDER = Path(__file__).parent / "training_data"
TRAINING_DATA_FOLDER.mkdir(exist_ok=True)

TEMPLATES = {
    "NLC_Payslip_Gold_Standard": {
        "name": "NLC Payslip Example V1",
        "path": str(GOLD_STANDARD_PDF),
        "description": "Official NLC payslip template used as gold standard for comparison",
        "expected_font": "Calibri",
        "document_type": "payslip"
    },
    "Lloyds_Bank_Statement_v2": {
        "expected_font": "Arial-Bold",
        "primary_logo_coord": {"x": 45, "y": 850},
        "data_fields": [
            {"name": "Sort Code", "x": 100, "y": 700, "alignment": "left"},
            {"name": "Balance", "x": 500, "y": 150, "alignment": "right"}
        ]
    },
    "Standard_Sage_Payslip": {
        "expected_font": "Calibri",
        "primary_logo_coord": {"x": 20, "y": 900},
        "data_fields": [
            {"name": "National Insurance", "x": 50, "y": 600},
            {"name": "Net Pay", "x": 450, "y": 100, "font_weight": "bold"}
        ]
    }
}