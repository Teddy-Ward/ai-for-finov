# Troubleshooting

## Common Issues

### "ModuleNotFoundError" when running applications

- **Most common cause**: Virtual environment is not activated
- Ensure you've activated your virtual environment (look for `(venv)` in your terminal prompt)
- Run `pip install -r requirements.txt` to install all dependencies
- If still having issues, try: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)

### "mock_applications.csv file not found" error

- The data file should already be included in the repository
- If missing, run `python generate_data.py` from the root directory to regenerate it

### Auditor Agent shows "No PDF documents found"

- The knowledge base should already be included in the repository
- If missing, run `python ingest.py` from the `auditor_agent_poc/` directory to rebuild it
- Ensure PDF documents are placed in `auditor_agent_poc/docs/` folder

### OpenAI API errors

- Verify your API key is correctly set in the `.env` file
- Ensure you have sufficient API credits
- Check that the API key has access to GPT-4-turbo and GPT-4o (for Visual Forensic Scanner)
- Visual Forensic Scanner requires GPT-4o with vision capabilities

### Streamlit app won't start

- Make sure you're in the correct directory (`auditor_agent_poc/`, `market_finder_poc/`, `visual_forensic_poc/`, etc.)
- Try `streamlit run app.py --server.port 8501` if port conflicts occur

### Training data not saving

- Check that `training_data/` folder exists in the respective POC directory
- Ensure you have write permissions to the directory
- Verify OpenAI API is working for change analysis features

### Visual Forensic Scanner errors

- Ensure you upload a PDF file (not image formats)
- Gold standard document must exist in `visual_forensic_poc/docs/` folder
- Check that all required libraries are installed: `opencv-python`, `scikit-image`, `PyMuPDF`

## Performance Notes

- **Ready to run**: All applications are ready to run immediately since data files are pre-included
- **Memory usage**: The Market Finder processes 1,000 records in real-time
- **API costs**:
  - Auditor Agent: ~£0.01-0.03 per audit (GPT-4-turbo)
  - Visual Forensic Scanner: ~£0.05-0.15 per document (GPT-4o vision + analysis)
- **Regeneration**: If you need to rebuild data files, the ingestion process may take 2-5 minutes to download the embedding model
- **Training Data**: Accumulates automatically with each use - no manual export needed
- **Storage**: JSON training files are typically 10-50KB each
