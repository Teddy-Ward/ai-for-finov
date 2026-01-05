# Setup Instructions

Follow these steps to get the project and its dependencies set up locally.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.9 or higher
- Git for cloning the repository

## 1. Clone the Repository

First, clone this repository to your local machine.

```bash
git clone https://github.com/Teddy-Ward/ai-for-finov.git
cd finova_ai_poc
```

## 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

## 3. Install Dependencies

With your virtual environment active, install all the required Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```

## 4. Set Up Environment Variables

The Auditor Agent requires an API key for a Large Language Model (e.g., OpenAI).

1. Create a new file in the root directory named `.env`.
2. Add your API key to this file:

```
OPENAI_API_KEY="sk-..."
```

**Note**: You can get an API key from [OpenAI](https://platform.openai.com/api-keys). The Auditor Agent uses GPT-4-turbo for optimal performance.

## 5. Verify Data Files

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

## Next Steps

Once setup is complete, see [RUNNING_POCS.md](RUNNING_POCS.md) to launch the applications.
