import pandas as pd
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configuration ---
DATA_PATH = '../data/mock_applications.csv'
DB_PATH = "knowledge_base/faiss_index"

@tool
def get_application_data(applicant_id: int) -> str:
    """
    Fetches the data for a single mortgage application using its ID.
    Returns the data as a string.
    """
    print(f"--- Calling get_application_data for Applicant ID: {applicant_id} ---")
    try:
        df = pd.read_csv(DATA_PATH)
        applicant_data = df[df['ApplicantID'] == applicant_id]

        if applicant_data.empty:
            return f"Error: No applicant found with ID {applicant_id}."

        # Convert the row to a string format for the LLM
        return applicant_data.to_string()
    except FileNotFoundError:
        return f"Error: The data file at {DATA_PATH} was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def compliance_checker(query: str) -> str:
    """
    Answers compliance and policy questions by searching the lender's policy document.
    Use this to check rules, limits, and requirements.
    Example query: 'What is the maximum LTV for a new-build property?'
    """
    print(f"--- Calling compliance_checker with query: '{query}' ---")
    try:
        # Load the local FAISS index and the embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

        # Set up the LLM and the RetrievalQA chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        
        # Run the query and return the result
        result = qa_chain.invoke(query)
        return result['result']
    except Exception as e:
        return f"An error occurred while checking compliance: {e}"