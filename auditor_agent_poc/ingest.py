import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DOCS_PATH = "docs/"
DB_PATH = "knowledge_base/faiss_index"

def create_vector_db():
    """
    Reads all PDF documents from the docs folder, splits them into chunks,
    creates embeddings, and saves them to a FAISS vector store.
    """
    # Check if the docs path exists
    if not os.path.exists(DOCS_PATH):
        print(f"Error: The directory '{DOCS_PATH}' does not exist.")
        return

    print("Starting the ingestion process...")

    # Load all PDF files from the specified directory
    documents = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(DOCS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    if not documents:
        print("No PDF documents found to ingest.")
        return

    print(f"Loaded {len(documents)} pages from PDF files.")

    # Split the documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # Use a pre-trained embedding model from Hugging Face
    # This model runs locally and is free to use.
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
    )
    print("Embedding model loaded.")

    # Create the FAISS vector store from the text chunks and embeddings
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the vector store locally
    db.save_local(DB_PATH)
    print("-" * 30)
    print(f"Knowledge base created successfully and saved to '{DB_PATH}'")
    print("-" * 30)

if __name__ == "__main__":
    create_vector_db()