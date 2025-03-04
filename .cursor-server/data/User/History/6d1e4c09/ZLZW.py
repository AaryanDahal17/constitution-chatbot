import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv()

# Constants
PDF_PATH = "nepal-constitution.pdf"
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("Testing vector store creation and retrieval...")
    
    # Initialize embeddings model
    print("Initializing embedding model...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Embedding model initialized in {time.time() - start_time:.2f} seconds")
    
    # Check if vector store exists
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store found at {VECTOR_STORE_PATH}")
        print("Loading existing vector store...")
        start_time = time.time()
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        print(f"Vector store loaded in {time.time() - start_time:.2f} seconds")
    else:
        print("Vector store not found, creating new one...")
        
        # Load document
        print("Loading PDF document...")
        start_time = time.time()
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"Document loaded in {time.time() - start_time:.2f} seconds")
        
        # Split document
        print("Splitting document into chunks...")
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(documents)
        print(f"Document split into {len(documents)} chunks in {time.time() - start_time:.2f} seconds")
        
        # Create vector store
        print("Creating vector store...")
        start_time = time.time()
        vector_store = FAISS.from_documents(documents, embeddings)
        print(f"Vector store created in {time.time() - start_time:.2f} seconds")
        
        # Save vector store
        print("Saving vector store...")
        start_time = time.time()
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store saved in {time.time() - start_time:.2f} seconds")
    
    # Test retrieval
    print("\nTesting retrieval with sample questions...")
    
    test_questions = [
        "What are the fundamental rights in Nepal?",
        "How is the President elected?",
        "What is the structure of the judiciary?",
        "How can the constitution be amended?",
        "What are the duties of the Prime Minister?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        start_time = time.time()
        docs = vector_store.similarity_search(question, k=2)
        print(f"Retrieved {len(docs)} documents in {time.time() - start_time:.2f} seconds")
        
        print("Top document content:")
        print("-" * 50)
        print(docs[0].page_content[:300] + "...")
        print("-" * 50)

if __name__ == "__main__":
    main() 