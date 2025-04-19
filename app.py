import streamlit as st
import os
from pathlib import Path
from ui.styling import set_page_config, apply_custom_css
from ui.sidebar import display_sidebar
from ui.chat import init_chat_state, display_chat_history, handle_user_input
from utils.llm import init_llm
from utils.embedding import init_embeddings
from utils.document_processor import load_documents
from utils.vector_store import get_vector_store
from utils.config import PDF_PATH
from utils.pdf_preprocessor import get_processed_document_path
from rag.chain import create_rag_chain

def main():
    """Main application entry point"""
    # Initialize page configuration and styling
    set_page_config()
    apply_custom_css()
    
    # Show app title and description
    st.markdown('<p class="main-header">📚 Constitution Chatbot</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="sub-header">
    Ask questions about the constitution document and get accurate answers with source references.
    </p>
    """, unsafe_allow_html=True)
    
    # Get settings from sidebar
    settings = display_sidebar()
    
    # Initialize chat state
    init_chat_state()
    
    # Check if the PDF needs preprocessing
    processed_docs_dir = "processed_docs"
    os.makedirs(processed_docs_dir, exist_ok=True)
    json_path = Path(processed_docs_dir) / f"{Path(PDF_PATH).stem}_processed.json"
    
    # Preprocess if not already done or if forced
    if settings.get("force_reload", False) or not json_path.exists():
        with st.spinner("Preprocessing document... This may take a moment."):
            # This will create the processed JSON file
            get_processed_document_path(PDF_PATH)
    
    # Initialize LLM with the selected model and temperature
    llm = init_llm(model_name=settings["model_name"], temperature=settings["temperature"])
    
    # Initialize embeddings
    embeddings = init_embeddings()
    
    # Load documents from preprocessed JSON
    with st.spinner("Loading documents..."):
        documents = load_documents(PDF_PATH)
    
    # Create or load vector store
    with st.spinner("Preparing vector database..."):
        vector_store = get_vector_store(documents, embeddings, settings["force_reload"])
    
    # Display chat history
    display_chat_history()
    
    # Handle user input with improved RAG chain
    handle_user_input(vector_store, llm, settings)

if __name__ == "__main__":
    main() 