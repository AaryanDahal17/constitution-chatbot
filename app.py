import streamlit as st
from ui.styling import set_page_config, apply_custom_css
from ui.sidebar import display_sidebar
from ui.chat import init_chat_state, display_chat_history, handle_user_input
from utils.llm import init_llm
from utils.embedding import init_embeddings
from utils.document_processor import load_documents
from utils.vector_store import get_vector_store
from utils.config import PDF_PATH

def main():
    """Main application entry point"""
    # Initialize page configuration and styling
    set_page_config()
    apply_custom_css()
    
    # App title and description
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
    
    # Initialize components
    llm = init_llm(model_name=settings["model_name"], temperature=settings["temperature"])
    embeddings = init_embeddings()
    
    # Load documents
    documents = load_documents(PDF_PATH)
    
    # Get vector store
    vector_store = get_vector_store(documents, embeddings, settings["force_reload"])
    
    # Display chat history
    display_chat_history()
    
    # Handle user input
    handle_user_input(vector_store, llm, settings)

if __name__ == "__main__":
    main() 