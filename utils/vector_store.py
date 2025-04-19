import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from utils.config import VECTOR_STORE_PATH

@st.cache_resource
def get_vector_store(_documents, _embeddings, _force_reload=False):
    """Create or load vector store
    
    Args:
        _documents: Document chunks
        _embeddings: Embeddings model
        _force_reload: Whether to force reload the vector store
        
    Returns:
        FAISS vector store
    """
    vector_store_path = Path(VECTOR_STORE_PATH)
    
    # If vector store exists and we're not forcing a reload, load it
    if vector_store_path.exists() and not _force_reload:
        return FAISS.load_local(VECTOR_STORE_PATH, _embeddings, allow_dangerous_deserialization=True)
    
    # Otherwise create a new vector store
    vector_store = FAISS.from_documents(_documents, _embeddings)
    # Save the vector store for future use
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store 