import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.config import EMBEDDING_MODEL

@st.cache_resource
def init_embeddings(model_name=EMBEDDING_MODEL):
    """Initialize the embeddings model
    
    Args:
        model_name: Name of the HuggingFace embeddings model
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(model_name=model_name) 