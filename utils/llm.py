import os
import streamlit as st
from langchain_groq import ChatGroq
from utils.config import DEFAULT_GROQ_MODEL

@st.cache_resource
def init_llm(model_name=DEFAULT_GROQ_MODEL, temperature=0.2):
    """Initialize the Groq LLM with specified parameters
    
    Args:
        model_name: Name of the Groq model to use
        temperature: Temperature for generation (0-1)
        
    Returns:
        ChatGroq instance
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables")
        st.stop()
    
    return ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        streaming=True  # Enable native streaming
    ) 