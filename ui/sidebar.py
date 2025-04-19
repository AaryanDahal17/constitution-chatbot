import streamlit as st
from utils.config import AVAILABLE_MODELS, DEFAULT_TEMPERATURE

def display_sidebar():
    """Display and handle sidebar elements, return the selected settings"""
    st.sidebar.title("Settings")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        options=AVAILABLE_MODELS,
        index=0
    )
    
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Higher values make the output more random, lower values make it more deterministic"
    )
    
    # Always use these defaults for simplicity
    force_reload = False
    use_custom_prompt = True
    
    # Optional: Add these back if needed
    # force_reload = st.sidebar.checkbox(
    #     "Force Reload Vector Store",
    #     value=False,
    #     help="Recreate the vector store from scratch"
    # )
    
    # use_custom_prompt = st.sidebar.checkbox(
    #     "Use Custom Prompt",
    #     value=True,
    #     help="Use a custom prompt template for better responses"
    # )
    
    # if st.sidebar.button("Clear Chat History"):
    #     st.session_state.messages = []
    #     if "conversation" in st.session_state:
    #         # Reset the conversation memory
    #         st.session_state.conversation = None
    #     st.experimental_rerun()
    
    return {
        "model_name": model_name,
        "temperature": temperature,
        "force_reload": force_reload,
        "use_custom_prompt": use_custom_prompt
    } 