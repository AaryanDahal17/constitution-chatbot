import os
import streamlit as st
from dotenv import load_dotenv
import time
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Nepal Constitution Chatbot",
    page_icon="🇳🇵",
    layout="wide"
)

# Constants
PDF_PATH = "nepal-constitution.pdf"
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama3-70b-8192"

# Get professional styling for chat messages
def get_professional_style():
    return '''<div style="background: linear-gradient(to right, #2c3e50, #34495e); 
            color: #ecf0f1; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #1abc9c; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);">'''

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e63946;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #457b9d;
    }
    .stApp {
        background-color: #f1faee;
    }
</style>
""", unsafe_allow_html=True)

# Create a streaming callback handler for Streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.style = get_professional_style()
        self.close_div = '</div>'

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(
            f"{self.style}{self.text}▌{self.close_div}",
            unsafe_allow_html=True
        )

# Initialize the LLM
@st.cache_resource
def init_llm(model_name=DEFAULT_GROQ_MODEL, temperature=0.2):
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

# Load and process the document
@st.cache_resource
def load_documents(file_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return text_splitter.split_documents(documents)

# Initialize embeddings model
@st.cache_resource
def init_embeddings(model_name=EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)

# Create or load vector store
@st.cache_resource
def get_vector_store(_documents, _embeddings, _force_reload=False):
    vector_store_path = Path(VECTOR_STORE_PATH)
    
    # If vector store exists and we're not forcing a reload, load it
    if vector_store_path.exists() and not _force_reload:
        return FAISS.load_local(VECTOR_STORE_PATH, _embeddings, allow_dangerous_deserialization=True)
    
    # Otherwise create a new vector store
    vector_store = FAISS.from_documents(_documents, _embeddings)
    # Save the vector store for future use
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

# Create custom prompt template
def get_custom_prompt():
    template = """
    You are an expert on the Constitution of Nepal and your task is to provide accurate information based on the document.
    
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep your answers concise, accurate, and based only on the provided context.
    
    Context:
    {context}
    
    Chat History:
    {chat_history}
    
    Question: {question}
    
    Answer:
    """
    return PromptTemplate.from_template(template)

# Create RAG chain
@st.cache_resource(show_spinner=False)
def create_rag_chain(_vector_store, _llm, _use_custom_prompt=True, _streaming_callback=None):
    # Use session state to maintain memory across streaming and non-streaming chains
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    chain_kwargs = {
        "llm": _llm,
        "retriever": _vector_store.as_retriever(search_kwargs={"k": 4}),
        "memory": st.session_state.memory,
        "verbose": False,  # Set to False to reduce overhead
    }
    
    if _use_custom_prompt:
        chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": get_custom_prompt()}
    
    if _streaming_callback:
        chain_kwargs["callbacks"] = [_streaming_callback]
    
    return ConversationalRetrievalChain.from_llm(**chain_kwargs)

def display_sidebar():
    st.sidebar.title("Settings")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        options=["deepseek-r1-distill-llama-70b","llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Higher values make the output more random, lower values make it more deterministic"
    )
    
    force_reload = False
    # # Force reload vector store
    # force_reload = st.sidebar.checkbox(
    #     "Force Reload Vector Store",
    #     value=False,
    #     help="Recreate the vector store from scratch"
    # )
    
    use_custom_prompt = True

    # # Use custom prompt
    # use_custom_prompt = st.sidebar.checkbox(
    #     "Use Custom Prompt",
    #     value=True,
    #     help="Use a custom prompt template for better responses"
    # )
    
    # # Clear chat history
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

def main():
    # App title and description
    st.markdown('<p class="main-header">🇳🇵 Nepal Constitution Chatbot</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="sub-header">
    Ask questions about the Constitution of Nepal and get accurate answers based on the official document.
    </p>
    """, unsafe_allow_html=True)
    
    # Get settings from sidebar
    settings = display_sidebar()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components
    llm = init_llm(model_name=settings["model_name"], temperature=settings["temperature"])
    embeddings = init_embeddings()
    
    # Load documents with the current settings
    documents = load_documents(
        PDF_PATH, 
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Get vector store
    vector_store = get_vector_store(documents, embeddings, settings["force_reload"])
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f'{get_professional_style()}{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the Nepal Constitution"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            
            # Create conversation chain with streaming
            conversation = create_rag_chain(
                vector_store, 
                llm, 
                _use_custom_prompt=settings["use_custom_prompt"],
                _streaming_callback=stream_handler
            )
            
            # Get response with streaming
            response = conversation({"question": prompt})
            final_response = response["answer"]
            
            # Display final response without cursor
            message_placeholder.markdown(
                f'{get_professional_style()}{final_response}</div>',
                unsafe_allow_html=True
            )
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main() 