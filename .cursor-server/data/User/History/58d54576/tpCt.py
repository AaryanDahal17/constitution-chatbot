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
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #a8dadc;
        border-left: 5px solid #457b9d;
    }
    .bot-message {
        background-color: #f1faee;
        border-left: 5px solid #e63946;
    }
    /* Style for all chat messages */
    .st-emotion-cache-1kg8v3r {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px !important;
        margin-bottom: 10px !important;
    }
    /* Make all message text more visible */
    .st-emotion-cache-1kg8v3r p {
        color: #333333 !important;
        font-weight: 400 !important;
    }
    /* Style for assistant avatar background */
    .st-emotion-cache-1v04i6i {
        background-color: #1e88e5 !important;
    }
    /* Style for assistant messages */
    .st-emotion-cache-1v04i6i + div {
        background-color: #e3f2fd !important;
        border-left: 5px solid #1e88e5 !important;
    }
    /* Make assistant message text dark blue and bold */
    .st-emotion-cache-1v04i6i + div p {
        color: #0d47a1 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

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
    )

# Load and process the document
@st.cache_resource
def load_documents(file_path):
    with st.spinner("Loading and processing the Nepal Constitution..."):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        return text_splitter.split_documents(documents)

# Initialize embeddings model
@st.cache_resource
def init_embeddings(model_name=EMBEDDING_MODEL):
    with st.spinner("Loading embedding model..."):
        return HuggingFaceEmbeddings(model_name=model_name)

# Create or load vector store
def get_vector_store(documents=None, embeddings=None, force_reload=False):
    vector_store_path = Path(VECTOR_STORE_PATH)
    
    # If vector store exists and we're not forcing a reload, load it
    if vector_store_path.exists() and not force_reload:
        with st.spinner("Loading existing vector store..."):
            return FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    
    # Otherwise create a new vector store
    if documents is None or embeddings is None:
        st.error("Documents and embeddings must be provided to create a new vector store")
        st.stop()
    
    with st.spinner("Creating vector store..."):
        vector_store = FAISS.from_documents(documents, embeddings)
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
def create_rag_chain(vector_store, llm, use_custom_prompt=True):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain_kwargs = {
        "llm": llm,
        "retriever": vector_store.as_retriever(search_kwargs={"k": 4}),
        "memory": memory,
        "verbose": True,
    }
    
    if use_custom_prompt:
        chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": get_custom_prompt()}
    
    return ConversationalRetrievalChain.from_llm(**chain_kwargs)

def display_sidebar():
    st.sidebar.title("Settings")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
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
    
    # Chunk size and overlap
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50,
        help="Overlap between text chunks"
    )
    
    # Force reload vector store
    force_reload = st.sidebar.checkbox(
        "Force Reload Vector Store",
        value=False,
        help="Recreate the vector store from scratch"
    )
    
    # Use custom prompt
    use_custom_prompt = st.sidebar.checkbox(
        "Use Custom Prompt",
        value=True,
        help="Use a custom prompt template for better responses"
    )
    
    # Clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation = None
        st.experimental_rerun()
    
    return {
        "model_name": model_name,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
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
    
    # Load documents and create/load vector store
    if settings["force_reload"]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["chunk_overlap"],
        )
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        documents = text_splitter.split_documents(documents)
        vector_store = get_vector_store(documents, embeddings, force_reload=True)
    else:
        try:
            vector_store = get_vector_store(embeddings=embeddings)
        except:
            documents = load_documents(PDF_PATH)
            vector_store = get_vector_store(documents, embeddings, force_reload=True)
    
    # Create conversation chain
    if "conversation" not in st.session_state or settings["force_reload"]:
        st.session_state.conversation = create_rag_chain(
            vector_store, 
            llm, 
            use_custom_prompt=settings["use_custom_prompt"]
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
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
            full_response = ""
            
            # Simulate streaming for better UX
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke({"question": prompt})
                answer = response["answer"]
                
                # Simulate streaming
                for chunk in answer.split():
                    full_response += chunk + " "
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main() 