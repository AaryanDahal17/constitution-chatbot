import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Nepal Constitution Chatbot",
    page_icon="🇳🇵",
    layout="wide"
)

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

# Initialize the LLM
@st.cache_resource
def init_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables")
        st.stop()
    
    return ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2,
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
def init_embeddings():
    with st.spinner("Loading embedding model..."):
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
@st.cache_resource
def create_vector_store(documents, embeddings):
    with st.spinner("Creating vector store..."):
        return FAISS.from_documents(documents, embeddings)

# Create RAG chain
@st.cache_resource
def create_rag_chain(vector_store, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
    )

def main():
    # App title and description
    st.title("🇳🇵 Nepal Constitution Chatbot")
    st.markdown("""
    This chatbot can answer your questions about the Constitution of Nepal. 
    It uses RAG (Retrieval Augmented Generation) to provide accurate information based on the official document.
    """)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize components
    llm = init_llm()
    documents = load_documents("nepal-constitution.pdf")
    embeddings = init_embeddings()
    vector_store = create_vector_store(documents, embeddings)
    
    # Create conversation chain
    if "conversation" not in st.session_state:
        st.session_state.conversation = create_rag_chain(vector_store, llm)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div style="background-color: #2b6cb0; color: #ffffff; padding: 10px; border-radius: 8px; border-left: 5px solid #1a365d;">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the Nepal Constitution"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke({"question": prompt})
                st.markdown(f'<div style="background-color: #2b6cb0; color: #ffffff; padding: 10px; border-radius: 8px; border-left: 5px solid #1a365d;">{response["answer"]}</div>', unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main() 