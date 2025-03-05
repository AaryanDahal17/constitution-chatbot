import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
PDF_PATH = "nepal-constitution.pdf"
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama3-70b-8192"

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

def init_llm(model_name=DEFAULT_GROQ_MODEL, temperature=0.2):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        streaming=False  # Disable streaming for API usage
    )

def load_documents(file_path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return text_splitter.split_documents(documents)

def init_embeddings(model_name=EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)

def get_vector_store(_documents, _embeddings, _force_reload=False):
    vector_store_path = Path(VECTOR_STORE_PATH)
    
    if vector_store_path.exists() and not _force_reload:
        return FAISS.load_local(VECTOR_STORE_PATH, _embeddings, allow_dangerous_deserialization=True)
    
    vector_store = FAISS.from_documents(_documents, _embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def create_rag_chain(_vector_store, _llm, _use_custom_prompt=True):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain_kwargs = {
        "llm": _llm,
        "retriever": _vector_store.as_retriever(search_kwargs={"k": 4}),
        "memory": memory,
        "verbose": False,
    }
    
    if _use_custom_prompt:
        chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": get_custom_prompt()}
    
    return ConversationalRetrievalChain.from_llm(**chain_kwargs)

def makeConstitutionQuery(params):
    """
    Make a query to the Nepal Constitution chatbot.
    
    Args:
        params (dict): Dictionary containing:
            - question (str): The question to ask about the constitution
            - model_name (str, optional): The Groq model to use. Defaults to "llama3-70b-8192"
            - temperature (float, optional): Temperature for response generation. Defaults to 0.2
            - force_reload (bool, optional): Whether to force reload the vector store. Defaults to False
    
    Returns:
        str: The response from the chatbot
    """
    # Extract parameters with defaults
    question = params.get('question')
    if not question:
        raise ValueError("Question is required")
        
    model_name = params.get('model_name', DEFAULT_GROQ_MODEL)
    temperature = params.get('temperature', 0.2)
    force_reload = params.get('force_reload', False)
    
    # Initialize components
    llm = init_llm(model_name=model_name, temperature=temperature)
    embeddings = init_embeddings()
    
    # Load documents
    documents = load_documents(PDF_PATH)
    
    # Get vector store
    vector_store = get_vector_store(documents, embeddings, force_reload)
    
    # Create conversation chain
    conversation = create_rag_chain(vector_store, llm)
    
    # Get response
    response = conversation({"question": question})
    return response["answer"]

def makeRawLLMQuery(params):
    """
    Make a query to the LLM without RAG, using only the model's base knowledge.
    
    Args:
        params (dict): Dictionary containing:
            - question (str): The question to ask about the constitution
            - model_name (str, optional): The Groq model to use. Defaults to "llama3-70b-8192"
            - temperature (float, optional): Temperature for response generation. Defaults to 0.2
    
    Returns:
        str: The response from the LLM
    """
    # Extract parameters with defaults
    question = params.get('question')
    if not question:
        raise ValueError("Question is required")
        
    model_name = params.get('model_name', DEFAULT_GROQ_MODEL)
    temperature = params.get('temperature', 0.2)
    
    # Initialize LLM
    llm = init_llm(model_name=model_name, temperature=temperature)
    
    # Create prompt template
    template = """
    You are an expert on the Constitution of Nepal. Please answer the following question about Nepal's constitution.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep your answers concise and accurate.
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    # Format and send prompt
    formatted_prompt = prompt.format(question=question)
    response = llm.invoke(formatted_prompt)
    
    return response.content

# Example usage:
if __name__ == "__main__":
    # Example queries for both methods
    params = {
        "question": "What are the fundamental rights in the Nepal Constitution?",
        "model_name": "llama3-70b-8192",
        "temperature": 0.2
    }
    
    try:
        # Test RAG-based query
        print("Testing RAG-based query:")
        rag_response = makeConstitutionQuery(params)
        print("RAG Response:", rag_response)
        
        print("\nTesting raw LLM query:")
        raw_response = makeRawLLMQuery(params)
        print("Raw LLM Response:", raw_response)
    except Exception as e:
        print(f"Error: {str(e)}") 