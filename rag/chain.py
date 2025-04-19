import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rag.prompt_templates import get_custom_prompt

@st.cache_resource(show_spinner=False)
def create_rag_chain(_vector_store, _llm, _use_custom_prompt=True, _streaming_callback=None):
    """Create a RAG chain for question answering
    
    Args:
        _vector_store: FAISS vector store
        _llm: LLM instance
        _use_custom_prompt: Whether to use custom prompt
        _streaming_callback: Callback for streaming responses
        
    Returns:
        ConversationalRetrievalChain instance
    """
    # Use session state to maintain memory across streaming and non-streaming chains
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify which output key to use for memory
        )
    
    # Configure retriever with more relevant chunks and search type
    # Retrieve more documents to increase chances of finding relevant references
    retriever = _vector_store.as_retriever(
        search_kwargs={
            "k": 8,  # Increased from 5 to get more potential references
            "search_type": "similarity",  # Changed from threshold to ensure we always get results
            "fetch_k": 15,  # Fetch more candidates before filtering
        }
    )
    
    chain_kwargs = {
        "llm": _llm,
        "retriever": retriever,
        "memory": st.session_state.memory,
        "verbose": True,  # Set to True to help debug
        "return_source_documents": True,  # CRITICAL: always return source documents
        "return_generated_question": True,  # Return generated question for better context
    }
    
    if _use_custom_prompt:
        chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": get_custom_prompt()}
    
    if _streaming_callback:
        chain_kwargs["callbacks"] = [_streaming_callback]
    
    return ConversationalRetrievalChain.from_llm(**chain_kwargs) 