import streamlit as st
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from typing import List, Dict, Any, Optional, Callable
import os
import json
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rag.prompt_templates import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, STANDALONE_QUESTION_PROMPT, get_custom_prompt

from utils.document_processor import load_documents
from utils.text_matcher import highlight_matches

class RAGChain:
    """Updated RAG chain that works with enhanced document structure from preprocessed JSON files."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        doc_sources: Optional[List[str]] = None,
        preprocessed_data_dir: Optional[str] = None,
        retriever: Optional[BaseRetriever] = None,
    ):
        """
        Initialize a RAG chain.
        
        Args:
            llm: The language model to use
            embedding_model_name: Name of the embedding model to use
            doc_sources: List of document sources (deprecated)
            preprocessed_data_dir: Directory containing preprocessed document JSON files
            retriever: Optional custom retriever to use instead of initializing with documents
        """
        self.llm = llm
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.retriever = retriever
        
        # Initialize the retriever if documents are provided
        if preprocessed_data_dir and not retriever:
            self.initialize_vector_store(preprocessed_data_dir)
            
        # Create the chain components
        self.question_generator = LLMChain(
            llm=self.llm,
            prompt=CONDENSE_QUESTION_PROMPT
        )
        
        # Create the standard answer chain
        self.answer_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=ANSWER_PROMPT
        )
    
    def initialize_vector_store(self, preprocessed_data_dir: str):
        """
        Initialize the vector store from preprocessed document JSON files.
        
        Args:
            preprocessed_data_dir: Directory containing preprocessed document JSON files
        """
        documents = []
        
        # Load all JSON files from the directory
        for filename in os.listdir(preprocessed_data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(preprocessed_data_dir, filename)
                with open(file_path, 'r') as f:
                    doc_data = json.load(f)
                    
                    # Create a Document object for each JSON file
                    # The document should contain the text and metadata from the JSON
                    doc = Document(
                        page_content=doc_data.get('text', ''),
                        metadata=doc_data.get('metadata', {})
                    )
                    documents.append(doc)
        
        # Create the vector store and retriever
        if documents:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
    
    def _condense_question(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Condenses the chat history and new question into a standalone question.
        
        Args:
            question: The new question from the user
            chat_history: The chat history as a list of dictionaries with "human" and "ai" keys
            
        Returns:
            A standalone question that incorporates context from the chat history
        """
        if not chat_history:
            return question
            
        # Format chat history
        formatted_chat_history = "\n".join(
            [f"Human: {turn['human']}\nAI: {turn['ai']}" for turn in chat_history]
        )
        
        # Generate standalone question
        result = self.question_generator.invoke({
            "question": question,
            "chat_history": formatted_chat_history
        })
        
        return result["text"].strip()
    
    def _get_relevant_documents(self, question: str) -> List[Document]:
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: The question to retrieve documents for
            
        Returns:
            A list of relevant Document objects
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call initialize_vector_store first.")
        
        return self.retriever.get_relevant_documents(question)
    
    def create_chain(self):
        """
        Create the complete RAG chain.
        
        Returns:
            A runnable chain that can be invoked with a question and optional chat history
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Cannot create chain.")
        
        # Define the retrieval chain using the standalone question
        standalone_retrieval_chain = (
            {"question": RunnablePassthrough()}
            | {"question": RunnablePassthrough(), "context": self.retriever}
            | self.answer_chain
            | StrOutputParser()
        )
        
        # Define the conversational chain with history
        conversational_retrieval_chain = (
            {
                "chat_history": lambda x: x.get("chat_history", []),
                "question": lambda x: x["question"]
            }
            | {
                "question": self._condense_question, 
                "context": lambda x: self.retriever.get_relevant_documents(x["question"]),
                "original_question": lambda x: x["question"]
            }
            | self.answer_chain
            | StrOutputParser()
        )
        
        # Return a combined chain that handles both with and without chat history
        def combined_chain(input_dict):
            if "chat_history" in input_dict and input_dict["chat_history"]:
                return conversational_retrieval_chain.invoke(input_dict)
            else:
                return standalone_retrieval_chain.invoke(input_dict["question"])
                
        return combined_chain
    
    def generate_answer(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate an answer to a question.
        
        Args:
            question: The question to answer
            chat_history: Optional chat history
            
        Returns:
            The generated answer
        """
        chain = self.create_chain()
        
        if chat_history:
            return chain({"question": question, "chat_history": chat_history})
        else:
            return chain({"question": question})

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
    # Configure retriever with more relevant chunks and search type
    # Retrieve more documents to increase chances of finding relevant references
    retriever = _vector_store.as_retriever(
        search_kwargs={
            "k": 8,  # Increased from 5 to get more potential references
            "search_type": "similarity",  # Changed from threshold to ensure we always get results
            "fetch_k": 15,  # Fetch more candidates before filtering
        }
    )
    
    # Use session state to maintain memory across streaming and non-streaming chains
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify which output key to use for memory
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
    
    # Enhanced document ranking - prioritize section documents
    def get_score_custom(doc):
        # Prioritize section documents as they are more precise
        base_score = 1.0
        if doc.metadata.get('is_section', False):
            base_score += 0.5  # Boost section documents
            
            # Further boost if the section has a title matching query keywords
            section_type = doc.metadata.get('section_type', '')
            section_number = doc.metadata.get('section_number', '')
            if section_type and section_number:
                base_score += 0.2
                
        # Boost recent pages for temporal priority
        page_num = doc.metadata.get('page', 0)
        if isinstance(page_num, int) and page_num > 0:
            # Small boost for being from the start of the document (often more relevant)
            if page_num < 10:
                base_score += 0.1
                
        # Give a small boost to shorter documents as they tend to be more focused
        doc_len = len(doc.page_content)
        if doc_len < 500:
            base_score += 0.1
            
        return base_score
    
    # Create the chain with document re-ranking
    chain = ConversationalRetrievalChain.from_llm(
        **chain_kwargs
    )
    
    return chain 