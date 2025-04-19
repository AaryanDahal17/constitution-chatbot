import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, CONSTITUTION_DATASET_PATH
from utils.json_loader import ConstitutionJsonLoader

@st.cache_resource
def load_documents(dataset_path=CONSTITUTION_DATASET_PATH, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Load and process documents from the Constitution JSON dataset
    
    Args:
        dataset_path: Path to the constitution dataset folder
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    # Use our custom loader for the JSON dataset
    loader = ConstitutionJsonLoader(dataset_path)
    documents = loader.load()
    
    # Log the number of documents loaded
    st.sidebar.success(f"Loaded {len(documents)} constitution articles")
    
    # Split documents into chunks for better searching
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Ensure all chunks have proper metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        
        # Add article and part info to help with references
        if 'article_number' not in chunk.metadata and 'source' in chunk.metadata:
            # Try to extract article info from source document if not already present
            for doc in documents:
                if doc.metadata.get('source') == chunk.metadata.get('source'):
                    chunk.metadata['article_number'] = doc.metadata.get('article_number')
                    chunk.metadata['article'] = doc.metadata.get('article')
                    chunk.metadata['part'] = doc.metadata.get('part')
                    chunk.metadata['part_title'] = doc.metadata.get('part_title')
                    break
    
    return chunks 