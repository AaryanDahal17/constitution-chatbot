import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

@st.cache_resource
def load_documents(file_path, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Load and process documents from PDF
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Add page number information in metadata
    for doc in documents:
        if 'page' not in doc.metadata:
            # Convert 0-based index to 1-based page number
            page_number = doc.metadata.get('page_number', 0) + 1
            doc.metadata['page'] = page_number
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Ensure all chunks have proper metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        # If 'page' is missing, preserve page from original doc
        if 'page' not in chunk.metadata and 'source' in chunk.metadata:
            # Extract page from source if available (e.g., "path/to/file:page")
            source = chunk.metadata['source']
            if ':' in source:
                chunk.metadata['page'] = int(source.split(':')[-1]) + 1  # Convert to 1-based
    
    return chunks 