import json
import streamlit as st
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from utils.pdf_preprocessor import get_processed_document_path

@st.cache_resource
def load_documents(file_path, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Load and process documents from preprocessed JSON
    
    Args:
        file_path: Path to the PDF file (will be preprocessed to JSON)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    # Preprocess the PDF to JSON (if not already processed)
    json_path = get_processed_document_path(file_path)
    
    # Load the processed JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        processed_doc = json.load(f)
    
    # Create Langchain documents from the JSON
    documents = []
    
    # Process each page into a document
    for page in processed_doc["pages"]:
        page_num = page["page_number"]
        text = page["content"]
        
        # Create metadata with enhanced information
        metadata = {
            "page": page_num,
            "source": file_path,
            "sections": [f"{s['type']} {s['number']}" for s in page["sections"]],
            "article_refs": page["article_refs"]
        }
        
        # Create document with meaningful ID
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        documents.append(doc)
    
    # Also create documents for each identified section for more precise retrieval
    for section_type, sections in processed_doc["structure"].items():
        for section in sections:
            section_doc = Document(
                page_content=section["context"],
                metadata={
                    "page": section["page"],
                    "source": file_path,
                    "section_type": section_type,
                    "section_number": section["number"],
                    "is_section": True
                }
            )
            documents.append(section_doc)
    
    # Split into chunks with metadata preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Ensure all chunks have proper metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        
        # Add descriptive chunk title for easier reference
        if 'is_section' in chunk.metadata and chunk.metadata['is_section']:
            section_type = chunk.metadata.get('section_type', 'Section')
            section_number = chunk.metadata.get('section_number', '')
            chunk.metadata['chunk_title'] = f"{section_type} {section_number}"
        else:
            chunk.metadata['chunk_title'] = f"Page {chunk.metadata.get('page', 'Unknown')}"
    
    return chunks 