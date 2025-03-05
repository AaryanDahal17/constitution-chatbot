# Import required libraries for PDF processing, LLM interaction, and data handling
import json
from pathlib import Path
from typing import List, Dict, Optional
import random
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from pydantic import RootModel

# Load environment variables from .env file
load_dotenv()

# Configuration constants for the script
PDF_PATH = "nepal-constitution.pdf"  # Path to the input PDF file
OUTPUT_FILE = "evaluation_sets.json"  # Output JSON file for storing evaluation sets
CHUNK_SIZE = 2500  # Size of text chunks for processing
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context
DEFAULT_GROQ_MODEL = "llama3-70b-8192"  # Default LLM model to use

# Pydantic models for type validation and data structure
class QAPair(BaseModel):
    """Model representing a single question-answer pair."""
    question: str
    answer: str

class QAResponse(RootModel):
    """Model representing a list of question-answer pairs."""
    root: List[QAPair]

def init_llm(model_name=DEFAULT_GROQ_MODEL, temperature=0.2):
    """
    Initialize the LLM (Language Learning Model) with specified parameters.
    
    Args:
        model_name (str): Name of the LLM model to use
        temperature (float): Temperature parameter for controlling randomness in model output
    
    Returns:
        ChatGroq: Initialized LLM instance
    
    Raises:
        ValueError: If GROQ_API_KEY is not found in environment variables
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        streaming=False
    )

def load_and_split_documents(file_path: str) -> List[Dict]:
    """
    Load a PDF file and split it into sections, skipping the first 3 pages.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        List[Dict]: List of dictionaries containing content and metadata for each section
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Skip first 3 pages (usually containing cover, table of contents, etc.)
    documents = documents[3:]
    
    # Convert each page to a chunk with its metadata
    chunks = []
    for doc in documents:
        chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    
    return chunks

def generate_qa_pairs(section: Dict, llm: ChatGroq) -> List[QAPair]:
    """
    Generate question-answer pairs for a given section using the LLM.
    
    Args:
        section (Dict): Dictionary containing the section content and metadata
        llm (ChatGroq): Initialized LLM instance
    
    Returns:
        List[QAPair]: List of generated question-answer pairs
    """
    prompt = f"""You are a JSON generator. Your task is to generate relevant questions and answers about the given text.
    You must ONLY return a valid JSON array containing question-answer pairs, nothing else.
    Do not include any explanations or additional text.
    
    Important rules:
    1. Only generate questions if the content is relevant and meaningful
    2. Generate EXACTLY 2 questions based on the content's relevance
    3. Each question should be specific and test understanding of the content
    4. If the content is not suitable for questions, return an empty array []

    Based on this section from the Nepal Constitution, generate exactly 2 relevant questions and answers:

    {section['content']}

    The response must be ONLY a JSON array in this exact format:
    [
        {{"question": "First question here", "answer": "First answer here"}},
        {{"question": "Second question here", "answer": "Second answer here"}}
    ]"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean up the response if needed - handle cases where the LLM might add extra text
        if not content.startswith('['):
            # Try to find the JSON array within the response
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx + 1]
            else:
                raise ValueError("No valid JSON array found in response")
        
        # Parse the response using Pydantic for validation
        qa_response = QAResponse.model_validate_json(content)
        return qa_response.root
    except Exception as e:
        print(f"Error generating QA pairs: {str(e)}")
        print(f"Raw response: {response.content}")
        return []

def load_existing_evaluation_sets() -> List[Dict]:
    """
    Load existing evaluation sets from the output JSON file if it exists.
    
    Returns:
        List[Dict]: List of existing evaluation sets or empty list if file doesn't exist
    """
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_evaluation_set(evaluation_set: Dict, existing_sets: List[Dict]):
    """
    Save a single evaluation set to the JSON file.
    
    Args:
        evaluation_set (Dict): The evaluation set to save
        existing_sets (List[Dict]): List of existing evaluation sets
    """
    existing_sets.append(evaluation_set)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_sets, f, ensure_ascii=False, indent=2)

def create_evaluation_sets():
    """
    Main function to create evaluation sets from the constitution document.
    Processes each section sequentially, generates QA pairs, and saves them to a JSON file.
    """
    # Initialize LLM with default settings
    llm = init_llm()
    
    # Load existing evaluation sets to avoid reprocessing
    evaluation_sets = load_existing_evaluation_sets()
    processed_section_ids = {set["section_id"] for set in evaluation_sets}
    
    # Load and split the PDF document into sections
    sections = load_and_split_documents(PDF_PATH)
    
    # Process each section sequentially
    for section_id, section in enumerate(sections):
        # Skip already processed sections
        if section_id in processed_section_ids:
            print(f"Skipping section {section_id} - already processed")
            continue
            
        print(f"Processing section {section_id}")
        qa_pairs = generate_qa_pairs(section, llm)
        
        # Save evaluation set if QA pairs were generated
        if qa_pairs:
            evaluation_set = {
                "section_id": section_id,
                "content": section["content"],
                "metadata": section["metadata"],
                "qa_pairs": [qa_pair.model_dump() for qa_pair in qa_pairs]
            }
            save_evaluation_set(evaluation_set, evaluation_sets)
            evaluation_sets.append(evaluation_set)
            print(f"Saved section {section_id} with {len(qa_pairs)} QA pairs")
    
    print(f"Completed processing all sections")
    print(f"Total evaluation sets: {len(evaluation_sets)}")
    print(f"Saved to {OUTPUT_FILE}")

# Entry point of the script
if __name__ == "__main__":
    create_evaluation_sets() 