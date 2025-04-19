import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
PDF_PATH = "c2_better.pdf"
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama3-70b-8192"

# Available models
AVAILABLE_MODELS = [
    "llama3-70b-8192",
    "deepseek-r1-distill-llama-70b", 
    "llama3-8b-8192", 
    "mixtral-8x7b-32768"
]

# Default settings
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200 