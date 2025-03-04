import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import time

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test the connection to Groq LLM."""
    
    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        return False
    
    print("Testing connection to Groq LLM...")
    
    try:
        # Initialize the LLM
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.2,
        )
        
        # Test with a simple prompt
        print("Sending test prompt to Groq...")
        start_time = time.time()
        
        response = llm.invoke("Hello, can you tell me about the Constitution of Nepal in one sentence?")
        
        elapsed_time = time.time() - start_time
        print(f"Response received in {elapsed_time:.2f} seconds")
        print("\nResponse from Groq:")
        print("-" * 50)
        print(response.content)
        print("-" * 50)
        
        # Test available models
        print("\nAvailable Groq models:")
        from langchain_groq.chat_models import AVAILABLE_MODELS
        for model in AVAILABLE_MODELS:
            print(f"- {model}")
        
        return True
    
    except Exception as e:
        print(f"Error connecting to Groq: {e}")
        return False

def main():
    success = test_groq_connection()
    
    if success:
        print("\nGroq LLM connection test successful!")
    else:
        print("\nGroq LLM connection test failed!")

if __name__ == "__main__":
    main() 