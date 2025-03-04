# Nepal Constitution RAG Chatbot

This is a Retrieval Augmented Generation (RAG) chatbot that can answer questions about the Constitution of Nepal. The application uses LangChain, Groq LLM, and HuggingFace embeddings to provide accurate information based on the official document.

## Features

- Interactive chat interface to ask questions about Nepal's Constitution
- Uses RAG to retrieve relevant information from the document
- Maintains conversation history for contextual responses
- Available as both a command-line application and a web interface

## Requirements

- Python 3.8+
- Groq API key (stored in `.env` file)

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install langchain langchain-groq langchain-community pypdf faiss-cpu sentence-transformers python-dotenv streamlit
```

3. Make sure your `.env` file contains the Groq API key:

```
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Command-line Interface

Run the command-line version of the chatbot:

```bash
python app.py
```

### Web Interface

Run the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

This will start a local web server and open the application in your default web browser.

## How It Works

1. The application loads the Nepal Constitution PDF document
2. The document is split into smaller chunks for efficient processing
3. Each chunk is embedded using HuggingFace's sentence-transformers model
4. The embeddings are stored in a FAISS vector database for fast retrieval
5. When you ask a question, the system:
   - Converts your question into an embedding
   - Finds the most relevant document chunks
   - Sends these chunks along with your question to the Groq LLM
   - Returns the generated answer

## Example Questions

- What is the structure of the government in Nepal?
- What are the fundamental rights guaranteed by the Nepal Constitution?
- How is the President elected in Nepal?
- What is the role of the Supreme Court in Nepal?
- How can the Constitution be amended? 