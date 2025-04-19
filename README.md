# Nepal Constitution Chatbot

A Streamlit-based chatbot that answers questions about the Constitution of Nepal using Retrieval-Augmented Generation (RAG).

## Features

- Chat interface to ask questions about Nepal's Constitution
- Uses FAISS for efficient vector storage and retrieval
- Powered by Groq AI models for fast responses
- Streaming responses for better user experience
- Customizable model selection and parameters

## Project Structure

```
.
├── app.py                  # Main application entry point
├── nepal-constitution.pdf  # Source document
├── requirements.txt        # Dependencies
├── utils/                  # Utility functions
│   ├── config.py           # Configuration and constants
│   ├── document_processor.py # Document loading and processing
│   ├── embedding.py        # Embeddings initialization
│   ├── llm.py              # LLM initialization
│   └── vector_store.py     # Vector store management
├── ui/                     # User interface components
│   ├── chat.py             # Chat UI components
│   ├── sidebar.py          # Sidebar components
│   └── styling.py          # Styling functions
└── rag/                    # RAG components
    ├── callbacks.py        # Callback handlers
    ├── chain.py            # RAG chain creation
    └── prompt_templates.py # Custom prompt templates
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Ask questions about Nepal's Constitution in the chat interface
2. Adjust model and parameters in the sidebar if needed
3. Get accurate, contextual answers based on the Constitution text

## License

MIT 