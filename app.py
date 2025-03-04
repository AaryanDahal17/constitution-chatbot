import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Initialize the LLM
def init_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2,
    )

# Load and process the document
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    return text_splitter.split_documents(documents)

# Initialize embeddings model
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
def create_vector_store(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)

# Create RAG chain
def create_rag_chain(vector_store, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        verbose=True,
    )

def main():
    print("Loading Nepal Constitution RAG Chatbot...")
    
    # Initialize components
    llm = init_llm()
    documents = load_documents("nepal-constitution.pdf")
    embeddings = init_embeddings()
    vector_store = create_vector_store(documents, embeddings)
    qa_chain = create_rag_chain(vector_store, llm)
    
    print("Chatbot is ready! Ask questions about the Nepal Constitution.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() == 'exit':
            print("Thank you for using the Nepal Constitution RAG Chatbot!")
            break
        
        result = qa_chain.invoke({"question": query})
        print("\nAnswer:", result["answer"])


if __name__ == "__main__":
    main() 