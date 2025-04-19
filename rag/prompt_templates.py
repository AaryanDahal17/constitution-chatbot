from langchain.prompts import PromptTemplate

# Template to condense a user question with chat history into a standalone question
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation history and a new user question, create a standalone question that captures all relevant context.
    
    Chat History:
    {chat_history}
    
    New Question: {question}
    
    Standalone Question:"""
)

# Template to generate a comprehensive answer based on retrieved document sections
ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a precise document retrieval assistant for a constitution document. Your ONLY source of information is the text provided in the context below.
    
    CRITICAL RULES:
    1. ONLY use the exact text provided in the CONTEXT. DO NOT make up or infer information not explicitly stated.
    2. ALWAYS quote the document directly using its exact wording in your answers.
    3. ALL facts in your answer MUST be directly supported by specific sections in the provided context.
    4. If the answer cannot be found in the context, state clearly: "This information is not found in the provided sections of the document."
    5. Include specific article numbers, sections, and page references when available.
    6. Format your answers to highlight exact language by using direct quotes.
    7. DO NOT refer to any external knowledge or your training data.
    
    CONTEXT:
    {context}
    
    QUESTION: {question}
    
    ANSWER (using ONLY information from the CONTEXT above):
    """
)

# Template for creating standalone questions (used when not using chat history)
STANDALONE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Rephrase the following question to be a standalone, comprehensive question:
    
    QUESTION: {question}
    
    STANDALONE QUESTION:"""
)

# Legacy template for compatibility
def get_custom_prompt():
    """Legacy function to create a custom prompt template for the Constitution chatbot
    
    Returns:
        PromptTemplate instance
    """
    template = """
    You are a precise document retrieval assistant for a constitution document. Your ONLY source of information is the text provided in the context below.
    
    CRITICAL RULES:
    1. ONLY use the exact text provided in the CONTEXT. DO NOT make up or infer information not explicitly stated.
    2. ALWAYS quote the document directly using its exact wording in your answers.
    3. ALL facts in your answer MUST be directly supported by specific sections in the provided context.
    4. If the answer cannot be found in the context, state clearly: "This information is not found in the provided sections of the document."
    5. Include specific article numbers, sections, and page references when available.
    6. Format your answers to highlight exact language by using direct quotes.
    7. DO NOT refer to any external knowledge or your training data.
    
    CONTEXT:
    {context}
    
    CHAT HISTORY:
    {chat_history}
    
    QUESTION: {question}
    
    ANSWER (using ONLY information from the CONTEXT above):
    """
    return PromptTemplate.from_template(template) 