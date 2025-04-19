import streamlit as st
from ui.styling import get_professional_style
from ui.reference import display_references, init_reference_css
from rag.callbacks import StreamHandler
from rag.chain import create_rag_chain
from utils.text_matcher import highlight_matches

def init_chat_state():
    """Initialize chat state if not already done"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Initialize reference CSS
    init_reference_css()

def display_chat_history():
    """Display all messages in the chat history"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f'{get_professional_style()}{message["content"]}</div>', unsafe_allow_html=True)
                
                # Display references if available
                if "references" in message:
                    display_references(message["references"], message["content"])

def handle_user_input(vector_store, llm, settings):
    """Process user input and generate response
    
    Args:
        vector_store: FAISS vector store
        llm: LLM instance
        settings: Settings from sidebar
    """
    # Get user input from chat input
    prompt = st.chat_input("Ask a question about the Nepal Constitution")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            
            try:
                # Create conversation chain with streaming
                conversation = create_rag_chain(
                    vector_store, 
                    llm, 
                    _use_custom_prompt=settings["use_custom_prompt"],
                    _streaming_callback=stream_handler
                )
                
                # Get response with streaming
                response = conversation({"question": prompt})
                final_response = response["answer"]
                
                # Process source documents to find matching text
                source_docs = response.get("source_documents", [])
                
                # Show "searching for references" message during processing
                ref_placeholder = st.empty()
                ref_placeholder.info("Finding constitutional references...")
                
                # Extract matches - this should always return at least something now
                matches = highlight_matches(source_docs, final_response)
                
                # Clear the temporary message
                ref_placeholder.empty()
                
                # Display final response without cursor
                message_placeholder.markdown(
                    f'{get_professional_style()}{final_response}</div>',
                    unsafe_allow_html=True
                )
                
                # Display reference buttons and source info 
                # This should always show something now due to our fallback mechanisms
                display_references(matches, final_response)
                
                # Store the response and matches in session state
                response_data = {
                    "role": "assistant", 
                    "content": final_response,
                    "references": matches
                }
                
                # Add assistant response to chat history
                st.session_state.messages.append(response_data)
                
            except Exception as e:
                # Handle any errors gracefully
                error_msg = f"Error retrieving answer: {str(e)}"
                message_placeholder.markdown(
                    f'{get_professional_style()}{error_msg}</div>',
                    unsafe_allow_html=True
                )
                st.error(f"Technical details: {str(e)}")
                
                # Add error message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                }) 