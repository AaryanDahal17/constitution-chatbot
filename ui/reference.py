import streamlit as st
from typing import List, Dict, Any, Optional
from utils.text_matcher import format_highlighted_text

def create_reference_button(ref_id: str, page_num: int) -> str:
    """Create an HTML button for a reference.
    
    Args:
        ref_id: Reference ID for the button
        page_num: Page number to display
        
    Returns:
        HTML code for the button
    """
    return f"""
    <button 
        class="ref-button" 
        id="{ref_id}" 
        onclick="toggleRef('{ref_id}')"
        style="background-color: #457b9d; 
               color: white; 
               border: none; 
               border-radius: 4px; 
               padding: 2px 8px; 
               font-size: 12px; 
               cursor: pointer;
               margin-right: 5px;">
        Page {page_num}
    </button>
    """

def create_reference_container(ref_id: str, content: str) -> str:
    """Create an HTML container for reference content.
    
    Args:
        ref_id: Reference ID for the container
        content: HTML content to display
        
    Returns:
        HTML code for the container
    """
    return f"""
    <div 
        id="{ref_id}_content" 
        style="display: none; 
               background-color: #f1faee; 
               border: 1px solid #457b9d; 
               border-radius: 4px; 
               padding: 10px; 
               margin: 5px 0; 
               max-height: 200px; 
               overflow-y: auto;">
        {content}
    </div>
    """

def create_toggle_script() -> str:
    """Create JavaScript for toggling reference visibility.
    
    Returns:
        JavaScript code as a string
    """
    return """
    <script>
    function toggleRef(refId) {
        var content = document.getElementById(refId + '_content');
        if (content.style.display === 'none') {
            content.style.display = 'block';
        } else {
            content.style.display = 'none';
        }
    }
    </script>
    """

def display_references(matches: Dict[str, List], answer_text: str):
    """Display reference buttons and containers for matches using Streamlit components.
    
    Args:
        matches: Dictionary of matches from highlight_matches
        answer_text: The answer text
    """
    if not matches:
        return
    
    # Add a subtle divider
    st.markdown("---")
    st.markdown("<small style='color: #666;'>References:</small>", unsafe_allow_html=True)
    
    # Create tabs for each document with matches
    if len(matches) > 0:
        all_tabs = []
        for doc_id, match_list in matches.items():
            for i, (match_text, metadata) in enumerate(match_list):
                page_num = metadata.get('page', 'N/A')
                tab_label = f"Page {page_num}"
                all_tabs.append(tab_label)
        
        # Only create tabs if we have content
        if all_tabs:
            tabs = st.tabs(all_tabs)
            
            # Fill each tab with content
            tab_index = 0
            for doc_id, match_list in matches.items():
                for i, (match_text, metadata) in enumerate(match_list):
                    with tabs[tab_index]:
                        # Show highlighted content
                        highlighted_text = format_highlighted_text(
                            metadata.get('source_text', match_text), 
                            match_text
                        )
                        
                        # Show some metadata about the source
                        page_num = metadata.get('page', 'N/A')
                        st.markdown(f"**Source**: Constitution of Nepal, Page {page_num}")
                        
                        # Display the highlighted text with explicit text color
                        st.markdown(
                            f"""<div style="background-color: #f1faee; 
                                   border: 1px solid #457b9d; 
                                   border-radius: 4px; 
                                   padding: 10px; 
                                   margin: 5px 0; 
                                   max-height: 300px; 
                                   overflow-y: auto;
                                   color: #333333;">
                                {highlighted_text}
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    tab_index += 1

def init_reference_css():
    """Initialize CSS for reference components"""
    st.markdown("""
    <style>
    /* Style the reference section */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    /* Tab styling with guaranteed visible text */
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        background-color: #457b9d;
        border-radius: 4px;
        padding: 0 16px;
    }
    
    /* Override internal Streamlit tab styles to ensure text visibility */
    .stTabs [data-baseweb="tab"] div {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Selected tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #1d3557;
    }
    
    /* Highlighted text styles */
    span[style*="background-color: #FFA500"] {
        padding: 2px 0;
        border-radius: 2px;
        color: black !important;
    }
    
    /* Ensure text is visible in all contexts */
    .stTabs [role="tabpanel"] {
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True) 