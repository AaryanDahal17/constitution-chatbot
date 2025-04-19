import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
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
    """Display references as tabs with enhanced document information
    
    Args:
        matches: Dictionary of matches from highlight_matches
        answer_text: The answer text
    """
    if not matches:
        st.info("No specific references found for this answer.")
        return
    
    # Group references by page number for better organization
    pages_dict = {}
    section_dict = {}
    
    # Process and organize references by type
    for doc_id, match_list in matches.items():
        for match_text, metadata in match_list:
            # Check if this is a section reference
            if 'is_section' in metadata and metadata['is_section']:
                section_type = metadata.get('section_type', 'Section')
                section_number = metadata.get('section_number', '')
                section_key = f"{section_type} {section_number}"
                
                if section_key not in section_dict:
                    section_dict[section_key] = {
                        "page": metadata.get('page', 'Unknown'),
                        "matches": [],
                        "metadata": metadata
                    }
                section_dict[section_key]["matches"].append(match_text)
            else:
                # Regular page reference
                page_num = metadata.get('page', 'Unknown')
                if page_num not in pages_dict:
                    pages_dict[page_num] = {
                        "matches": [],
                        "metadata": metadata
                    }
                pages_dict[page_num]["matches"].append(match_text)
    
    # Create tabs for references
    if section_dict or pages_dict:
        # Determine what kinds of tabs to show
        tab_names = []
        
        # Add section tabs first (they're usually more specific)
        for section_key in sorted(section_dict.keys()):
            page_num = section_dict[section_key]["page"]
            tab_names.append(f"{section_key} (Page {page_num})")
        
        # Then add regular page tabs 
        for page_num in sorted(pages_dict.keys()):
            # Skip pages that are already covered by sections
            if any(section_dict[s]["page"] == page_num for s in section_dict):
                continue
            tab_names.append(f"Page {page_num}")
        
        # Create tab interface
        if tab_names:
            tabs = st.tabs(tab_names)
            
            # Fill section tabs
            tab_index = 0
            for section_key in sorted(section_dict.keys()):
                section_data = section_dict[section_key]
                with tabs[tab_index]:
                    # Show section metadata
                    st.caption(f"Reference: {section_key}")
                    
                    # Create a container for the content
                    content_container = st.container(border=True)
                    with content_container:
                        # Get full source text if available
                        source_text = section_data["metadata"].get('source_text', '')
                        
                        if source_text:
                            highlighted_text = source_text
                            # Highlight each match in the source text
                            for match in section_data["matches"]:
                                if match in source_text:
                                    highlighted_text = highlighted_text.replace(
                                        match,
                                        f'<span style="background-color: #FFA500; color: #000000; font-weight: 500;">{match}</span>'
                                    )
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                        else:
                            # Just show the matches if we don't have the full source
                            for match in section_data["matches"]:
                                st.markdown(f"• {match}")
                
                tab_index += 1
            
            # Fill page tabs
            for page_num in sorted(pages_dict.keys()):
                # Skip pages that are already covered by sections
                if any(section_dict[s]["page"] == page_num for s in section_dict):
                    continue
                    
                page_data = pages_dict[page_num]
                with tabs[tab_index]:
                    # Show page information
                    st.caption(f"Page {page_num} Reference")
                    
                    # Create a container for the content
                    content_container = st.container(border=True)
                    with content_container:
                        # Get full source text if available
                        source_text = page_data["metadata"].get('source_text', '')
                        
                        if source_text:
                            highlighted_text = source_text
                            # Highlight each match in the source text
                            for match in page_data["matches"]:
                                if match in source_text:
                                    highlighted_text = highlighted_text.replace(
                                        match,
                                        f'<span style="background-color: #FFA500; color: #000000; font-weight: 500;">{match}</span>'
                                    )
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                        else:
                            # Just show the matches if we don't have the full source
                            for match in page_data["matches"]:
                                st.markdown(f"• {match}")
                
                tab_index += 1

def init_reference_css():
    """Initialize CSS for reference components"""
    st.markdown("""
    <style>
    /* Reference styling */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: #457b9d;
        border-radius: 4px;
        padding: 4px 12px;
        margin: 0;
    }
    
    /* Make sure tab text is visible */
    [data-testid="stTabs"] [data-baseweb="tab"] div {
        color: white !important;
        font-weight: 500;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #1d3557;
    }
    
    /* Highlighted text styling */
    span[style*="background-color: #FFA500"] {
        padding: 2px 0;
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True) 