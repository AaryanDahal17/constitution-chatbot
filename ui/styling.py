import streamlit as st

def apply_custom_css():
    """Apply custom CSS to the Streamlit app"""
    st.markdown("""
    <style>
        /* Main headers */
        .main-header {
            font-size: 2.5rem;
            color: #e63946;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #457b9d;
        }
        
        /* Background and text colors */
        .stApp {
            background-color: #f1faee;
        }
        
        
        /* HEADER/NAVBAR STYLING - COMPREHENSIVE */
        /* Main navbar container */
        header[data-testid="stHeader"] {
            background-color: #1d3557 !important;
        }
        
        /* All elements in header */
        header[data-testid="stHeader"] * {
            color: white !important;
        }
        
        /* SVG icons in header */
        header[data-testid="stHeader"] svg {
            fill: white !important;
        }
        
        /* Specifically target the hamburger menu */
        button[kind="headerNoPadding"] {
            color: white !important;
        }
        
        /* Dropdown menu in header */
        .stDeployButton > button {
            color: white !important;
            border-color: white !important;
        }
        
        /* View options container */
        .viewerBadge {
            color: white !important;
            background-color: #2c5282 !important;
        }
        
        /* All buttons in header */
        header[data-testid="stHeader"] button {
            color: white !important;
        }
        
        /* All header icons */
        header[data-testid="stHeader"] path, 
        header[data-testid="stHeader"] circle, 
        header[data-testid="stHeader"] rect, 
        header[data-testid="stHeader"] line {
            stroke: white !important;
        }
        
        /* All SVG elements in header */
        header[data-testid="stHeader"] svg * {
            fill: white !important;
            stroke: white !important;
        }
        
        /* SIDEBAR STYLING - COMPREHENSIVE */
        /* Sidebar container */
        [data-testid="stSidebar"] {
            background-color: #1d3557 !important;
        }
        
        /* All direct children of sidebar */
        [data-testid="stSidebar"] > div > div > div > * {
            color: white !important;
        }
        
        /* All text elements in sidebar */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stText {
            color: white !important;
        }
        
        /* Sidebar selectbox */
        [data-testid="stSidebar"] .stSelectbox label {
            color: white !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
            color: white !important;
            background-color: #2c5282 !important;
        }
        [data-testid="stSidebar"] .stSelectbox [role="combobox"],
        [data-testid="stSidebar"] .stSelectbox [role="listbox"] {
            background-color: #2c5282 !important;
        }
        
        /* Sidebar slider */
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSlider p,
        [data-testid="stSidebar"] .stSlider div {
            color: white !important;
        }
        
        /* Expander arrows and buttons */
        .st-emotion-cache-16txtl3,
        .st-emotion-cache-ue6h4q,
        .st-emotion-cache-1wmy9hl,
        .st-emotion-cache-16idsys p {
            color: white !important;
        }
        
        /* Dropdown elements */
        [data-baseweb="select"] {
            background-color: #2c5282;
        }
        [data-baseweb="popover"] * {
            background-color: #2c5282;
            color: white !important;
        }
        
        /* Cover all emotion cache classes - Streamlit uses these extensively */
        .st-emotion-cache-* {
            color: white !important;
        }
        
        /* Target dropdown button styling */
        div[data-baseweb="select"] * {
            color: white !important;
            fill: white !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg path {
            fill: white !important;
        }
        
        /* Reference styling */
        .ref-button {
            transition: background-color 0.3s;
        }
        .ref-container {
            margin-top: 5px;
            font-size: 0.9em;
            color: #333333;
        }
        .highlight {
            background-color: #FFA500 !important;
            color: #000000 !important;
            padding: 2px 0;
            font-weight: 500;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab"] {
            background-color: #457b9d;
            height: 35px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .stTabs [data-baseweb="tab"] div {
            color: white !important;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1d3557;
        }
        
        /* Chat elements proper coloring */
        .stChatMessage {
            color: #333333;
        }
        .stChatMessage [data-testid="stChatMessageContent"] {
            color: #333333;
        }
        
        /* Chat input field */
        .stChatInput [data-testid="stChatInputTextArea"] {
            color: #333333;
            background-color: white;
        }
        
        /* New theme-system compatibility */
        [data-testid="stAppViewBlockContainer"] {
            background-color: #f1faee;
        }
        
        /* SVG Icons in sidebar and header */
        [data-testid="stSidebar"] svg,
        [data-testid="stSidebar"] svg *,
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] svg * {
            fill: white !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

def get_professional_style():
    """Return HTML/CSS for professional looking chat messages"""
    return '''<div style="background: linear-gradient(to right, #2c3e50, #34495e); 
            color: #ecf0f1; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #1abc9c; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);">'''

def set_page_config():
    """Set Streamlit page configuration"""
    st.set_page_config(
        page_title="Constitution Chatbot",
        page_icon="📚",
        layout="wide"
    ) 