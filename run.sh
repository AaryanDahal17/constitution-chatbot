#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Nepal Constitution RAG Chatbot${NC}"
echo -e "${YELLOW}============================${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file with your Groq API key:"
    echo "GROQ_API_KEY=your_api_key_here"
    exit 1
fi

# Check if requirements are installed
echo -e "${YELLOW}Checking requirements...${NC}"
pip install -r requirements.txt > /dev/null

# Test Groq connection
echo -e "${YELLOW}Testing Groq LLM connection...${NC}"
python test_groq.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to connect to Groq LLM!${NC}"
    exit 1
fi

# Create or load vector store
echo -e "${YELLOW}Setting up vector store...${NC}"
python test_vectorstore.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create or load vector store!${NC}"
    exit 1
fi

# Ask user which version to run
echo -e "${GREEN}Which version would you like to run?${NC}"
echo "1) Command-line version (app.py)"
echo "2) Basic web interface (streamlit_app.py)"
echo "3) Advanced web interface (advanced_app.py)"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting command-line version...${NC}"
        python app.py
        ;;
    2)
        echo -e "${GREEN}Starting basic web interface...${NC}"
        streamlit run streamlit_app.py
        ;;
    3)
        echo -e "${GREEN}Starting advanced web interface...${NC}"
        streamlit run advanced_app.py
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac 