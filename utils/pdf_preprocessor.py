import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFPreprocessor:
    """Preprocesses PDF documents into structured JSON format with enhanced metadata."""
    
    def __init__(self, pdf_path: str):
        """Initialize the preprocessor.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.output_dir = "processed_docs"
        self.json_path = os.path.join(self.output_dir, 
                                     Path(pdf_path).stem + "_processed.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_text_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract text from PDF with detailed metadata.
        
        Returns:
            List of dictionaries containing text and metadata
        """
        logger.info(f"Extracting text from {self.pdf_path}")
        
        doc = fitz.open(self.pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            
            # Extract page dimensions
            width, height = page.rect.width, page.rect.height
            
            # Extract images if needed
            # images = self._extract_images(page)
            
            # Create page metadata
            page_data = {
                "page_number": page_num + 1,  # 1-based page numbering
                "content": text,
                "dimensions": {"width": width, "height": height},
                "sections": self._identify_sections(text, page_num + 1),
                "article_refs": self._extract_article_references(text)
            }
            
            pages.append(page_data)
        
        logger.info(f"Extracted {len(pages)} pages from PDF")
        return pages
    
    def _identify_sections(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Identify sections in the text.
        
        Args:
            text: Page text
            page_num: Page number
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Look for article-like patterns
        article_pattern = r"(Article|ARTICLE|Art\.|Section|SECTION|\bSec\.\s*)\s*(\d+[A-Za-z]*)[\s\.]+"
        matches = re.finditer(article_pattern, text)
        
        for match in matches:
            section_type = match.group(1)
            section_num = match.group(2)
            
            # Get some context after the section header
            start_pos = match.start()
            end_pos = min(start_pos + 500, len(text))
            context = text[start_pos:end_pos]
            
            sections.append({
                "type": section_type,
                "number": section_num,
                "position": start_pos,
                "context": context.strip(),
                "page": page_num
            })
        
        return sections
    
    def _extract_article_references(self, text: str) -> List[str]:
        """Extract references to articles or sections.
        
        Args:
            text: The text to search for references
            
        Returns:
            List of article references
        """
        # Pattern to find references to articles
        ref_pattern = r"(?:refer(?:ring|s|red)?\s+to|as\s+per|according\s+to|under|in)\s+(Article|ARTICLE|Art\.|Section|SECTION|\bSec\.\s*)\s*(\d+[A-Za-z]*)"
        
        matches = re.finditer(ref_pattern, text, re.IGNORECASE)
        refs = []
        
        for match in matches:
            refs.append(f"{match.group(1)} {match.group(2)}")
        
        return list(set(refs))  # Remove duplicates
    
    def process_and_save(self) -> str:
        """Process the PDF and save as JSON.
        
        Returns:
            Path to the saved JSON file
        """
        # Check if already processed
        if os.path.exists(self.json_path):
            logger.info(f"Found existing processed file: {self.json_path}")
            return self.json_path
        
        # Extract pages with metadata
        pages = self.extract_text_with_metadata()
        
        # Create document structure with metadata
        document = {
            "metadata": {
                "source": self.pdf_path,
                "num_pages": len(pages),
                "processed_date": str(Path(self.pdf_path).stat().st_mtime),
                "file_size": os.path.getsize(self.pdf_path)
            },
            "pages": pages,
            "structure": self._build_document_structure(pages)
        }
        
        # Save to JSON
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed document to {self.json_path}")
        return self.json_path
    
    def _build_document_structure(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a hierarchical document structure from pages.
        
        Args:
            pages: List of page dictionaries
            
        Returns:
            Document structure dictionary
        """
        # Collect all sections across all pages
        all_sections = []
        for page in pages:
            for section in page["sections"]:
                section_entry = {
                    "type": section["type"],
                    "number": section["number"],
                    "page": page["page_number"],
                    "context": section["context"]
                }
                all_sections.append(section_entry)
        
        # Group by type (Article, Section, etc.)
        structure = {}
        for section in all_sections:
            section_type = section["type"]
            if section_type not in structure:
                structure[section_type] = []
            structure[section_type].append(section)
        
        return structure


def get_processed_document_path(pdf_path: str) -> str:
    """Get the path to the processed document, creating it if needed.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Path to the processed JSON file
    """
    preprocessor = PDFPreprocessor(pdf_path)
    return preprocessor.process_and_save() 