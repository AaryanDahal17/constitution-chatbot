import os
import json
from typing import List, Dict, Any
from langchain.docstore.document import Document
import glob

class ConstitutionJsonLoader:
    """Loader for the Nepal Constitution JSON dataset"""
    
    def __init__(self, dataset_path: str):
        """Initialize the loader with the path to the dataset
        
        Args:
            dataset_path: Path to the constitution-of-nepal-dataset folder
        """
        self.dataset_path = dataset_path
        self.index_path = os.path.join(dataset_path, "index.json")
        self.articles_path = os.path.join(dataset_path, "articles")
    
    def _read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read a JSON file and return its contents
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _format_article_content(self, article_data: Dict[str, Any], part_number: int, part_title: str) -> str:
        """Format an article's content into a readable string
        
        Args:
            article_data: Article data from JSON file
            part_number: Part number the article belongs to
            part_title: Title of the part the article belongs to
            
        Returns:
            Formatted string representation of the article
        """
        article_number = article_data.get("article_number", "")
        article_title = article_data.get("title", "")
        
        # Start with header
        formatted = f"PART {part_number}: {part_title}\n"
        formatted += f"Article {article_number}: {article_title}\n\n"
        
        # Add content based on structure
        content = article_data.get("content", [])
        
        # Check if content is a string (like in preamble) or a list of clauses
        if isinstance(content, str):
            formatted += content
        else:
            # Process clauses
            for clause in content:
                clause_number = clause.get("clause_number", "")
                clause_text = clause.get("text", "")
                formatted += f"({clause_number}) {clause_text}\n\n"
        
        return formatted
    
    def load(self) -> List[Document]:
        """Load all constitution articles and create Document objects
        
        Returns:
            List of Document objects with constitution content
        """
        documents = []
        
        # Read the index file
        index_data = self._read_json_file(self.index_path)
        
        # Process preamble if it exists
        if "preamble" in index_data and "file" in index_data["preamble"]:
            preamble_path = os.path.join(self.dataset_path, index_data["preamble"]["file"])
            try:
                preamble_data = self._read_json_file(preamble_path)
                
                # Create a document for the preamble
                preamble_text = self._format_article_content(
                    preamble_data, 
                    0,  # 0 for preamble as it's before Part 1
                    "Preamble"
                )
                
                documents.append(Document(
                    page_content=preamble_text,
                    metadata={
                        "source": preamble_path,
                        "part": "Preamble",
                        "article": "Preamble",
                        "article_number": 0
                    }
                ))
            except Exception as e:
                print(f"Error loading preamble: {e}")
        
        # Process all parts and articles
        for part in index_data.get("parts", []):
            part_number = part.get("part_number", "")
            part_title = part.get("title", "")
            
            for article in part.get("articles", []):
                article_number = article.get("article_number", "")
                article_title = article.get("title", "")
                article_file = article.get("file", "")
                
                if article_file:
                    try:
                        article_path = os.path.join(self.dataset_path, article_file)
                        article_data = self._read_json_file(article_path)
                        
                        # Format the article content
                        article_text = self._format_article_content(
                            article_data,
                            part_number,
                            part_title
                        )
                        
                        # Create a document for this article
                        documents.append(Document(
                            page_content=article_text,
                            metadata={
                                "source": article_path,
                                "part": part_number,
                                "part_title": part_title,
                                "article": article_title,
                                "article_number": article_number
                            }
                        ))
                    except Exception as e:
                        print(f"Error loading article {article_number}: {e}")
        
        return documents
    
    def get_all_article_files(self) -> List[str]:
        """Get paths to all article JSON files
        
        Returns:
            List of file paths
        """
        article_files = []
        pattern = os.path.join(self.articles_path, "**/*.json")
        article_files = glob.glob(pattern, recursive=True)
        return article_files 