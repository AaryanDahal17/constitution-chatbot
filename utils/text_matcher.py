import re
import html
import streamlit as st
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

def find_best_sentence_in_source(query: str, source_sentences: List[str]) -> str:
    """Find the best matching sentence in source for a given query.
    
    Args:
        query: The query text to find in source
        source_sentences: List of source sentences to search through
        
    Returns:
        Best matching sentence, regardless of match quality
    """
    best_match = None
    best_ratio = 0.0  # Accept any match, no matter how poor
    
    for src_sentence in source_sentences:
        # Calculate similarity ratio
        ratio = SequenceMatcher(None, query.lower(), src_sentence.lower()).ratio()
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = src_sentence
    
    # If no match found, return the first sentence as fallback
    if not best_match and source_sentences:
        return source_sentences[0]
        
    return best_match if best_match else ""

def find_matching_text(source_text: str, answer_text: str) -> List[str]:
    """Find sections of source text that closely match parts of the answer.
    
    Args:
        source_text: Original source document text
        answer_text: Generated answer text
        
    Returns:
        List of matching source text sections
    """
    # Split the text into sentences for better matching
    source_sentences = re.split(r'(?<=[.!?])\s+', source_text)
    answer_sentences = re.split(r'(?<=[.!?])\s+', answer_text)
    
    # Look for quoted text (which is likely direct quotes from constitution)
    quote_matches = []
    
    # Find text between double quotes
    double_quotes = re.findall(r'"([^"]+)"', answer_text)
    if double_quotes:
        quote_matches.extend(double_quotes)
    
    # Find text between single quotes
    single_quotes = re.findall(r"'([^']+)'", answer_text)
    if single_quotes:
        quote_matches.extend(single_quotes)
    
    # Look for article/section references in the answer
    article_refs = []
    ref_pattern = r"(Article|ARTICLE|Art\.|Section|SECTION|\bSec\.\s*)\s*(\d+[A-Za-z]*)"
    ref_matches = re.finditer(ref_pattern, answer_text)
    
    for match in ref_matches:
        article_refs.append(f"{match.group(1)} {match.group(2)}")
    
    # For direct quotes, find the exact matching sentences
    exact_matches = []
    for quote in quote_matches:
        if len(quote.strip()) < 5:  # Skip very short quotes
            continue
            
        for src_sentence in source_sentences:
            if quote in src_sentence and src_sentence not in exact_matches:
                exact_matches.append(src_sentence)
    
    # If we have exact matches from quotes, prioritize them
    if exact_matches:
        return exact_matches
    
    # Otherwise use fuzzy matching
    matches = []
    
    # For each sentence in the answer, find the best matching source sentence
    for ans_sentence in answer_sentences:
        if len(ans_sentence.strip()) < 10:  # Skip very short sentences
            continue
            
        best_match = None
        best_ratio = 0.5  # Lower threshold to ensure we get matches
        
        for src_sentence in source_sentences:
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, ans_sentence.lower(), src_sentence.lower()).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = src_sentence
        
        if best_match and best_match not in matches:
            matches.append(best_match)
    
    # If still no matches, take the first few sentences from source as fallback
    if not matches and source_sentences:
        # Take the first 3 sentences as fallback references
        matches = source_sentences[:min(3, len(source_sentences))]
    
    return matches

def highlight_matches(source_docs: List[Dict], answer_text: str) -> Dict[str, List[Tuple[str, Dict]]]:
    """Find and organize matching text from source documents.
    
    Args:
        source_docs: List of source documents
        answer_text: Generated answer text
        
    Returns:
        Dictionary mapping source document IDs to lists of (matching_text, metadata) tuples
    """
    results = {}
    
    # If no source documents provided, handle gracefully
    if not source_docs:
        st.warning("No source documents found for this answer. The information may not be directly from the document.")
        return results
    
    # Extract article/section references from the answer
    article_pattern = r"(Article|ARTICLE|Art\.|Section|SECTION|\bSec\.\s*)\s*(\d+[A-Za-z]*)"
    article_refs = []
    
    for match in re.finditer(article_pattern, answer_text):
        article_refs.append(f"{match.group(1)} {match.group(2)}".lower())
    
    # First prioritize documents that are explicitly referenced sections
    section_matches = {}
    for doc in source_docs:
        # Check if this is a section document
        if doc.metadata.get('is_section', False):
            section_id = f"{doc.metadata.get('section_type', 'Section')} {doc.metadata.get('section_number', '')}".lower()
            
            # If this section is referenced in the answer, prioritize it
            if section_id in article_refs:
                doc_id = f"section_{doc.metadata.get('section_type', 'unknown')}_{doc.metadata.get('section_number', 'unknown')}"
                
                # Use the section context as the match
                metadata = doc.metadata.copy()
                metadata['source_text'] = doc.page_content
                metadata['match_type'] = 'direct_section_reference'
                
                # Split into sentences to get the most relevant part
                sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
                if sentences:
                    intro_text = sentences[0]  # First sentence often has the title/intro
                    section_matches[doc_id] = [(intro_text, metadata)]
    
    # If we have direct section matches, use those
    if section_matches:
        return section_matches
    
    # Otherwise fallback to regular content matching
    for doc in source_docs:
        doc_text = doc.page_content
        
        # Create more descriptive IDs based on metadata
        if doc.metadata.get('is_section', False):
            doc_id = f"section_{doc.metadata.get('section_type', 'unknown')}_{doc.metadata.get('section_number', 'unknown')}"
        else:
            doc_id = f"page_{doc.metadata.get('page', 'unknown')}"
        
        # Store the full text in metadata for reference
        metadata = doc.metadata.copy()
        metadata['source_text'] = doc_text
        
        # Find matches between this document and the answer
        matches = find_matching_text(doc_text, answer_text)
        
        if matches:
            if doc_id not in results:
                results[doc_id] = []
            
            # Store each match with metadata
            for match in matches:
                results[doc_id].append((match, metadata))
    
    # If we still have no results, use the first document as fallback
    if not results and source_docs:
        doc = source_docs[0]
        doc_text = doc.page_content
        
        if doc.metadata.get('is_section', False):
            doc_id = f"section_{doc.metadata.get('section_type', 'unknown')}_{doc.metadata.get('section_number', 'unknown')}"
        else:
            doc_id = f"page_{doc.metadata.get('page', 'unknown')}"
        
        metadata = doc.metadata.copy()
        metadata['source_text'] = doc_text
        
        # Split into sentences and take the first 2
        sentences = re.split(r'(?<=[.!?])\s+', doc_text)
        matches = sentences[:min(2, len(sentences))]
        
        results[doc_id] = [(match, metadata) for match in matches]
    
    return results

def format_highlighted_text(text: str, match: str) -> str:
    """Format text with HTML to highlight the matching part.
    
    Args:
        text: Full text to display
        match: Substring to highlight
        
    Returns:
        HTML formatted text with highlight
    """
    if not match or match not in text:
        return html.escape(text)
    
    # Escape HTML to prevent rendering issues
    escaped_text = html.escape(text)
    escaped_match = html.escape(match)
    
    # Simple replacement with highlighted span - using a more visible orange highlight with black text
    highlighted = escaped_text.replace(
        escaped_match, 
        f'<span style="background-color: #FFA500; color: #000000; font-weight: 500;">{escaped_match}</span>'
    )
    
    return highlighted 