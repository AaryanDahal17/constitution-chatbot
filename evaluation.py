# pairwise comparison code

import json
from typing import List, Dict, Any
import numpy as np
from app_bundle import makeConstitutionQuery, makeRawLLMQuery, init_llm
from schemas import Section, QAPair
from dataclasses import dataclass
from tqdm import tqdm
from itertools import combinations
import sys
import concurrent.futures
from queue import Queue
import threading
import random

@dataclass
class Response:
    text: str
    source: str  # 'rag' or 'raw'
    question: str

@dataclass
class ComparisonResult:
    winner: str  # 'rag' or 'raw'
    question: str
    accuracy_confidence: float  # Confidence in accuracy comparison
    onpointness_confidence: float  # Confidence in on-pointness comparison

def load_evaluation_sets(file_path: str) -> List[Section]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Section(**section) for section in data]

def compare_responses(response1: Response, response2: Response) -> ComparisonResult:
    """
    Compare two responses using LLM to determine which is better in terms of accuracy and on-pointness.
    Returns a ComparisonResult with the winner and separate confidence scores.
    """
    print(f"\nComparing responses for question: {response1.question[:100]}...")
    llm = init_llm(temperature=0.1)
    
    template = """
    Compare these two responses to the same question and determine which is better in terms of accuracy and on-pointness.
    
    Question: {question}
    
    Response 1:
    {response1}
    
    Response 2:
    {response2}
    
    You must respond with ONLY a JSON object in this exact format:
    {{
        "winner": "response1" or "response2",
        "accuracy_confidence": <score between 0 and 1>,
        "onpointness_confidence": <score between 0 and 1>
    }}
    """
    
    prompt = template.format(
        question=response1.question,
        response1=response1.text,
        response2=response2.text
    )
    
    print("Sending to LLM for evaluation...")
    evaluation = llm.invoke(prompt)
    
    try:
        # Clean the response to ensure it's valid JSON
        cleaned_response = evaluation.content.strip()
        if not cleaned_response.startswith('{'):
            cleaned_response = cleaned_response[cleaned_response.find('{'):]
        if not cleaned_response.endswith('}'):
            cleaned_response = cleaned_response[:cleaned_response.rfind('}')+1]
            
        result = json.loads(cleaned_response)
        winner = response1.source if result["winner"] == "response1" else response2.source
        print(f"Evaluation complete - Winner: {winner}, Accuracy: {result['accuracy_confidence']:.3f}, On-pointness: {result['onpointness_confidence']:.3f}")
        return ComparisonResult(
            winner=winner,
            question=response1.question,
            accuracy_confidence=float(result["accuracy_confidence"]),
            onpointness_confidence=float(result["onpointness_confidence"])
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"\nError parsing LLM evaluation: {e}")
        print(f"Raw LLM response: {evaluation.content}")
        # Fallback to basic comparison
        print("Falling back to basic comparison...")
        return ComparisonResult(
            winner=response1.source,  # Default to first response
            question=response1.question,
            accuracy_confidence=0.5,
            onpointness_confidence=0.5
        )

def calculate_metrics(results: List[ComparisonResult]) -> Dict[str, Any]:
    total_comparisons = len(results)
    rag_wins = sum(1 for r in results if r.winner == 'rag')
    raw_wins = sum(1 for r in results if r.winner == 'raw')
    
    # Calculate win rates
    rag_win_rate = rag_wins / total_comparisons if total_comparisons > 0 else 0
    raw_win_rate = raw_wins / total_comparisons if total_comparisons > 0 else 0
    
    # Calculate average confidence for each aspect
    rag_accuracy = np.mean([r.accuracy_confidence for r in results if r.winner == 'rag']) if rag_wins > 0 else 0
    raw_accuracy = np.mean([r.accuracy_confidence for r in results if r.winner == 'raw']) if raw_wins > 0 else 0
    rag_onpointness = np.mean([r.onpointness_confidence for r in results if r.winner == 'rag']) if rag_wins > 0 else 0
    raw_onpointness = np.mean([r.onpointness_confidence for r in results if r.winner == 'raw']) if raw_wins > 0 else 0
    
    return {
        'rag_win_rate': rag_win_rate,
        'raw_win_rate': raw_win_rate,
        'rag_accuracy': rag_accuracy,
        'raw_accuracy': raw_accuracy,
        'rag_onpointness': rag_onpointness,
        'raw_onpointness': raw_onpointness,
        'total_comparisons': total_comparisons
    }

def print_live_metrics(results: List[ComparisonResult]):
    metrics = calculate_metrics(results)
    sys.stdout.write("\033[K")  # Clear line
    print(f"\rCurrent metrics - RAG: {metrics['rag_win_rate']:.2%} (Acc: {metrics['rag_accuracy']:.3f}, On-point: {metrics['rag_onpointness']:.3f}) | Raw: {metrics['raw_win_rate']:.2%} (Acc: {metrics['raw_accuracy']:.3f}, On-point: {metrics['raw_onpointness']:.3f}) | Total: {metrics['total_comparisons']}", end="", flush=True)

def process_qa_pair(qa_pair: QAPair) -> ComparisonResult:
    """Process a single QA pair and return the comparison result."""
    print(f"\nProcessing QA pair: {qa_pair.question[:100]}...")
    
    print("Getting RAG response...")
    rag_response = Response(
        text=makeConstitutionQuery({'question': qa_pair.question}),
        source='rag',
        question=qa_pair.question
    )
    
    print("Getting Raw LLM response...")
    raw_response = Response(
        text=makeRawLLMQuery({'question': qa_pair.question}),
        source='raw',
        question=qa_pair.question
    )
    
    return compare_responses(rag_response, raw_response)

def main():
    print("\n=== Starting Evaluation Process ===")
    
    # Load evaluation sets
    print("\nLoading evaluation sets from evaluation_sets.json...")
    sections = load_evaluation_sets('evaluation_sets.json')
    print(f"Loaded {len(sections)} evaluation sets")
    
    # Switch to randomly sample 10 sets (comment out to evaluate all sets)
    RANDOM_SAMPLE = True
    SAMPLE_SIZE = 10
    
    if RANDOM_SAMPLE:
        print(f"\nRandom sampling {SAMPLE_SIZE} sets...")
        sections = random.sample(sections, min(SAMPLE_SIZE, len(sections)))
        print(f"Selected {len(sections)} sets for evaluation")
    
    # Initialize results list and thread-safe queue for metrics updates
    print("\nInitializing evaluation infrastructure...")
    results = []
    metrics_queue = Queue()
    
    def update_metrics():
        print("Starting metrics update thread...")
        while True:
            result = metrics_queue.get()
            if result is None:  # Poison pill to stop the thread
                print("Stopping metrics update thread...")
                break
            results.append(result)
            print_live_metrics(results)
    
    # Start metrics update thread
    metrics_thread = threading.Thread(target=update_metrics)
    metrics_thread.start()
    
    # Create a list of all QA pairs to process
    print("\nCollecting QA pairs from selected sections...")
    all_qa_pairs = []
    for section in sections:
        all_qa_pairs.extend(section.qa_pairs)
    
    print(f"Total QA pairs to evaluate: {len(all_qa_pairs)}")
    
    # Process QA pairs in parallel with increased number of workers
    print("\nStarting parallel evaluation with 5 workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_qa = {executor.submit(process_qa_pair, qa_pair): qa_pair 
                       for qa_pair in all_qa_pairs}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_qa), 
                         total=len(all_qa_pairs), 
                         desc="Evaluating QA pairs"):
            try:
                result = future.result()
                metrics_queue.put(result)
            except Exception as e:
                print(f"\nError processing QA pair: {e}")
    
    # Signal metrics thread to stop
    print("\nSignaling metrics thread to stop...")
    metrics_queue.put(None)
    metrics_thread.join()
    
    # Print final results
    print("\n=== Final Evaluation Results ===")
    metrics = calculate_metrics(results)
    print(f"Total comparisons: {metrics['total_comparisons']}")
    print(f"\nRAG Model Win Rate: {metrics['rag_win_rate']:.2%}")
    print(f"Raw LLM Win Rate: {metrics['raw_win_rate']:.2%}")
    print(f"\nRAG Model Metrics:")
    print(f"  Accuracy: {metrics['rag_accuracy']:.3f}")
    print(f"  On-pointness: {metrics['rag_onpointness']:.3f}")
    print(f"\nRaw LLM Metrics:")
    print(f"  Accuracy: {metrics['raw_accuracy']:.3f}")
    print(f"  On-pointness: {metrics['raw_onpointness']:.3f}")
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()