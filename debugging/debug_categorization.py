#!/usr/bin/env python3
"""
Debug script to identify why categorization still returns ENTITY
"""

import sys
from pathlib import Path
import spacy
import numpy as np

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tax_span_cat.SpanCategorizer import SpanCategorizer

def create_simple_taxonomy():
    """Create a very simple taxonomy for debugging"""
    return {
        "children": {
            "person.n.01": {
                "label": "Persons",
                "embedding": np.array([1.0, 0.0, 0.0]),  # Add explicit embedding
                "children": {
                    "victim.n.01": {
                        "label": "Victims", 
                        "description": "Crime victims",
                        "embedding": np.array([0.9, 0.1, 0.0])  # Add explicit embedding
                    }
                }
            }
        }
    }

def debug_categorization():
    print("=== DEBUGGING CATEGORIZATION ===")
    
    # Create simple taxonomy
    taxonomy = create_simple_taxonomy()
    print("Created simple taxonomy:")
    print(f"Root children: {list(taxonomy['children'].keys())}")
    
    # Create categorizer
    categorizer = SpanCategorizer(taxonomy=taxonomy, threshold=0.1)
    print(f"Threshold: {categorizer.threshold}")
    
    # Test direct hierarchical search
    print("\n=== Testing direct hierarchical search ===")
    result = categorizer._hierarchical_sem_search(
        query="victim",
        current_label="ENTITY",
        current_node={"children": taxonomy}
    )
    print(f"Direct search result for 'victim': {result}")
    
    # Test with SpaCy doc
    print("\n=== Testing with SpaCy doc ===")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The victim was hurt.")
    
    print("Noun chunks:")
    for chunk in doc.noun_chunks:
        print(f"  '{chunk.text}' (start: {chunk.start}, end: {chunk.end})")
    
    # Process document
    result_doc = categorizer(doc)
    print(f"\nSpan categories found:")
    for span in result_doc.spans.get('sc', []):
        print(f"  '{span.text}' -> '{span.label_}'")

if __name__ == "__main__":
    debug_categorization()