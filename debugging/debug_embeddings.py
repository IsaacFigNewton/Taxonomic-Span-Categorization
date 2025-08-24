#!/usr/bin/env python3
"""
Debug script to check embedding process
"""

import sys
from pathlib import Path
import spacy
import numpy as np

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tax_span_cat.SpanCategorizer import SpanCategorizer

def debug_embeddings():
    print("=== DEBUGGING EMBEDDINGS ===")
    
    # Create simple taxonomy WITHOUT pre-computed embeddings
    taxonomy = {
        "children": {
            "person.n.01": {
                "label": "Persons",
                "description": "Human beings and individuals",
                "wordnet_synsets": ["person.n.01"],
                "children": {
                    "victim.n.01": {
                        "label": "Victims", 
                        "description": "Crime victims",
                        "wordnet_synsets": ["victim.n.01"]
                    }
                }
            }
        }
    }
    
    print("Original taxonomy structure:")
    print(f"Root children: {list(taxonomy['children'].keys())}")
    print(f"person.n.01 has: {list(taxonomy['children']['person.n.01'].keys())}")
    print(f"victim.n.01 has: {list(taxonomy['children']['person.n.01']['children']['victim.n.01'].keys())}")
    
    # Create categorizer (this will embed the taxonomy)
    categorizer = SpanCategorizer(taxonomy=taxonomy, threshold=0.1)
    
    print("\n=== After embedding ===")
    embedded_taxonomy = categorizer.taxonomy
    print(f"Root children: {list(embedded_taxonomy.keys())}")
    if 'person.n.01' in embedded_taxonomy:
        person_node = embedded_taxonomy['person.n.01']
        print(f"person.n.01 has: {list(person_node.keys())}")
        print(f"person.n.01 embedding shape: {person_node.get('embedding', 'MISSING').shape if 'embedding' in person_node else 'MISSING'}")
        
        if 'children' in person_node and 'victim.n.01' in person_node['children']:
            victim_node = person_node['children']['victim.n.01']
            print(f"victim.n.01 has: {list(victim_node.keys())}")
            print(f"victim.n.01 embedding shape: {victim_node.get('embedding', 'MISSING').shape if 'embedding' in victim_node else 'MISSING'}")
    
    print("\n=== Testing query embedding ===")
    query = "victim"
    query_embedding = categorizer._embed(query)
    print(f"Query '{query}' embedding shape: {query_embedding.shape}")
    print(f"Query embedding (first 5): {query_embedding[:5]}")
    
    print("\n=== Testing semantic search ===")
    # Test the semantic search directly
    if 'person.n.01' in embedded_taxonomy and 'embedding' in embedded_taxonomy['person.n.01']:
        person_embedding = embedded_taxonomy['person.n.01']['embedding']
        print(f"Person embedding (first 5): {person_embedding[:5]}")
        
        similarity, idx = categorizer._semantic_search(query_embedding, [person_embedding])
        print(f"Similarity between '{query}' and 'person.n.01': {similarity}")
        print(f"Threshold: {categorizer.threshold}")
        print(f"Above threshold: {similarity > categorizer.threshold}")

if __name__ == "__main__":
    debug_embeddings()