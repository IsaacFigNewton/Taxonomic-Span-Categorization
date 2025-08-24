#!/usr/bin/env python3
"""
Debug script to check taxonomy structure after embedding
"""

import sys
from pathlib import Path
import json

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tax_span_cat.SpanCategorizer import SpanCategorizer

def debug_structure():
    print("=== DEBUGGING TAXONOMY STRUCTURE ===")
    
    # Create simple taxonomy
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
    print(json.dumps(list(taxonomy.keys()), indent=2))
    
    # Create categorizer (this will embed the taxonomy)
    categorizer = SpanCategorizer(taxonomy=taxonomy, threshold=0.1)
    
    print("\nEmbedded taxonomy structure:")
    embedded_taxonomy = categorizer.taxonomy
    print(f"Top level keys: {list(embedded_taxonomy.keys())}")
    
    if 'children' in embedded_taxonomy:
        print(f"children keys: {list(embedded_taxonomy['children'].keys())}")
        
    print(f"\nWhat gets passed to hierarchical search:")
    search_node = {"children": embedded_taxonomy}
    print(f"search_node keys: {list(search_node.keys())}")
    print(f"search_node['children'] keys: {list(search_node['children'].keys())}")
    
    if 'children' in search_node['children']:
        print(f"search_node['children']['children'] keys: {list(search_node['children']['children'].keys())}")
    
    print(f"\nCorrect structure should be:")
    correct_node = embedded_taxonomy
    print(f"correct_node keys: {list(correct_node.keys())}")
    if 'children' in correct_node:
        print(f"correct_node['children'] keys: {list(correct_node['children'].keys())}")

if __name__ == "__main__":
    debug_structure()