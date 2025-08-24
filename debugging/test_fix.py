#!/usr/bin/env python3
"""
Test the fix for categorization
"""

import sys
from pathlib import Path
import spacy

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tax_span_cat.SpanCategorizer import SpanCategorizer

def test_fix():
    print("=== TESTING THE FIX ===")
    
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
                    },
                    "suspect.n.01": {
                        "label": "Suspects", 
                        "description": "Criminal suspects",
                        "wordnet_synsets": ["suspect.n.01"]
                    }
                }
            },
            "artifact.n.01": {
                "label": "Artifacts",
                "description": "Human-made objects",
                "wordnet_synsets": ["artifact.n.01"],
                "children": {
                    "weapon.n.01": {
                        "label": "Weapons", 
                        "description": "Weapons and arms",
                        "wordnet_synsets": ["weapon.n.01"]
                    },
                    "vehicle.n.01": {
                        "label": "Vehicles", 
                        "description": "Transportation vehicles",
                        "wordnet_synsets": ["vehicle.n.01"]
                    }
                }
            }
        }
    }
    
    # Create categorizer with low threshold to allow matches
    categorizer = SpanCategorizer(taxonomy=taxonomy, threshold=0.1)
    
    # Test with SpaCy doc
    nlp = spacy.load("en_core_web_sm")
    test_sentences = [
        "The victim was hurt.",
        "The suspect was arrested.",
        "The weapon was found.",
        "The vehicle was stolen."
    ]
    
    for sentence in test_sentences:
        print(f"\nTesting: '{sentence}'")
        doc = nlp(sentence)
        result_doc = categorizer(doc)
        
        for span in result_doc.spans.get('sc', []):
            print(f"  '{span.text}' -> '{span.label_}'")

if __name__ == "__main__":
    test_fix()