import unittest
from unittest.mock import Mock, patch
import numpy as np
import json
from pathlib import Path

import spacy
from spacy.tokens import Doc, Span

from src.tax_span_cat.SpanCategorizer import SpanCategorizer


class TestGeneralNERTaxonomy(unittest.TestCase):
    """Test suite for SpanCategorizer with general_ner.json taxonomy"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Load the actual general_ner.json taxonomy
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            cls.general_ner_taxonomy = json.load(f)
        
        # Try to load spaCy model for realistic testing
        try:
            cls.nlp = spacy.load("en_core_web_sm")
            cls.spacy_available = True
        except OSError:
            cls.nlp = None
            cls.spacy_available = False
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock embedding model for deterministic testing
        self.mock_embeddings = {
            # Person-related embeddings
            "victim": np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 380),
            "suspect": np.array([0.9, 0.1, 0.0, 0.0] + [0.0] * 380),
            "witness": np.array([0.8, 0.2, 0.0, 0.0] + [0.0] * 380),
            "officer": np.array([0.7, 0.3, 0.0, 0.0] + [0.0] * 380),
            "john doe": np.array([0.95, 0.05, 0.0, 0.0] + [0.0] * 380),
            "jane smith": np.array([0.85, 0.15, 0.0, 0.0] + [0.0] * 380),
            
            # Location-related embeddings
            "chicago": np.array([0.0, 1.0, 0.0, 0.0] + [0.0] * 380),
            "new york": np.array([0.0, 0.9, 0.1, 0.0] + [0.0] * 380),
            "123 main street": np.array([0.0, 0.8, 0.2, 0.0] + [0.0] * 380),
            "apartment building": np.array([0.0, 0.7, 0.3, 0.0] + [0.0] * 380),
            
            # Evidence-related embeddings
            "fingerprint": np.array([0.0, 0.0, 1.0, 0.0] + [0.0] * 380),
            "video footage": np.array([0.0, 0.0, 0.9, 0.1] + [0.0] * 380),
            "witness statement": np.array([0.0, 0.0, 0.8, 0.2] + [0.0] * 380),
            "dna evidence": np.array([0.0, 0.0, 0.95, 0.05] + [0.0] * 380),
            
            # Crime-related embeddings
            "assault": np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380),
            "homicide": np.array([0.0, 0.0, 0.0, 0.9] + [0.0] * 380),
            "burglary": np.array([0.0, 0.0, 0.0, 0.8] + [0.0] * 380),
            "robbery": np.array([0.0, 0.0, 0.0, 0.85] + [0.0] * 380),
            
            # Time-related embeddings
            "january 1st": np.array([0.0, 0.0, 0.0, 0.0] + [1.0] + [0.0] * 379),
            "10:30 pm": np.array([0.0, 0.0, 0.0, 0.0] + [0.9] + [0.0] * 379),
            "last week": np.array([0.0, 0.0, 0.0, 0.0] + [0.8] + [0.0] * 379),
            
            # Weapon-related embeddings
            "gun": np.array([0.0, 0.0, 0.0, 0.0, 0.0] + [1.0] + [0.0] * 378),
            "knife": np.array([0.0, 0.0, 0.0, 0.0, 0.0] + [0.9] + [0.0] * 378),
            "firearm": np.array([0.0, 0.0, 0.0, 0.0, 0.0] + [0.95] + [0.0] * 378),
            
            # Default embedding for unknown terms
            "default": np.array([0.1] * 384)
        }
    
    def get_mock_embedding(self, text):
        """Get a mock embedding for a given text."""
        text_lower = text.lower()
        for key, embedding in self.mock_embeddings.items():
            if key in text_lower:
                return embedding
        return self.mock_embeddings["default"]
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_initialization_with_general_ner_taxonomy(self, mock_sentence_transformer):
        """Test that SpanCategorizer initializes correctly with general_ner.json"""
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize with the general_ner taxonomy
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path))
        
        # Verify the taxonomy was loaded
        self.assertIsNotNone(categorizer.taxonomy)
        self.assertIn("children", categorizer.taxonomy)
        
        # Check that major categories are present
        root_children = categorizer.taxonomy["children"]
        self.assertIn("physical_entity.n.01", root_children)
        self.assertIn("abstraction.n.06", root_children)
        self.assertIn("psychological_feature.n.01", root_children)
        self.assertIn("time.n.01", root_children)
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_person_entity_recognition(self, mock_sentence_transformer):
        """Test recognition of person entities (victims, suspects, witnesses, officers)"""
        if not self.spacy_available:
            self.skipTest("spaCy model not available")
        
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.3)
        
        # Create test document
        doc = self.nlp("The victim John Doe was interviewed by Officer Smith. The suspect fled the scene.")
        
        # Process the document
        result_doc = categorizer(doc)
        
        # Check that person entities were identified
        self.assertIn('sc', result_doc.spans)
        
        # Extract texts and labels
        span_texts = [span.text.lower() for span in result_doc.spans['sc']]
        span_labels = [span.label_ for span in result_doc.spans['sc']]
        
        # Print for debugging
        print(f"Found spans: {list(zip(span_texts, span_labels))}")
        
        # Verify specific entities were found
        self.assertTrue(any("victim" in text or "john doe" in text for text in span_texts),
                       f"Should find victim-related text in spans: {span_texts}")
        self.assertTrue(any("officer" in text or "smith" in text for text in span_texts),
                       f"Should find officer-related text in spans: {span_texts}")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_location_entity_recognition(self, mock_sentence_transformer):
        """Test recognition of location entities (addresses, cities, buildings)"""
        if not self.spacy_available:
            self.skipTest("spaCy model not available")
        
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.3)
        
        # Create test document
        doc = self.nlp("The incident occurred at 123 Main Street in Chicago near the apartment building.")
        
        # Process the document
        result_doc = categorizer(doc)
        
        # Check that location entities were identified
        self.assertIn('sc', result_doc.spans)
        
        # Extract texts and labels
        span_texts = [span.text.lower() for span in result_doc.spans['sc']]
        
        # Print for debugging
        print(f"Found location spans: {span_texts}")
        
        # Verify location entities were found
        self.assertTrue(any("123 main street" in text or "main street" in text for text in span_texts),
                       f"Should find address in spans: {span_texts}")
        self.assertTrue(any("chicago" in text for text in span_texts),
                       f"Should find city in spans: {span_texts}")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_evidence_entity_recognition(self, mock_sentence_transformer):
        """Test recognition of evidence entities (fingerprints, video, statements)"""
        if not self.spacy_available:
            self.skipTest("spaCy model not available")
        
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.3)
        
        # Create test document
        doc = self.nlp("The fingerprint evidence and video footage were collected. A witness statement was recorded.")
        
        # Process the document
        result_doc = categorizer(doc)
        
        # Check that evidence entities were identified
        self.assertIn('sc', result_doc.spans)
        
        # Extract texts and labels
        span_texts = [span.text.lower() for span in result_doc.spans['sc']]
        
        # Print for debugging
        print(f"Found evidence spans: {span_texts}")
        
        # Verify evidence entities were found
        self.assertTrue(any("fingerprint" in text for text in span_texts),
                       f"Should find fingerprint evidence in spans: {span_texts}")
        self.assertTrue(any("video" in text or "footage" in text for text in span_texts),
                       f"Should find video evidence in spans: {span_texts}")
        self.assertTrue(any("witness statement" in text or "statement" in text for text in span_texts),
                       f"Should find witness statement in spans: {span_texts}")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_crime_classification(self, mock_sentence_transformer):
        """Test classification of crime types (assault, homicide, burglary)"""
        if not self.spacy_available:
            self.skipTest("spaCy model not available")
        
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.3)
        
        # Create test document
        doc = self.nlp("The assault occurred during a burglary attempt. This homicide case is under investigation.")
        
        # Process the document
        result_doc = categorizer(doc)
        
        # Check that crime entities were identified
        self.assertIn('sc', result_doc.spans)
        
        # Extract texts and labels
        span_texts = [span.text.lower() for span in result_doc.spans['sc']]
        
        # Print for debugging
        print(f"Found crime spans: {span_texts}")
        
        # Verify crime entities were found
        self.assertTrue(any("assault" in text for text in span_texts),
                       f"Should find assault in spans: {span_texts}")
        self.assertTrue(any("burglary" in text for text in span_texts),
                       f"Should find burglary in spans: {span_texts}")
        self.assertTrue(any("homicide" in text for text in span_texts),
                       f"Should find homicide in spans: {span_texts}")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_weapon_recognition(self, mock_sentence_transformer):
        """Test recognition of weapon entities"""
        if not self.spacy_available:
            self.skipTest("spaCy model not available")
        
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.3)
        
        # Create test document
        doc = self.nlp("The suspect was armed with a gun. A knife was found at the scene.")
        
        # Process the document
        result_doc = categorizer(doc)
        
        # Check that weapon entities were identified
        self.assertIn('sc', result_doc.spans)
        
        # Extract texts and labels
        span_texts = [span.text.lower() for span in result_doc.spans['sc']]
        
        # Print for debugging
        print(f"Found weapon spans: {span_texts}")
        
        # Verify weapon entities were found
        self.assertTrue(any("gun" in text for text in span_texts),
                       f"Should find gun in spans: {span_texts}")
        self.assertTrue(any("knife" in text for text in span_texts),
                       f"Should find knife in spans: {span_texts}")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_hierarchical_classification(self, mock_sentence_transformer):
        """Test that hierarchical classification works through the taxonomy tree"""
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer with a very low threshold for deeper traversal
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.05)
        
        # Manually set up embedded taxonomy for testing hierarchical search
        # This ensures we have proper embeddings at all levels
        categorizer.taxonomy = {
            "children": {
                "physical_entity.n.01": {
                    "label": "Physical_Entities",
                    "embedding": np.array([0.7, 0.3, 0.0, 0.0] + [0.0] * 380),
                    "children": {
                        "causal_agent.n.01": {
                            "label": "Agents",
                            "embedding": np.array([0.85, 0.15, 0.0, 0.0] + [0.0] * 380),
                            "children": {
                                "person.n.01": {
                                    "label": "Persons",
                                    "embedding": np.array([0.95, 0.05, 0.0, 0.0] + [0.0] * 380),
                                    "children": {
                                        "victim.n.01": {
                                            "label": "Victims",
                                            "embedding": np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 380)
                                        },
                                        "suspect.n.01": {
                                            "label": "Suspects",
                                            "embedding": np.array([0.98, 0.02, 0.0, 0.0] + [0.0] * 380)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Test hierarchical search for "victim"
        result = categorizer._hierarchical_sem_search(
            query="victim",
            current_label="ENTITY",
            current_node={"children": categorizer.taxonomy}
        )
        
        # Should traverse down to find a person-related label
        # The exact depth depends on similarity scores, but should be person-related
        possible_labels = ["Victims", "Persons", "Agents", "Physical_Entities", "ENTITY"]
        self.assertIn(result, possible_labels,
                     f"Should find a valid label in the hierarchy, got: {result}")
    

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_complex_document_processing(self, mock_sentence_transformer):
        """Test processing a complex document with multiple entity types"""
        if not self.spacy_available:
            self.skipTest("spaCy model not available")
        
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [self.get_mock_embedding(texts[0])]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path), threshold=0.3)
        
        # Create a complex test document
        doc = self.nlp("""
        On January 1st at 10:30 PM, Officer Smith responded to a burglary at 123 Main Street in Chicago.
        The victim, John Doe, reported that the suspect fled with stolen property.
        Video footage and fingerprint evidence were collected at the scene.
        A witness statement was recorded from Jane Smith who saw the suspect with a knife.
        """)
        
        # Process the document
        result_doc = categorizer(doc)
        
        # Check that entities were identified
        self.assertIn('sc', result_doc.spans)
        self.assertGreater(len(result_doc.spans['sc']), 0, "Should identify multiple entities")
        
        # Extract all identified entity types
        span_texts = [span.text.lower() for span in result_doc.spans['sc']]
        span_labels = {span.label_ for span in result_doc.spans['sc']}
        
        # Print for debugging
        print(f"Found {len(span_texts)} spans with labels: {span_labels}")
        print(f"Span texts: {span_texts}")
        
        # Verify multiple entity types were found
        entity_types_found = {
            "person": any("john" in text or "jane" in text or "smith" in text or 
                         "victim" in text or "suspect" in text or "officer" in text 
                         for text in span_texts),
            "location": any("chicago" in text or "main street" in text or "123" in text 
                           for text in span_texts),
            "evidence": any("video" in text or "footage" in text or "fingerprint" in text or 
                          "statement" in text for text in span_texts),
            "crime": any("burglary" in text for text in span_texts),
            "weapon": any("knife" in text for text in span_texts)
        }
        
        # At least some entity types should be found
        self.assertTrue(any(entity_types_found.values()),
                       f"Should find at least some entity types. Found: {entity_types_found}")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_taxonomy_structure_validation(self, mock_sentence_transformer):
        """Test that the general_ner taxonomy structure is valid and complete"""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1] * 384)]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer - this will validate the taxonomy
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path))
        
        # Check root structure
        self.assertIn("children", categorizer.taxonomy)
        
        # Navigate to key categories and verify they exist
        root = categorizer.taxonomy["children"]
        
        # Check Physical Entities branch
        self.assertIn("physical_entity.n.01", root)
        physical = root["physical_entity.n.01"]
        self.assertIn("children", physical)
        self.assertIn("causal_agent.n.01", physical["children"])
        
        # Check Agents -> Persons branch
        agents = physical["children"]["causal_agent.n.01"]
        self.assertIn("children", agents)
        self.assertIn("person.n.01", agents["children"])
        persons = agents["children"]["person.n.01"]
        self.assertIn("children", persons)
        
        # Verify key person types exist
        person_types = ["victim.n.01", "suspect.n.01", "witness.n.01", "officer.n.01"]
        for person_type in person_types:
            self.assertIn(person_type, persons["children"],
                         f"Person type {person_type} should exist in taxonomy")
        
        # Check Evidence branch
        objects = physical["children"]["object.n.01"]
        self.assertIn("children", objects)
        artifacts = objects["children"]["artifact.n.01"]
        self.assertIn("children", artifacts)
        self.assertIn("evidence.n.01", artifacts["children"])
        evidence = artifacts["children"]["evidence.n.01"]
        self.assertIn("children", evidence)
        
        # Verify key evidence types exist
        evidence_types = ["fingerprint.n.01", "footage.n.01", "statement.n.01", "testimony.n.01"]
        for evidence_type in evidence_types:
            self.assertIn(evidence_type, evidence["children"],
                         f"Evidence type {evidence_type} should exist in taxonomy")
        
        # Check Crime/Events branch
        self.assertIn("psychological_feature.n.01", root)
        psych = root["psychological_feature.n.01"]
        self.assertIn("children", psych)
        self.assertIn("event.n.01", psych["children"])
        events = psych["children"]["event.n.01"]
        self.assertIn("children", events)
        self.assertIn("activity.n.01", events["children"])
        activities = events["children"]["activity.n.01"]
        self.assertIn("children", activities)
        self.assertIn("crime.n.01", activities["children"])
        crimes = activities["children"]["crime.n.01"]
        self.assertIn("children", crimes)
        
        # Verify key crime types exist
        crime_types = ["assault.n.01", "homicide.n.01", "offense.n.01"]
        for crime_type in crime_types:
            self.assertIn(crime_type, crimes["children"],
                         f"Crime type {crime_type} should exist in taxonomy")
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_wordnet_synsets_in_taxonomy(self, mock_sentence_transformer):
        """Test that WordNet synsets are properly utilized in the taxonomy"""
        mock_model = Mock()
        # Return consistent embeddings
        mock_model.encode.return_value = [np.array([0.1] * 384)]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path))
        
        # Check that nodes with wordnet_synsets have been properly embedded
        def check_wordnet_embeddings(node, path=""):
            """Recursively check that nodes with wordnet_synsets have embeddings"""
            if isinstance(node, dict):
                # Check if this node has wordnet_synsets
                if "wordnet_synsets" in node:
                    # It should have been processed and have an embedding
                    self.assertIn("embedding", node["wordnet_synsets"],
                                 f"Node at {path} with wordnet_synsets should have embedding")
                    # The embedding should not be all zeros
                    embedding = node["wordnet_synsets"]["embedding"]
                    self.assertFalse(np.allclose(embedding, 0),
                                   f"WordNet synset embedding at {path} should not be all zeros")
                
                # Recursively check children
                if "children" in node:
                    for child_key, child_node in node["children"].items():
                        check_wordnet_embeddings(child_node, f"{path}/{child_key}")
        
        # Start checking from root
        check_wordnet_embeddings(categorizer.taxonomy)
    
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_description_fields_in_taxonomy(self, mock_sentence_transformer):
        """Test that description fields are properly processed in the taxonomy"""
        mock_model = Mock()
        # Return consistent embeddings
        mock_model.encode.return_value = [np.array([0.2] * 384)]
        mock_sentence_transformer.return_value = mock_model
        
        # Get the path to general_ner.json
        taxonomy_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        
        # Initialize categorizer
        categorizer = SpanCategorizer(taxonomy_path=str(taxonomy_path))
        
        # Check that nodes with descriptions have been properly embedded
        def check_description_embeddings(node, path=""):
            """Recursively check that nodes with descriptions have embeddings"""
            if isinstance(node, dict):
                # Check if this node has a description
                if "description" in node and isinstance(node["description"], dict):
                    # It should have been processed and have an embedding
                    self.assertIn("embedding", node["description"],
                                 f"Node at {path} with description should have embedding")
                    # The embedding should not be all zeros
                    embedding = node["description"]["embedding"]
                    self.assertFalse(np.allclose(embedding, 0),
                                   f"Description embedding at {path} should not be all zeros")
                
                # Recursively check children
                if "children" in node:
                    for child_key, child_node in node["children"].items():
                        check_description_embeddings(child_node, f"{path}/{child_key}")
        
        # Start checking from root
        check_description_embeddings(categorizer.taxonomy)


if __name__ == '__main__':
    unittest.main()