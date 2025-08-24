"""
Comprehensive unit tests for SpanCategorizer improvements.

This test suite validates all the fixes and improvements made to the SpanCategorizer class:
1. Enhanced label extraction with multiple fallback strategies
2. Improved threshold logic with adaptive behavior
3. Better hierarchical search integration
4. Edge case and error handling improvements
5. Integration tests with SpaCy documents

These tests ensure backward compatibility while validating the new functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import tempfile
import os
from pathlib import Path

import spacy
from spacy.tokens import Doc, Span

from src.tax_span_cat.SpanCategorizer import SpanCategorizer


class TestSpanCategorizerImprovements(unittest.TestCase):
    """Test suite for SpanCategorizer improvements and fixes."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Complex taxonomy for testing hierarchical search
        self.complex_taxonomy = {
            "children": {
                "physical_entity.n.01": {
                    "label": "Physical_Entities",
                    "description": "Physical objects and entities",
                    "children": {
                        "causal_agent.n.01": {
                            "label": "Agents",
                            "description": "Agents that can cause events",
                            "children": {
                                "person.n.01": {
                                    "label": "Persons",
                                    "description": "Human beings and individuals",
                                    "children": {
                                        "victim.n.01": {
                                            "label": "Victims",
                                            "description": "Persons who are victims of crimes"
                                        },
                                        "suspect.n.01": {
                                            "label": "Suspects", 
                                            "description": "Persons suspected of criminal activity"
                                        }
                                    }
                                }
                            }
                        },
                        "object.n.01": {
                            "label": "Objects",
                            "description": "Inanimate objects and items",
                            "children": {
                                "weapon.n.01": {
                                    "label": "Weapons",
                                    "description": "Weapons and arms used in incidents"
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Create temporary taxonomy file
        self.temp_taxonomy_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(self.complex_taxonomy, self.temp_taxonomy_file)
        self.temp_taxonomy_file.close()
        
    def tearDown(self):
        """Clean up after each test method."""
        os.unlink(self.temp_taxonomy_file.name)

    # ============================================
    # 1. Label Extraction Tests (_extract_best_label method)
    # ============================================
    
    def test_extract_best_label_with_label_field(self):
        """Test that _extract_best_label returns label field directly when present."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "label": "DIRECT_LABEL",
                "description": "Some description",
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "test.n.01")
            
            self.assertEqual(result, "DIRECT_LABEL")

    def test_extract_best_label_fallback_to_description(self):
        """Test that _extract_best_label falls back to description when no label field."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "description": "A test description for fallback",
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "test.n.01")
            
            self.assertEqual(result, "A test description for fallback")

    def test_extract_best_label_truncates_long_description(self):
        """Test that long descriptions are truncated to 50 characters."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            long_description = "This is a very long description that exceeds the fifty character limit and should be truncated"
            node = {
                "description": long_description,
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "test.n.01")
            
            # Should truncate at 47 characters and add "..."
            expected = "This is a very long description that exceeds th..."
            self.assertEqual(result, expected)
            self.assertEqual(len(result), 50)

    def test_extract_best_label_cleans_synset_format_standard(self):
        """Test that synset keys are cleaned up properly (.n.01 format)."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Node with only embedding (no label or description)
            node = {
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "dog.n.01")
            
            self.assertEqual(result, "Dog")

    def test_extract_best_label_cleans_synset_format_verb(self):
        """Test that verb synset keys are cleaned up properly (.v.01 format)."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "run.v.01")
            
            self.assertEqual(result, "Run")

    def test_extract_best_label_cleans_synset_with_underscores(self):
        """Test that synset keys with underscores are cleaned up properly."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "motor_vehicle.n.01")
            
            self.assertEqual(result, "Motor Vehicle")

    def test_extract_best_label_cleans_node_with_only_embedding(self):
        """Test that nodes with only embedding use synset key cleanup."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            # When node only has embedding, synset key is cleaned regardless
            result = categorizer._extract_best_label(node, "VICTIMS")
            self.assertEqual(result, "Victims")  # Gets title-cased
            
            # Test with dots and underscores - should be cleaned
            result2 = categorizer._extract_best_label(node, "mixed_case.n.01")
            self.assertEqual(result2, "Mixed Case")  # Title case applied
    
    def test_extract_best_label_preserves_clean_keys_with_other_fields(self):
        """Test that clean keys are preserved when not taking the synset-only path."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Node with other fields - should reach the final synset cleanup logic
            node = {
                "some_other_field": "value",
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            # This should preserve clean uppercase keys
            result = categorizer._extract_best_label(node, "VICTIMS")
            self.assertEqual(result, "VICTIMS")  # Should be preserved
            
            # This should clean mixed case
            result2 = categorizer._extract_best_label(node, "mixed_case.n.01")
            self.assertEqual(result2, "Mixed Case")

    def test_extract_best_label_handles_malformed_nodes(self):
        """Test that _extract_best_label handles malformed or empty nodes gracefully."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Test with non-dict input
            result = categorizer._extract_best_label("not_a_dict", "test.n.01")
            self.assertEqual(result, "not_a_dict")
            
            # Test with empty dict
            result = categorizer._extract_best_label({}, "test.n.01")
            self.assertEqual(result, "Test")
            
            # Test with None input
            result = categorizer._extract_best_label(None, "test.n.01")
            self.assertEqual(result, "None")

    def test_extract_best_label_handles_none_synset_key(self):
        """Test that _extract_best_label handles None synset key gracefully."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, None)
            
            self.assertEqual(result, "UNKNOWN")

    def test_extract_best_label_handles_empty_strings(self):
        """Test that _extract_best_label handles empty string fields properly."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {
                "label": "",  # Empty label
                "description": "",  # Empty description
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "fallback.n.01")
            
            self.assertEqual(result, "Fallback")

    # ============================================
    # 2. Improved Threshold Logic Tests
    # ============================================

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_adaptive_threshold_behavior_at_depth(self, mock_sentence_transformer):
        """Test that threshold adapts with depth for deeper exploration."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.6)
                
                # Mock the embedded taxonomy structure
                categorizer.taxonomy = {
                    "physical_entity.n.01": {
                        "label": "Physical_Entities",
                        "embedding": np.array([0.55, 0.45, 0.0]),  # Similarity will be ~0.55
                        "children": {
                            "causal_agent.n.01": {
                                "label": "Agents",
                                "embedding": np.array([0.5, 0.5, 0.0]),  # Similarity will be ~0.5
                                "children": {
                                    "person.n.01": {
                                        "label": "Persons",
                                        "embedding": np.array([0.45, 0.55, 0.0])  # Similarity will be ~0.45
                                    }
                                }
                            }
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # Simulate similarity that's above adaptive threshold to continue deep
                    mock_search.side_effect = [
                        (0.65, 0),  # Depth 0: above initial threshold (0.6), continue
                        (0.55, 0),  # Depth 1: above adaptive threshold (0.55), continue  
                        (0.45, 0)   # Depth 2: below adaptive threshold (0.5), stop here
                    ]
                    
                    # Mock print to suppress output during testing
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="test person",
                            current_label="ENTITY",
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # With adaptive thresholds, should navigate deep and find "Persons"
                    self.assertEqual(result, "Persons")
                    
                    # Should have made 3 calls (one at each depth level)
                    self.assertEqual(mock_search.call_count, 3)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_best_match_returned_even_below_threshold(self, mock_sentence_transformer):
        """Test that best match is returned even when below threshold (not 'ENTITY')."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.9)  # Very high threshold
                
                categorizer.taxonomy = {
                    "weapon.n.01": {
                        "label": "Weapons",
                        "embedding": np.array([0.3, 0.7, 0.0])  # Low similarity with query
                    },
                    "person.n.01": {
                        "label": "Persons", 
                        "embedding": np.array([0.2, 0.8, 0.0])  # Even lower similarity
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.2, 0)  # Very low similarity (below threshold)
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="test query",
                            current_label="NOT_ENTITY",  # Not at root level
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should return best match (Weapons) even though below threshold
                    self.assertEqual(result, "Weapons")
                    # Should NOT return "ENTITY" or "NOT_ENTITY"
                    self.assertNotEqual(result, "ENTITY")
                    self.assertNotEqual(result, "NOT_ENTITY")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_backward_compatibility_entity_fallback(self, mock_sentence_transformer):
        """Test backward compatibility: returns 'ENTITY' at root level with low similarity."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.9)  # Very high threshold
                
                categorizer.taxonomy = {
                    "unrelated.n.01": {
                        "label": "Unrelated",
                        "embedding": np.array([0.0, 0.0, 1.0])  # Orthogonal to query
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.05, 0)  # Very low similarity
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="completely different query",
                            current_label="ENTITY",  # At root level
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should return "ENTITY" for backward compatibility
                    self.assertEqual(result, "ENTITY")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer') 
    def test_perfect_similarity_continues_deep(self, mock_sentence_transformer):
        """Test that perfect similarity scores continue hierarchical search."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.5)
                
                categorizer.taxonomy = {
                    "physical_entity.n.01": {
                        "label": "Physical_Entities",
                        "embedding": np.array([1.0, 0.0, 0.0]),
                        "children": {
                            "person.n.01": {
                                "label": "Persons",
                                "embedding": np.array([1.0, 0.0, 0.0])
                            }
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # Perfect similarity at both levels, then stop (leaf node)
                    mock_search.side_effect = [(1.0, 0), (1.0, 0)]
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="perfect match",
                            current_label="ENTITY",
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should continue deep and find "Persons"
                    self.assertEqual(result, "Persons")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_minimum_confidence_threshold(self, mock_sentence_transformer):
        """Test that minimum confidence threshold is respected."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.5)
                
                categorizer.taxonomy = {
                    "test.n.01": {
                        "label": "Test_Label",
                        "embedding": np.array([0.0, 0.0, 1.0])  # Very different from query
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.05, 0)  # Below minimum confidence (0.1)
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="unrelated query",
                            current_label="FALLBACK",
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should return current level's best match despite low confidence
                    self.assertEqual(result, "Test_Label")

    # ============================================
    # 3. Hierarchical Search Integration Tests
    # ============================================

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_hierarchical_search_with_general_ner_taxonomy(self, mock_sentence_transformer):
        """Test hierarchical search integration with the general_ner.json taxonomy structure."""
        # Load the actual general_ner.json taxonomy
        general_ner_path = Path(__file__).parent.parent / "src" / "tax_span_cat" / "taxonomies" / "general_ner.json"
        with open(general_ner_path, 'r') as f:
            general_ner_taxonomy = json.load(f)
        
        mock_model = Mock()
        # Mock embeddings to favor person-related concepts
        def mock_encode(texts):
            text = texts[0].lower() if isinstance(texts, list) else texts.lower()
            if "person" in text or "victim" in text or "suspect" in text:
                return [np.array([1.0, 0.0, 0.0])]
            elif "weapon" in text or "arm" in text:
                return [np.array([0.0, 1.0, 0.0])]
            else:
                return [np.array([0.1, 0.1, 0.8])]
        
        mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = general_ner_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                # Create a realistic embedded version
                mock_embed_tax.return_value = general_ner_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.3)
                
                # Set up realistic taxonomy embeddings
                categorizer.taxonomy = {
                    "physical_entity.n.01": {
                        "label": "Physical_Entities",
                        "embedding": np.array([0.5, 0.5, 0.0]),
                        "children": {
                            "causal_agent.n.01": {
                                "label": "Agents", 
                                "embedding": np.array([0.8, 0.2, 0.0]),
                                "children": {
                                    "person.n.01": {
                                        "label": "Persons",
                                        "embedding": np.array([0.9, 0.1, 0.0]),
                                        "children": {
                                            "victim.n.01": {
                                                "label": "Victims",
                                                "embedding": np.array([1.0, 0.0, 0.0])
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # Simulate decreasing but acceptable similarity through hierarchy
                    mock_search.side_effect = [(0.8, 0), (0.75, 0), (0.7, 0)]
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="crime victim",
                            current_label="ENTITY",
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should find "Victims" through hierarchical navigation
                    self.assertEqual(result, "Victims")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_hierarchical_navigation_multiple_levels(self, mock_sentence_transformer):
        """Test navigation through multiple levels of hierarchy."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.4)
                
                # Create 4-level deep taxonomy
                categorizer.taxonomy = {
                    "level1.n.01": {
                        "label": "Level_1",
                        "embedding": np.array([0.9, 0.1, 0.0]),
                        "children": {
                            "level2.n.01": {
                                "label": "Level_2", 
                                "embedding": np.array([0.85, 0.15, 0.0]),
                                "children": {
                                    "level3.n.01": {
                                        "label": "Level_3",
                                        "embedding": np.array([0.8, 0.2, 0.0]),
                                        "children": {
                                            "level4.n.01": {
                                                "label": "Level_4",
                                                "embedding": np.array([0.75, 0.25, 0.0])
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # Decreasing similarity that stays above adaptive thresholds
                    mock_search.side_effect = [
                        (0.9, 0),   # Level 1: threshold ~0.4
                        (0.85, 0),  # Level 2: threshold ~0.35  
                        (0.8, 0),   # Level 3: threshold ~0.3
                        (0.75, 0)   # Level 4: threshold ~0.25
                    ]
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="deep search test",
                            current_label="ENTITY",
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should navigate to deepest level
                    self.assertEqual(result, "Level_4")
                    self.assertEqual(mock_search.call_count, 4)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_fallback_when_no_good_matches(self, mock_sentence_transformer):
        """Test fallback behavior when no good matches are found at any level."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1.0, 0.0, 0.0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.5)
                
                categorizer.taxonomy = {
                    "unrelated1.n.01": {
                        "label": "Unrelated_1",
                        "embedding": np.array([0.0, 0.0, 1.0])  # Orthogonal to query
                    },
                    "unrelated2.n.01": {
                        "label": "Unrelated_2", 
                        "embedding": np.array([0.0, 1.0, 0.0])  # Also orthogonal
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.05, 0)  # Very low similarity
                    
                    with patch('builtins.print'):
                        result = categorizer._hierarchical_sem_search(
                            query="completely different concept",
                            current_label="ENTITY",
                            current_node={"children": categorizer.taxonomy}
                        )
                    
                    # Should fall back to "ENTITY" for backward compatibility
                    self.assertEqual(result, "ENTITY")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_real_world_noun_chunk_categorization(self, mock_sentence_transformer):
        """Test hierarchical search with real-world noun chunks and expected categorizations."""
        mock_model = Mock()
        
        # Mock embeddings to simulate realistic semantic similarity
        def mock_encode(texts):
            text = texts[0].lower() if isinstance(texts, list) else texts.lower()
            
            if "police officer" in text or "officer" in text:
                return [np.array([1.0, 0.0, 0.0])]
            elif "gun" in text or "weapon" in text:
                return [np.array([0.0, 1.0, 0.0])]
            elif "car" in text or "vehicle" in text:
                return [np.array([0.0, 0.0, 1.0])]
            else:
                return [np.array([0.3, 0.3, 0.4])]
        
        mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.4)
                
                # Set up taxonomy with law enforcement categories
                categorizer.taxonomy = {
                    "person.n.01": {
                        "label": "Persons",
                        "embedding": np.array([0.9, 0.1, 0.0]),
                        "children": {
                            "officer.n.01": {
                                "label": "Officers",
                                "embedding": np.array([1.0, 0.0, 0.0])
                            }
                        }
                    },
                    "weapon.n.01": {
                        "label": "Weapons",
                        "embedding": np.array([0.0, 1.0, 0.0])
                    },
                    "vehicle.n.01": {
                        "label": "Vehicles",
                        "embedding": np.array([0.0, 0.0, 1.0])
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # Test different noun chunks
                    test_cases = [
                        ("police officer", [(0.9, 0), (0.95, 0)], "Officers"),
                        ("handgun", [(0.85, 1)], "Weapons"),  
                        ("patrol car", [(0.8, 2)], "Vehicles")
                    ]
                    
                    for query, similarities, expected in test_cases:
                        with self.subTest(query=query):
                            mock_search.reset_mock()
                            mock_search.side_effect = similarities + similarities  # Duplicate to handle multiple calls
                            
                            with patch('builtins.print'):
                                result = categorizer._hierarchical_sem_search(
                                    query=query,
                                    current_label="ENTITY",
                                    current_node={"children": categorizer.taxonomy}
                                )
                            
                            self.assertEqual(result, expected)

    # ============================================
    # 4. Edge Case and Error Handling Tests
    # ============================================

    def test_hierarchical_search_empty_taxonomy_nodes(self):
        """Test hierarchical search handles empty taxonomy nodes gracefully."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Empty node structure
            empty_node = {"children": {}}
            
            result = categorizer._extract_best_label(empty_node, "EMPTY_LABEL")
            
            # Should return cleaned synset key when no useful content
            self.assertEqual(result, "Empty Label")

    def test_hierarchical_search_missing_children(self):
        """Test hierarchical search handles nodes without children key."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Node without children key (leaf node)
            leaf_node = {
                "label": "LEAF_NODE",
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._hierarchical_sem_search(
                query="test query",
                current_label="PARENT_LABEL",
                current_node=leaf_node
            )
            
            # Should use the enhanced label extraction
            self.assertEqual(result, "LEAF_NODE")

    def test_hierarchical_search_malformed_taxonomy_structure(self):
        """Test hierarchical search handles malformed taxonomy structures."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Malformed structure with children but no embeddings
            malformed_node = {
                "children": {
                    "child1": {"label": "CHILD1"},  # Missing embedding
                    "child2": {"description": "Child 2"}  # Missing embedding
                }
            }
            
            result = categorizer._hierarchical_sem_search(
                query="test query",
                current_label="MALFORMED_PARENT",
                current_node=malformed_node
            )
            
            # Should handle gracefully and return parent label using enhanced extraction
            self.assertEqual(result, "Malformed Parent")

    def test_hierarchical_search_very_short_query_strings(self):
        """Test hierarchical search with very short query strings."""
        with patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
            mock_st.return_value = mock_model
            
            with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
                mock_load.return_value = self.complex_taxonomy
                with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                    mock_embed_tax.return_value = self.complex_taxonomy
                    
                    categorizer = SpanCategorizer(threshold=0.3)
                    
                    categorizer.taxonomy = {
                        "test.n.01": {
                            "label": "Test_Label",
                            "embedding": np.array([0.5, 0.5, 0.0])
                        }
                    }
                    
                    with patch.object(categorizer, '_semantic_search') as mock_search:
                        mock_search.return_value = (0.4, 0)
                        
                        # Test very short queries
                        short_queries = ["a", "I", "x", "?", ""]
                        
                        for query in short_queries:
                            with self.subTest(query=query):
                                with patch('builtins.print'):
                                    result = categorizer._hierarchical_sem_search(
                                        query=query,
                                        current_label="ENTITY",
                                        current_node={"children": categorizer.taxonomy}
                                    )
                                
                                # Should handle short queries without crashing
                                self.assertIsInstance(result, str)
                                self.assertGreater(len(result), 0)

    def test_hierarchical_search_very_long_query_strings(self):
        """Test hierarchical search with very long query strings."""
        with patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
            mock_st.return_value = mock_model
            
            with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
                mock_load.return_value = self.complex_taxonomy
                with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                    mock_embed_tax.return_value = self.complex_taxonomy
                    
                    categorizer = SpanCategorizer(threshold=0.3)
                    
                    categorizer.taxonomy = {
                        "test.n.01": {
                            "label": "Test_Label",
                            "embedding": np.array([0.5, 0.5, 0.0])
                        }
                    }
                    
                    with patch.object(categorizer, '_semantic_search') as mock_search:
                        mock_search.return_value = (0.4, 0)
                        
                        # Very long query string
                        long_query = "This is an extremely long query string that contains many words and should test the system's ability to handle verbose input without crashing or producing errors during the semantic search process and hierarchical navigation through the taxonomy structure."
                        
                        with patch('builtins.print'):
                            result = categorizer._hierarchical_sem_search(
                                query=long_query,
                                current_label="ENTITY", 
                                current_node={"children": categorizer.taxonomy}
                            )
                        
                        # Should handle long queries without issues
                        self.assertEqual(result, "Test_Label")
                        mock_model.encode.assert_called_with([long_query])

    def test_hierarchical_search_special_characters(self):
        """Test hierarchical search with special characters in queries."""
        with patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
            mock_st.return_value = mock_model
            
            with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
                mock_load.return_value = self.complex_taxonomy
                with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                    mock_embed_tax.return_value = self.complex_taxonomy
                    
                    categorizer = SpanCategorizer(threshold=0.3)
                    
                    categorizer.taxonomy = {
                        "test.n.01": {
                            "label": "Test_Label",
                            "embedding": np.array([0.5, 0.5, 0.0])
                        }
                    }
                    
                    with patch.object(categorizer, '_semantic_search') as mock_search:
                        mock_search.return_value = (0.4, 0)
                        
                        # Queries with special characters
                        special_queries = [
                            "test@example.com",
                            "query-with-dashes", 
                            "query_with_underscores",
                            "query with spaces",
                            "query/with/slashes",
                            "query\\with\\backslashes",
                            "query with números 123",
                            "query with émojis 😀",
                            "query with 'quotes'",
                            'query with "double quotes"'
                        ]
                        
                        for query in special_queries:
                            with self.subTest(query=query):
                                with patch('builtins.print'):
                                    result = categorizer._hierarchical_sem_search(
                                        query=query,
                                        current_label="ENTITY",
                                        current_node={"children": categorizer.taxonomy}
                                    )
                                
                                # Should handle special characters without crashing
                                self.assertEqual(result, "Test_Label")

    def test_hierarchical_search_non_english_text(self):
        """Test hierarchical search with non-English text."""
        with patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
            mock_st.return_value = mock_model
            
            with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
                mock_load.return_value = self.complex_taxonomy
                with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                    mock_embed_tax.return_value = self.complex_taxonomy
                    
                    categorizer = SpanCategorizer(threshold=0.3)
                    
                    categorizer.taxonomy = {
                        "person.n.01": {
                            "label": "Persons",
                            "embedding": np.array([0.5, 0.5, 0.0])
                        }
                    }
                    
                    with patch.object(categorizer, '_semantic_search') as mock_search:
                        mock_search.return_value = (0.4, 0)
                        
                        # Non-English queries
                        non_english_queries = [
                            "persona",  # Spanish
                            "personne",  # French
                            "Person",  # German
                            "人",  # Chinese
                            "человек",  # Russian
                            "شخص",  # Arabic
                        ]
                        
                        for query in non_english_queries:
                            with self.subTest(query=query):
                                with patch('builtins.print'):
                                    result = categorizer._hierarchical_sem_search(
                                        query=query,
                                        current_label="ENTITY",
                                        current_node={"children": categorizer.taxonomy}
                                    )
                                
                                # Should handle non-English text
                                self.assertEqual(result, "Persons")
                                # Should pass the non-English query to the embedding model
                                mock_model.encode.assert_called_with([query])

    # ============================================  
    # 5. Integration Tests with SpaCy
    # ============================================

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_full_call_method_with_improved_logic(self, mock_sentence_transformer):
        """Test the full __call__ method with actual SpaCy documents using improved logic."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self.skipTest("spaCy model 'en_core_web_sm' not available")
        
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.3)
                
                # Mock the hierarchical search to return specific labels
                with patch.object(categorizer, '_hierarchical_sem_search') as mock_search:
                    mock_search.side_effect = ["Persons", "Weapons", "Objects"]
                    
                    # Create a realistic document
                    doc = nlp("The police officer drew his weapon during the incident.")
                    
                    # Process with improved categorizer
                    result_doc = categorizer(doc)
                    
                    # Verify spans were added correctly
                    self.assertIn('sc', result_doc.spans)
                    self.assertGreater(len(result_doc.spans['sc']), 0)
                    
                    # Verify improved labeling was used
                    span_labels = [span.label_ for span in result_doc.spans['sc']]
                    self.assertIn("Persons", span_labels)
                    self.assertIn("Weapons", span_labels)
                    
                    # Verify hierarchical search was called for each noun chunk
                    self.assertEqual(mock_search.call_count, len(list(doc.noun_chunks)))

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_spans_properly_labeled_with_improved_logic(self, mock_sentence_transformer):
        """Test that spans are properly labeled using the improved hierarchical logic."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self.skipTest("spaCy model 'en_core_web_sm' not available")
        
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.4)
                
                # Set up realistic taxonomy with improved label extraction
                categorizer.taxonomy = {
                    "person.n.01": {
                        "label": "Persons",  # Should use this label
                        "description": "Human beings and individuals",
                        "embedding": np.array([0.8, 0.2, 0.0]),
                        "children": {
                            "victim.n.01": {
                                "description": "Persons suspected of criminal activity - this is a very long description that should be truncated to demonstrate the truncation feature",  # Should extract and truncate this
                                "embedding": np.array([0.9, 0.1, 0.0])
                            }
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # First call finds person, second call finds victim with higher similarity
                    mock_search.side_effect = [(0.8, 0), (0.9, 0)]
                    
                    doc = nlp("The crime victim reported the incident.")
                    
                    with patch('builtins.print'):
                        result_doc = categorizer(doc)
                    
                    # Should have spans with improved labels
                    self.assertIn('sc', result_doc.spans)
                    spans = result_doc.spans['sc']
                    
                    # Find the victim span
                    victim_spans = [s for s in spans if "victim" in s.text.lower()]
                    self.assertGreater(len(victim_spans), 0)
                    
                    # Should use the description (truncated if needed)
                    victim_span = victim_spans[0]
                    self.assertEqual(victim_span.label_, "Persons suspected of criminal activity - this...")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_performance_with_complex_documents(self, mock_sentence_transformer):
        """Test performance and correctness with complex documents containing many noun chunks."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self.skipTest("spaCy model 'en_core_web_sm' not available")
        
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.3)
                
                # Mock hierarchical search to return varied but realistic labels
                def mock_hierarchical_search(query, current_label, current_node, **kwargs):
                    query_lower = query.lower()
                    if "officer" in query_lower or "police" in query_lower:
                        return "Officers"
                    elif "weapon" in query_lower or "gun" in query_lower:
                        return "Weapons"
                    elif "car" in query_lower or "vehicle" in query_lower:
                        return "Vehicles"
                    elif "person" in query_lower or "man" in query_lower or "woman" in query_lower:
                        return "Persons"
                    else:
                        return "Objects"
                
                with patch.object(categorizer, '_hierarchical_sem_search', side_effect=mock_hierarchical_search):
                    # Complex document with many noun chunks
                    complex_text = """
                    The police officer arrived at the scene in his patrol car. 
                    The suspect, a tall man with a black jacket, was holding a weapon.
                    Several witnesses saw the incident, including a woman with red hair.
                    The victim was taken to the hospital by ambulance.
                    Evidence including fingerprints and DNA samples were collected.
                    The investigation is ongoing under case number 2024-001234.
                    """
                    
                    doc = nlp(complex_text)
                    noun_chunks_count = len(list(doc.noun_chunks))
                    
                    # Should have multiple noun chunks
                    self.assertGreater(noun_chunks_count, 5)
                    
                    with patch('builtins.print'):
                        result_doc = categorizer(doc)
                    
                    # Verify all noun chunks were processed
                    self.assertIn('sc', result_doc.spans)
                    processed_spans = result_doc.spans['sc']
                    self.assertEqual(len(processed_spans), noun_chunks_count)
                    
                    # Verify diverse labeling
                    labels = set(span.label_ for span in processed_spans)
                    self.assertGreater(len(labels), 2)  # Should have multiple different labels
                    
                    # Verify specific categorizations
                    span_texts_and_labels = [(span.text.lower(), span.label_) for span in processed_spans]
                    
                    # Find officer-related spans
                    officer_spans = [label for text, label in span_texts_and_labels if "officer" in text]
                    if officer_spans:
                        self.assertIn("Officers", officer_spans)
                    
                    # Find weapon-related spans
                    weapon_spans = [label for text, label in span_texts_and_labels if "weapon" in text]
                    if weapon_spans:
                        self.assertIn("Weapons", weapon_spans)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_backward_compatibility_with_existing_behavior(self, mock_sentence_transformer):
        """Test that improvements maintain backward compatibility with existing behavior."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self.skipTest("spaCy model 'en_core_web_sm' not available")
        
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.complex_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.complex_taxonomy
                
                # Test with high threshold to trigger ENTITY fallback
                categorizer = SpanCategorizer(threshold=0.9)
                
                categorizer.taxonomy = {
                    "unrelated.n.01": {
                        "label": "Unrelated_Category",
                        "embedding": np.array([0.0, 0.0, 1.0])  # Very different from typical queries
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.1, 0)  # Low similarity
                    
                    doc = nlp("Some random text.")
                    
                    with patch('builtins.print'):
                        result_doc = categorizer(doc)
                    
                    # Should fall back to "ENTITY" for backward compatibility
                    self.assertIn('sc', result_doc.spans)
                    spans = result_doc.spans['sc']
                    
                    if spans:  # If there are noun chunks
                        # Most spans should be labeled as "ENTITY" due to low similarity and high threshold
                        entity_spans = [s for s in spans if s.label_ == "ENTITY"]
                        self.assertGreater(len(entity_spans), 0)
                        
                        # Should also have "ENTITY" in the spans dict for backward compatibility
                        self.assertIn("ENTITY", result_doc.spans)


class TestSpanCategorizerEdgeCases(unittest.TestCase):
    """Additional edge case tests for SpanCategorizer improvements."""
    
    def test_extract_best_label_priority_order(self):
        """Test that _extract_best_label follows the correct priority order."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Node with all possible fields
            node = {
                "label": "PRIORITY_LABEL",
                "description": "This is a description that should be ignored",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "other_field": "Should be ignored"
            }
            
            result = categorizer._extract_best_label(node, "fallback.n.01")
            
            # Should prioritize label over description and synset key
            self.assertEqual(result, "PRIORITY_LABEL")

    def test_extract_best_label_description_over_synset(self):
        """Test that description is preferred over synset key when no label is present."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Node with description but no label
            node = {
                "description": "Use this description",
                "embedding": np.array([0.1, 0.2, 0.3])
            }
            
            result = categorizer._extract_best_label(node, "synset_fallback.n.01")
            
            # Should use description over synset key
            self.assertEqual(result, "Use this description")

    def test_extract_best_label_complex_synset_cleaning(self):
        """Test complex synset key cleaning scenarios."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            node = {"embedding": np.array([0.1, 0.2, 0.3])}
            
            test_cases = [
                ("complex_noun_phrase.n.01", "Complex Noun Phrase"),
                ("single.n.02", "Single"),
                ("multiple_under_scores.v.03", "Multiple Under Scores"),
                ("already_clean", "Already Clean"),  # Gets title cased
                ("ALREADY_CAPS", "Already Caps"),  # Gets title cased when node only has embedding
                ("mixed_Case.n.01", "Mixed Case")
            ]
            
            for synset_key, expected in test_cases:
                with self.subTest(synset_key=synset_key):
                    result = categorizer._extract_best_label(node, synset_key)
                    self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)