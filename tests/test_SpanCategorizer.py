import unittest
from unittest.mock import Mock, patch, mock_open
import numpy as np
import json
import tempfile
import os
from pathlib import Path

import spacy
from spacy.tokens import Doc, Span

from src.tax_span_cat.SpanCategorizer import SpanCategorizer


class TestSpanCategorizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_taxonomy = {
            "causal_agent.n.01": {
                "label": "Animals",
                "description": "Living creatures",
                "children": {
                    "dog.n.01": {
                        "label": "DOG",
                        "description": "Canine animals"
                    },
                    "cat.n.01": {
                        "label": "CAT",
                        "description": "Feline animals" 
                    }
                }
            },
            "physical_entity.n.01": {
                "label": "Objects",
                "description": "Inanimate things",
                "children": {
                    "tool.n.01": {
                        "label": "TOOL",
                        "description": "Instruments for work"
                    }
                }
            }
        }
        
        # Create temporary taxonomy file
        self.temp_taxonomy_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(self.sample_taxonomy, self.temp_taxonomy_file)
        self.temp_taxonomy_file.close()
        
    def tearDown(self):
        """Clean up after each test method."""
        os.unlink(self.temp_taxonomy_file.name)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_init_default_parameters(self, mock_sentence_transformer):
        """Test SpanCategorizer initialization with default parameters."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                self.assertEqual(categorizer.threshold, 0.5)
                mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_init_custom_parameters(self, mock_sentence_transformer):
        """Test SpanCategorizer initialization with custom parameters."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer(
                    embedding_model="custom-model",
                    taxonomy_path=self.temp_taxonomy_file.name,
                    threshold=0.7
                )
                
                self.assertEqual(categorizer.threshold, 0.7)
                mock_sentence_transformer.assert_called_once_with("custom-model")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    @patch('src.tax_span_cat.SpanCategorizer.spacy.load')
    def test_init_embedding_model_fallback(self, mock_spacy_load, mock_sentence_transformer):
        """Test fallback to SpaCy model when SentenceTransformer fails."""
        mock_sentence_transformer.side_effect = Exception("Model not found")
        mock_spacy_model = Mock()
        mock_spacy_load.return_value = mock_spacy_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                mock_spacy_load.assert_called_once_with("en_core_web_lg")
                self.assertEqual(categorizer.embedding_model, mock_spacy_model)

    def test_load_taxonomy_from_path_success(self):
        """Test successful taxonomy loading from file."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            result = categorizer._load_taxonomy_from_path(self.temp_taxonomy_file.name)
            
            self.assertEqual(result, self.sample_taxonomy)

    def test_load_taxonomy_from_path_file_not_found(self):
        """Test taxonomy loading with non-existent file falls back to default."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.default_taxonomy_path = self.temp_taxonomy_file.name
            
            # First call will fail, second will succeed with default
            with patch('builtins.open', side_effect=[FileNotFoundError, open(self.temp_taxonomy_file.name)]):
                result = categorizer._load_taxonomy_from_path("nonexistent.json")
                
                self.assertEqual(result, self.sample_taxonomy)

    def test_load_taxonomy_from_path_max_iterations(self):
        """Test that FileNotFoundError is raised after max iterations."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.default_taxonomy_path = "nonexistent.json"
            
            with self.assertRaises(FileNotFoundError):
                categorizer._load_taxonomy_from_path("nonexistent.json", iters=2)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_embed_sentence_transformer(self, mock_sentence_transformer):
        """Test embedding with SentenceTransformer model."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                result = categorizer._embed("test text")
                
                # The result should be normalized, so we need to check the shape
                self.assertEqual(result.shape, (3,))
                mock_model.encode.assert_called_once_with(["test text"])

    def test_embed_taxonomy_leaf_node(self):
        """Test embedding of leaf node (string)."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            result = categorizer._embed_taxonomy("test description")
            
            expected = {"embedding": np.array([0.1, 0.2, 0.3])}
            self.assertEqual(list(result.keys()), list(expected.keys()))
            np.testing.assert_array_equal(result["embedding"], expected["embedding"])

    def test_embed_taxonomy_internal_node(self):
        """Test embedding of internal node (dictionary)."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer._embed = Mock(side_effect=[
                np.array([0.1, 0.2, 0.3]),  # First child
                np.array([0.4, 0.5, 0.6])   # Second child
            ])
            
            test_node = {
                "child1": "description1",
                "child2": "description2"
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            expected_centroid = np.mean([
                np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, 0.6])
            ], axis=0)
            
            self.assertIn("embedding", result)
            # Check that the result embedding is a numpy array or scalar
            self.assertTrue(isinstance(result["embedding"], (np.ndarray, np.floating, float)))

    def test_semantic_search(self):
        """Test semantic search functionality."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            query_vect = np.array([1, 0, 0])
            corpus_vects = [
                np.array([0.8, 0.6, 0]),  # High similarity
                np.array([0, 1, 0]),      # Medium similarity  
                np.array([-1, 0, 0])      # Low similarity
            ]
            
            similarity, index = categorizer._semantic_search(query_vect, corpus_vects)
            
            self.assertEqual(index, 0)  # First vector should be most similar
            self.assertIsInstance(similarity, (float, np.float64))
            self.assertGreater(similarity, 0)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_hierarchical_sem_search_above_threshold(self, mock_sentence_transformer):
        """Test hierarchical search when similarity is above threshold."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1, 0, 0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.3)
                
                # Mock the embedded taxonomy structure
                categorizer.taxonomy = {
                    "causal_agent.n.01": {
                        "label": "Animals",
                        "embedding": np.array([0.9, 0.1, 0]),
                        "children": {
                            "dog.n.01": {
                                "label": "DOG",
                                "embedding": np.array([0.8, 0.2, 0])
                            }
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # First call returns high similarity, second call should also have high similarity
                    mock_search.side_effect = [(0.8, 0), (0.2, 0)]  # First high, then low to stop recursion
                    
                    result = categorizer._hierarchical_sem_search(
                        query="dog",
                        current_label="ENTITY", 
                        current_node={"children": categorizer.taxonomy}
                    )
                    
                    # Should return Animals since the second call has low similarity
                    self.assertEqual(result, "Animals")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_hierarchical_sem_search_below_threshold(self, mock_sentence_transformer):
        """Test hierarchical search when similarity is below threshold."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1, 0, 0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.9)
                
                categorizer.taxonomy = {
                    "causal_agent.n.01": {
                        "label": "Animals", 
                        "embedding": np.array([0.1, 0.9, 0])
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.2, 0)  # Low similarity
                    
                    result = categorizer._hierarchical_sem_search(
                        query="dog",
                        current_label="ENTITY",
                        current_node={"children": categorizer.taxonomy}
                    )
                    
                    self.assertEqual(result, "ENTITY")

    def test_hierarchical_sem_search_missing_children(self):
        """Test hierarchical search handles leaf nodes correctly."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            # Leaf node (no children, should return the label if present)
            leaf_node = {"label": "LEAF_LABEL", "embedding": np.array([1, 0, 0])}
            
            result = categorizer._hierarchical_sem_search(
                query="test",
                current_label="TEST",
                current_node=leaf_node
            )
            
            # Should return the leaf label
            self.assertEqual(result, "LEAF_LABEL")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_call_method_with_doc(self, mock_sentence_transformer):
        """Test the main __call__ method with a SpaCy document."""
        
        # Create a mock document
        mock_doc = Mock()
        mock_copied_doc = Mock()
        mock_doc.copy.return_value = mock_copied_doc
        
        # Mock noun chunks
        mock_chunk = Mock()
        mock_chunk.text = "quick brown fox"
        mock_chunk.start = 1
        mock_chunk.end = 4
        mock_doc.noun_chunks = [mock_chunk]
        
        # Set up spans, ents, and set_ents mock
        mock_copied_doc.spans = {}
        mock_copied_doc.ents = []  # Make ents iterable (empty list initially)
        mock_copied_doc.set_ents = Mock()
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1, 0, 0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                # Mock the hierarchical search to return a label
                with patch.object(categorizer, '_hierarchical_sem_search') as mock_search:
                    mock_search.return_value = "Animals"
                    
                    # Mock Span creation
                    with patch('src.tax_span_cat.SpanCategorizer.Span') as mock_span_class:
                        mock_span = Mock()
                        mock_span_class.return_value = mock_span
                        
                        result = categorizer(mock_doc)
                        
                        self.assertEqual(result, mock_copied_doc)
                        mock_search.assert_called_once()

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_threshold_property(self, mock_sentence_transformer):
        """Test that threshold property is set correctly."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.75)
                self.assertEqual(categorizer.threshold, 0.75)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_default_taxonomy_path_attribute(self, mock_sentence_transformer):
        """Test that default taxonomy path is set correctly."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                self.assertTrue(hasattr(categorizer, 'default_taxonomy_path'))


    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_default_embedding_model_initialization(self, mock_sentence_transformer):
        """Test that SpanCategorizer uses default embedding model when none specified."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
                self.assertEqual(categorizer.embedding_model, mock_model)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_default_threshold_initialization(self, mock_sentence_transformer):
        """Test that SpanCategorizer uses default threshold when none specified."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                self.assertEqual(categorizer.threshold, 0.5)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_default_taxonomy_path_initialization(self, mock_sentence_transformer):
        """Test that SpanCategorizer uses default taxonomy path when none specified."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                mock_load.assert_called_once_with(categorizer.default_taxonomy_path)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_no_parameters_initialization(self, mock_sentence_transformer):
        """Test SpanCategorizer initialization with no parameters uses all defaults."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                self.assertEqual(categorizer.threshold, 0.5)
                self.assertEqual(categorizer.embedding_model, mock_model)
                mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
                mock_load.assert_called_once_with(categorizer.default_taxonomy_path)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_call_with_empty_document(self, mock_sentence_transformer):
        """Test SpanCategorizer behavior with document containing no noun chunks."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                mock_doc = Mock()
                mock_copied_doc = Mock()
                mock_doc.copy.return_value = mock_copied_doc
                mock_doc.noun_chunks = []
                mock_copied_doc.spans = {}
                
                result = categorizer(mock_doc)
                
                self.assertEqual(result, mock_copied_doc)
                self.assertEqual(len(mock_copied_doc.spans), 0)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_call_with_minimal_document(self, mock_sentence_transformer):
        """Test SpanCategorizer behavior with minimal document structure."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1, 0, 0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                mock_doc = Mock()
                mock_copied_doc = Mock()
                mock_doc.copy.return_value = mock_copied_doc
                
                mock_chunk = Mock()
                mock_chunk.text = "test"
                mock_chunk.start = 0
                mock_chunk.end = 1
                mock_doc.noun_chunks = [mock_chunk]
                
                mock_copied_doc.spans = {}
                mock_copied_doc.ents = []  # Make ents iterable (empty list initially)
                mock_copied_doc.set_ents = Mock()
                
                with patch.object(categorizer, '_hierarchical_sem_search') as mock_search:
                    mock_search.return_value = "ENTITY"
                    
                    with patch('src.tax_span_cat.SpanCategorizer.Span') as mock_span_class:
                        mock_span = Mock()
                        mock_span_class.return_value = mock_span
                        
                        result = categorizer(mock_doc)
                        
                        self.assertEqual(result, mock_copied_doc)
                        mock_search.assert_called_once_with(
                            query="test",
                            current_label="ENTITY", 
                            current_node={"children": categorizer.taxonomy}
                        )

    @patch('src.tax_span_cat.SpanCategorizer.spacy.load')
    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_embedding_model_fallback_behavior(self, mock_sentence_transformer, mock_spacy_load):
        """Test default fallback to SpaCy model when SentenceTransformer initialization fails."""
        mock_sentence_transformer.side_effect = Exception("SentenceTransformer failed")
        mock_spacy_model = Mock()
        mock_spacy_load.return_value = mock_spacy_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                mock_spacy_load.assert_called_once_with("en_core_web_lg")
                self.assertEqual(categorizer.embedding_model, mock_spacy_model)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_default_entity_label_below_threshold(self, mock_sentence_transformer):
        """Test that 'ENTITY' label is returned when similarity is below threshold."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1, 0, 0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer(threshold=0.9)
                
                categorizer.taxonomy = {
                    "children": {
                        "Animals": {
                            "embedding": np.array([0.1, 0.9, 0]),
                            "children": {}
                        }
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.1, 0)
                    
                    result = categorizer._hierarchical_sem_search(
                        query="unrelated text",
                        current_label="ENTITY",
                        current_node=categorizer.taxonomy
                    )
                    
                    self.assertEqual(result, "ENTITY")

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_hierarchical_search_filters_embedding_key(self, mock_sentence_transformer):
        """Test that the hierarchical search properly filters out 'embedding' keys from children."""
        mock_model = Mock()
        mock_model.encode.return_value = [np.array([1, 0, 0])]
        mock_sentence_transformer.return_value = mock_model
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer()
                
                # Create a taxonomy structure that mimics what _embed_taxonomy produces
                # (with 'embedding' keys mixed in with actual children)
                categorizer.taxonomy = {
                    "children": {
                        "Animals": {
                            "embedding": np.array([0.8, 0.2, 0]),
                            "children": {
                                "DOG": {
                                    "embedding": np.array([0.7, 0.3, 0]),
                                    "children": {}
                                },
                                "embedding": np.array([0.75, 0.25, 0])  # This should be filtered out
                            }
                        },
                        "Objects": {
                            "embedding": np.array([0.3, 0.7, 0]),
                            "children": {
                                "embedding": np.array([0.3, 0.7, 0])  # This should be filtered out
                            }
                        },
                        "embedding": np.array([0.55, 0.45, 0])  # This should be filtered out
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    # First call returns high similarity with Animals, second call returns DOG
                    mock_search.side_effect = [(0.8, 0), (0.8, 0)]  # High similarity for both levels
                    
                    result = categorizer._hierarchical_sem_search(
                        query="dog",
                        current_label="ENTITY",
                        current_node=categorizer.taxonomy
                    )
                    
                    # Should find DOG (leaf node) after drilling down through Animals
                    self.assertEqual(result, "DOG")
                    
                    # Verify _semantic_search was called twice (once for top level, once for Animals level)
                    self.assertEqual(mock_search.call_count, 2)
                    
                    # Check that the first call had 2 corpus vectors (Animals, Objects) not 3 (which would include 'embedding')
                    first_call_args = mock_search.call_args_list[0][0]
                    query_vect, corpus_vects = first_call_args
                    self.assertEqual(len(corpus_vects), 2)  # Animals and Objects, no embedding key

    def test_embed_taxonomy_children_processing(self):
        """Test that _embed_taxonomy correctly processes 'children' key."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["description"]
            
            test_node = {
                "children": {
                    "child1": "desc1",
                    "child2": "desc2"
                }
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should have processed children recursively
            self.assertIn("children", result)
            self.assertIn("embedding", result)
            
            # The children structure should be processed as a subtree, but child1/child2 
            # are filtered out because they're not in taxonomic_features or "children"
            children_result = result["children"]
            self.assertIn("embedding", children_result)
            # child1 and child2 should NOT be present because they're filtered out
            self.assertNotIn("child1", children_result)
            self.assertNotIn("child2", children_result)

    def test_embed_taxonomy_taxonomic_features_processing(self):
        """Test that _embed_taxonomy correctly processes taxonomic features."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["description", "wordnet_synsets"]
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            test_node = {
                "description": "A test description",
                "wordnet_synsets": ["test.n.01"],
                "ignored_key": "should be ignored"
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should have processed taxonomic features as text embeddings
            self.assertIn("description", result)
            self.assertIn("wordnet_synsets", result)
            self.assertIn("embedding", result)
            
            # Taxonomic features should have embedding structure
            self.assertIn("embedding", result["description"])
            self.assertIn("embedding", result["wordnet_synsets"])
            
            # Ignored key should not be processed when using specific taxonomic features
            self.assertNotIn("ignored_key", result)
            
            # Should have called _embed at least once (description + any synset lemmas)
            self.assertGreater(categorizer._embed.call_count, 0)
            categorizer._embed.assert_any_call("A test description")

    def test_embed_taxonomy_custom_taxonomic_features(self):
        """Test that custom taxonomic_features parameter works correctly."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["description"]  # Use a known feature
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            test_node = {
                "description": "custom content",
                "another_feature": "should be ignored",
                "wordnet_synsets": ["should.be.ignored"]
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should process only the specified taxonomic features
            self.assertIn("description", result)
            
            # Should not process features not in taxonomic_features
            self.assertNotIn("another_feature", result)
            self.assertNotIn("wordnet_synsets", result)

    def test_embed_taxonomy_ignores_unknown_keys(self):
        """Test that _embed_taxonomy ignores unknown keys that are not taxonomic features or children."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["description"]
            categorizer._embed = Mock(side_effect=[
                np.array([0.1, 0.2, 0.3])  # Description embedding
            ])
            
            test_node = {
                "description": "valid feature",
                "unknown_key1": "should be ignored",
                "unknown_key2": "should be ignored"
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should process only taxonomic features
            self.assertIn("description", result)
            self.assertIn("embedding", result)
            # Should ignore unknown keys
            self.assertNotIn("unknown_key1", result)
            self.assertNotIn("unknown_key2", result)


    def test_embed_taxonomy_default_features_fallback(self):
        """Test that default taxonomic features are used when taxonomic_features is not set."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            # Don't set taxonomic_features - should use default
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            test_node = {
                "description": "test description",
                "wordnet_synsets": ["test.n.01"],
                "other_key": "should be ignored"
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should process default taxonomic features
            self.assertIn("description", result)
            self.assertIn("wordnet_synsets", result)
            # Should ignore unknown key
            self.assertNotIn("other_key", result)
            self.assertIn("embedding", result)

    def test_embed_taxonomy_wordnet_synsets_processing(self):
        """Test that _embed_taxonomy correctly processes wordnet_synsets."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["wordnet_synsets"]
            
            # Mock the _embed method to return predictable embeddings
            # Use return_value instead of side_effect to handle variable number of calls
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            test_node = {
                "wordnet_synsets": ["person.n.01"]  # This is a real WordNet synset
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should have processed wordnet_synsets
            self.assertIn("wordnet_synsets", result)
            self.assertIn("embedding", result)
            
            # Should have embedding for wordnet_synsets
            self.assertIn("embedding", result["wordnet_synsets"])
            
            # Should have called _embed at least once (person.n.01 has multiple lemmas)
            self.assertGreater(categorizer._embed.call_count, 0)

    def test_embed_taxonomy_wordnet_synsets_with_underscore_replacement(self):
        """Test that underscores in synset lemmas are replaced with spaces."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["wordnet_synsets"]
            
            # Track what gets passed to _embed
            embed_calls = []
            def mock_embed(text):
                embed_calls.append(text)
                return np.array([0.1, 0.2, 0.3])
            
            categorizer._embed = Mock(side_effect=mock_embed)
            
            test_node = {
                "wordnet_synsets": ["building.n.01"]  # This synset has lemmas with underscores
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should have processed the synset
            self.assertIn("wordnet_synsets", result)
            
            # Check that any underscores in the embed calls were replaced with spaces
            for call in embed_calls:
                self.assertNotIn("_", call, f"Underscore found in embed call: '{call}'")

    def test_embed_taxonomy_wordnet_synsets_error_handling(self):
        """Test that _embed_taxonomy handles synset errors gracefully."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            categorizer.taxonomic_features = ["wordnet_synsets"]
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            test_node = {
                "wordnet_synsets": ["invalid.synset.01", "another.invalid.01"]
            }
            
            # Should not raise exception, but should handle error gracefully
            result = categorizer._embed_taxonomy(test_node)
            
            # Should still return result with embedding structure
            self.assertIn("embedding", result)
            # Since no valid synsets, wordnet_synsets key should not be in result
            self.assertNotIn("wordnet_synsets", result)

    def test_taxonomic_features_fallback_when_missing_attribute(self):
        """Test that _embed_taxonomy uses default taxonomic features when attribute is missing."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            # Intentionally don't set taxonomic_features attribute
            categorizer._embed = Mock(return_value=np.array([0.1, 0.2, 0.3]))
            
            test_node = {
                "description": "test description",
                "wordnet_synsets": ["test.n.01"],
                "unknown_feature": "should be ignored"
            }
            
            result = categorizer._embed_taxonomy(test_node)
            
            # Should use default taxonomic features
            self.assertIn("description", result)
            self.assertIn("wordnet_synsets", result)
            # Should ignore unknown feature
            self.assertNotIn("unknown_feature", result)
            self.assertIn("embedding", result)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_custom_taxonomic_features_initialization(self, mock_sentence_transformer):
        """Test that custom taxonomic_features parameter is properly set during initialization."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                custom_features = ["description"]
                categorizer = SpanCategorizer(taxonomic_features=custom_features)
                
                self.assertEqual(categorizer.taxonomic_features, custom_features)

    @patch('src.tax_span_cat.SpanCategorizer.SentenceTransformer')
    def test_default_taxonomic_features_when_none_provided(self, mock_sentence_transformer):
        """Test that default taxonomic features are used when None is provided."""
        mock_sentence_transformer.return_value = Mock()
        
        with patch.object(SpanCategorizer, '_load_taxonomy_from_path') as mock_load:
            mock_load.return_value = self.sample_taxonomy
            with patch.object(SpanCategorizer, '_embed_taxonomy') as mock_embed_tax:
                mock_embed_tax.return_value = self.sample_taxonomy
                
                categorizer = SpanCategorizer(taxonomic_features=None)
                
                self.assertEqual(categorizer.taxonomic_features, SpanCategorizer.default_taxonomic_features)


if __name__ == '__main__':
    unittest.main()