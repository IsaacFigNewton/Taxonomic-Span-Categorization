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
            "Animals": {
                "description": "Living creatures",
                "children": {
                    "DOG": {
                        "description": "Canine animals"
                    },
                    "CAT": {
                        "description": "Feline animals" 
                    }
                }
            },
            "Objects": {
                "description": "Inanimate things",
                "children": {
                    "TOOL": {
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
            self.assertIn("child1", result)
            self.assertIn("child2", result)
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
                    "children": {
                        "Animals": {
                            "embedding": np.array([0.9, 0.1, 0]),
                            "children": {
                                "DOG": {
                                    "embedding": np.array([0.8, 0.2, 0]),
                                    "children": {}
                                }
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
                        current_node=categorizer.taxonomy
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
                    "children": {
                        "Animals": {"embedding": np.array([0.1, 0.9, 0])}
                    }
                }
                
                with patch.object(categorizer, '_semantic_search') as mock_search:
                    mock_search.return_value = (0.2, 0)  # Low similarity
                    
                    result = categorizer._hierarchical_sem_search(
                        query="dog",
                        current_label="ENTITY",
                        current_node=categorizer.taxonomy
                    )
                    
                    self.assertEqual(result, "ENTITY")

    def test_hierarchical_sem_search_missing_children(self):
        """Test hierarchical search raises error when children are missing."""
        with patch.object(SpanCategorizer, '_init_embedding_model'):
            categorizer = SpanCategorizer.__new__(SpanCategorizer)
            
            invalid_node = {"embedding": np.array([1, 0, 0])}
            
            with self.assertRaises(KeyError) as context:
                categorizer._hierarchical_sem_search(
                    query="test",
                    current_label="TEST",
                    current_node=invalid_node
                )
                
            self.assertIn("children", str(context.exception))

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
        
        # Set up spans and set_ents mock
        mock_copied_doc.spans = {}
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
                            current_node=categorizer.taxonomy
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


if __name__ == '__main__':
    unittest.main()