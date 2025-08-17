import unittest
import tempfile
import json
import os
from unittest.mock import patch, Mock

from src.tax_span_cat.TaxonomyValidator import (
    TaxonomyValidator, 
    ValidationError, 
    ValidationResult
)


class TestTaxonomyValidator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.validator = TaxonomyValidator()
        
        # Valid taxonomy structures for testing
        self.valid_simple_taxonomy = {
            "animal.n.01": {
                "label": "ANIMAL",
                "description": "A living creature"
            }
        }
        
        self.valid_complex_taxonomy = {
            "animal.n.01": {
                "label": "ANIMAL", 
                "description": "A living creature",
                "children": {
                    "dog.n.01": {
                        "label": "DOG",
                        "description": "A domesticated canine",
                        "wordnet_synsets": ["dog.n.01", "domestic_dog.n.01"]
                    },
                    "cat.n.01": {
                        "label": "CAT",
                        "description": "A domesticated feline"
                    }
                }
            },
            "person.n.01": {
                "label": "PERSON",
                "description": "A human being"
            }
        }
    
    def test_init_default_taxonomic_features(self):
        """Test initialization with default taxonomic features."""
        validator = TaxonomyValidator()
        self.assertEqual(validator.taxonomic_features, ["description", "wordnet_synsets"])
    
    def test_init_custom_taxonomic_features(self):
        """Test initialization with custom taxonomic features."""
        custom_features = ["description", "custom_feature"]
        validator = TaxonomyValidator(taxonomic_features=custom_features)
        self.assertEqual(validator.taxonomic_features, custom_features)
    
    def test_validate_synset_valid(self):
        """Test validation of valid WordNet synsets."""
        # Test common valid synsets
        self.assertIsNotNone(self.validator.validate_synset("dog.n.01"))
        self.assertIsNotNone(self.validator.validate_synset("cat.n.01"))
        self.assertIsNotNone(self.validator.validate_synset("person.n.01"))
    
    def test_validate_synset_invalid_format(self):
        """Test validation of synsets with invalid format."""
        # Invalid format cases
        self.assertIsNone(self.validator.validate_synset("dog"))
        self.assertIsNone(self.validator.validate_synset("dog.n"))
        self.assertIsNone(self.validator.validate_synset("dog.n."))
        self.assertIsNone(self.validator.validate_synset("dog.n.abc"))
        self.assertIsNone(self.validator.validate_synset(""))
        self.assertIsNone(self.validator.validate_synset("dog.n.01.extra"))
    
    def test_validate_synset_nonexistent(self):
        """Test validation of non-existent synsets."""
        self.assertIsNone(self.validator.validate_synset("nonexistent.n.999"))
        self.assertIsNone(self.validator.validate_synset("fake.word.01"))
    
    def test_validate_description_valid(self):
        """Test validation of valid descriptions."""
        self.assertTrue(self.validator.validate_description("A valid description"))
        self.assertTrue(self.validator.validate_description("  Trimmed  "))
        self.assertTrue(self.validator.validate_description("Single word"))
    
    def test_validate_description_invalid(self):
        """Test validation of invalid descriptions."""
        self.assertFalse(self.validator.validate_description(""))
        self.assertFalse(self.validator.validate_description("   "))
        self.assertFalse(self.validator.validate_description(123))
        self.assertFalse(self.validator.validate_description(None))
    
    def test_validate_label_valid(self):
        """Test validation of valid labels."""
        self.assertTrue(self.validator.validate_label("ANIMAL"))
        self.assertTrue(self.validator.validate_label("DOG"))
        self.assertTrue(self.validator.validate_label("  CAT  "))
    
    def test_validate_label_invalid(self):
        """Test validation of invalid labels."""
        self.assertFalse(self.validator.validate_label(""))
        self.assertFalse(self.validator.validate_label("   "))
        self.assertFalse(self.validator.validate_label(123))
        self.assertFalse(self.validator.validate_label(None))
    
    def test_validate_node_structure_valid_dict(self):
        """Test validation of valid node structure."""
        valid_node = {
            "label": "ANIMAL",
            "description": "A living creature"
        }
        errors = self.validator.validate_node_structure(valid_node, "test")
        self.assertEqual(len(errors), 0)
    
    def test_validate_node_structure_invalid_type(self):
        """Test validation of node with invalid type."""
        errors = self.validator.validate_node_structure("not a dict", "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "type_error")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_node_structure_unknown_keys(self):
        """Test validation of node with unknown keys."""
        node_with_unknown = {
            "label": "ANIMAL",
            "unknown_key": "value"
        }
        errors = self.validator.validate_node_structure(node_with_unknown, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "unknown_keys")
        self.assertEqual(errors[0].severity, "warning")
    
    def test_validate_node_structure_invalid_label(self):
        """Test validation of node with invalid label."""
        node_with_invalid_label = {
            "label": ""
        }
        errors = self.validator.validate_node_structure(node_with_invalid_label, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "invalid_label")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_node_structure_invalid_description(self):
        """Test validation of node with invalid description."""
        node_with_invalid_desc = {
            "description": 123
        }
        errors = self.validator.validate_node_structure(node_with_invalid_desc, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "invalid_description")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_node_structure_invalid_synsets_type(self):
        """Test validation of node with invalid synsets type."""
        node_with_invalid_synsets = {
            "wordnet_synsets": "not a list"
        }
        errors = self.validator.validate_node_structure(node_with_invalid_synsets, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "invalid_synsets_type")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_node_structure_invalid_synset_items(self):
        """Test validation of node with invalid synset items."""
        node_with_invalid_synset_items = {
            "wordnet_synsets": [123, "dog.n.01"]
        }
        errors = self.validator.validate_node_structure(node_with_invalid_synset_items, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "invalid_synset_type")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_node_structure_nonexistent_synsets(self):
        """Test validation of node with non-existent synsets."""
        node_with_nonexistent_synsets = {
            "wordnet_synsets": ["fake.n.999", "dog.n.01"]
        }
        errors = self.validator.validate_node_structure(node_with_nonexistent_synsets, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "invalid_synset")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_node_structure_invalid_children_type(self):
        """Test validation of node with invalid children type."""
        node_with_invalid_children = {
            "children": "not a dict"
        }
        errors = self.validator.validate_node_structure(node_with_invalid_children, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "invalid_children_type")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_leaf_node_valid(self):
        """Test validation of valid leaf node."""
        valid_leaf = {
            "label": "DOG",
            "description": "A canine animal"
        }
        errors = self.validator.validate_leaf_node(valid_leaf, "test")
        self.assertEqual(len(errors), 0)
    
    def test_validate_leaf_node_missing_label(self):
        """Test validation of leaf node missing label."""
        leaf_without_label = {
            "description": "A canine animal"
        }
        errors = self.validator.validate_leaf_node(leaf_without_label, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "missing_label")
        self.assertEqual(errors[0].severity, "error")
    
    def test_validate_leaf_node_no_taxonomic_features(self):
        """Test validation of leaf node without taxonomic features."""
        leaf_without_features = {
            "label": "DOG"
        }
        errors = self.validator.validate_leaf_node(leaf_without_features, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "no_taxonomic_features")
        self.assertEqual(errors[0].severity, "warning")
    
    def test_validate_internal_node_valid(self):
        """Test validation of valid internal node."""
        valid_internal = {
            "label": "ANIMAL",
            "description": "A living creature",
            "children": {
                "dog.n.01": {"label": "DOG"}
            }
        }
        errors = self.validator.validate_internal_node(valid_internal, "test")
        self.assertEqual(len(errors), 0)
    
    def test_validate_internal_node_missing_label(self):
        """Test validation of internal node missing label."""
        internal_without_label = {
            "children": {
                "dog.n.01": {"label": "DOG"}
            }
        }
        errors = self.validator.validate_internal_node(internal_without_label, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "missing_label")
        self.assertEqual(errors[0].severity, "warning")
    
    def test_validate_internal_node_empty_children(self):
        """Test validation of internal node with empty children."""
        internal_with_empty_children = {
            "label": "ANIMAL",
            "children": {}
        }
        errors = self.validator.validate_internal_node(internal_with_empty_children, "test")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].error_type, "empty_children")
        self.assertEqual(errors[0].severity, "warning")
    
    def test_validate_taxonomy_valid_simple(self):
        """Test validation of valid simple taxonomy."""
        result = self.validator.validate_taxonomy(self.valid_simple_taxonomy)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        # May have warnings about unknown keys (taxonomy node names)
    
    def test_validate_taxonomy_valid_complex(self):
        """Test validation of valid complex taxonomy."""
        result = self.validator.validate_taxonomy(self.valid_complex_taxonomy)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        # May have warnings about unknown keys (taxonomy node names)
    
    def test_validate_taxonomy_invalid_root_type(self):
        """Test validation of taxonomy with invalid root type."""
        result = self.validator.validate_taxonomy("not a dict")
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertEqual(result.errors[0].error_type, "type_error")
    
    def test_validate_taxonomy_circular_reference(self):
        """Test detection of circular references."""
        # Create a circular reference manually
        node1 = {
            "label": "NODE1",
            "children": {}
        }
        
        node2 = {
            "label": "NODE2", 
            "children": {}
        }
        
        # Add circular reference
        node1["children"]["node2"] = node2
        node2["children"]["node1"] = node1  # Circular reference
        
        circular_taxonomy = {
            "node1": node1
        }
        
        result = self.validator.validate_taxonomy(circular_taxonomy)
        self.assertFalse(result.is_valid)
        # Should detect circular reference
        circular_errors = [e for e in result.errors if e.error_type == "circular_reference"]
        self.assertGreater(len(circular_errors), 0)
    
    def test_validation_result_properties(self):
        """Test ValidationResult properties."""
        # Test with errors and warnings
        result = ValidationResult(
            is_valid=False,
            errors=[ValidationError("path", "error", "test error", "error")],
            warnings=[ValidationError("path", "warning", "test warning", "warning")],
            info=[]
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors)
        self.assertTrue(result.has_warnings)
        
        # Test without errors
        result_clean = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            info=[]
        )
        
        self.assertTrue(result_clean.is_valid)
        self.assertFalse(result_clean.has_errors)
        self.assertFalse(result_clean.has_warnings)
    
    def test_validate_taxonomy_file_valid(self):
        """Test validation of taxonomy from valid JSON file."""
        # Create temporary file with valid taxonomy
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.valid_simple_taxonomy, f)
            temp_file = f.name
        
        try:
            result = self.validator.validate_taxonomy_file(temp_file)
            self.assertTrue(result.is_valid)
            self.assertEqual(len(result.errors), 0)
            # May have warnings about unknown keys (taxonomy node names)
        finally:
            os.unlink(temp_file)
    
    def test_validate_taxonomy_file_not_found(self):
        """Test validation of non-existent file."""
        result = self.validator.validate_taxonomy_file("nonexistent.json")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].error_type, "file_not_found")
    
    def test_validate_taxonomy_file_invalid_json(self):
        """Test validation of file with invalid JSON."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            temp_file = f.name
        
        try:
            result = self.validator.validate_taxonomy_file(temp_file)
            self.assertFalse(result.is_valid)
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(result.errors[0].error_type, "invalid_json")
        finally:
            os.unlink(temp_file)
    
    def test_validate_taxonomy_deep_nesting(self):
        """Test validation of deeply nested taxonomy."""
        deep_taxonomy = {
            "level1": {
                "label": "LEVEL1",
                "children": {
                    "level2": {
                        "label": "LEVEL2",
                        "children": {
                            "level3": {
                                "label": "LEVEL3",
                                "description": "Deep level"
                            }
                        }
                    }
                }
            }
        }
        
        result = self.validator.validate_taxonomy(deep_taxonomy)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        # May have warnings about unknown keys (taxonomy node names)
    
    def test_validate_taxonomy_mixed_valid_invalid(self):
        """Test validation of taxonomy with both valid and invalid nodes."""
        mixed_taxonomy = {
            "valid_node": {
                "label": "VALID",
                "description": "A valid node"
            },
            "invalid_node": {
                "label": "",  # Invalid empty label
                "description": 123  # Invalid description type
            }
        }
        
        result = self.validator.validate_taxonomy(mixed_taxonomy)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        
        # Should have specific errors
        error_types = [e.error_type for e in result.errors]
        self.assertIn("invalid_label", error_types)
        self.assertIn("invalid_description", error_types)
    
    def test_custom_taxonomic_features(self):
        """Test validator with custom taxonomic features."""
        custom_validator = TaxonomyValidator(taxonomic_features=["description", "custom_field"])
        
        node_with_custom = {
            "label": "TEST",
            "custom_field": "custom value"
        }
        
        # Should not warn about missing taxonomic features since custom_field is present
        result = custom_validator.validate_taxonomy(node_with_custom)
        self.assertTrue(result.is_valid)
        
        # Test with missing custom features - wrap in proper taxonomy structure
        taxonomy_without_custom_features = {
            "test_node": {
                "label": "TEST"
                # No taxonomic features
            }
        }
        
        result = custom_validator.validate_taxonomy(taxonomy_without_custom_features)
        self.assertTrue(result.is_valid)  # Valid but should have warnings
        
        # Check for warning about missing taxonomic features
        warning_types = [w.error_type for w in result.warnings]
        self.assertIn("no_taxonomic_features", warning_types)
    
    def test_validation_error_paths(self):
        """Test that validation errors include correct paths."""
        nested_invalid = {
            "parent": {
                "label": "PARENT",
                "children": {
                    "child": {
                        "label": "",  # Invalid
                        "wordnet_synsets": ["invalid.synset.999"]  # Invalid
                    }
                }
            }
        }
        
        result = self.validator.validate_taxonomy(nested_invalid)
        self.assertFalse(result.is_valid)
        
        # Check that paths are correctly reported
        paths = [e.path for e in result.errors]
        self.assertTrue(any("root.parent.children.child" in path for path in paths))
    
    def test_embedded_taxonomy_compatibility(self):
        """Test that validator handles embedded taxonomies (with embedding keys)."""
        embedded_taxonomy = {
            "animal.n.01": {
                "label": "ANIMAL",
                "description": "A living creature",
                "embedding": [0.1, 0.2, 0.3],  # Should be ignored/allowed
                "children": {
                    "dog.n.01": {
                        "label": "DOG",
                        "embedding": [0.4, 0.5, 0.6]  # Should be ignored/allowed
                    }
                }
            }
        }
        
        result = self.validator.validate_taxonomy(embedded_taxonomy)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        # May have warnings about unknown keys (taxonomy node names)


if __name__ == '__main__':
    unittest.main()