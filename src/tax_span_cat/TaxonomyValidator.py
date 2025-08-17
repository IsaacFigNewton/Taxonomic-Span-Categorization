import json
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
from nltk.corpus import wordnet as wn


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    path: str
    error_type: str
    message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Results of taxonomy validation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class TaxonomyValidator:
    """
    Validates taxonomy structure and content based on the requirements
    discovered from SpanCategorizer debugging.
    """
    
    # Default taxonomic features that should be supported
    DEFAULT_TAXONOMIC_FEATURES = ["description", "wordnet_synsets"]
    
    # Required keys for leaf nodes
    LEAF_REQUIRED_KEYS = {"label"}
    
    # Optional keys for any node
    OPTIONAL_KEYS = {"description", "wordnet_synsets", "children", "label"}
    
    def __init__(self, taxonomic_features: Optional[List[str]] = None):
        """
        Initialize validator.
        
        Args:
            taxonomic_features: List of taxonomic features to validate.
                               Defaults to DEFAULT_TAXONOMIC_FEATURES.
        """
        self.taxonomic_features = taxonomic_features or self.DEFAULT_TAXONOMIC_FEATURES
        self.visited_nodes: Set[str] = set()  # For circular reference detection
    
    def validate_synset(self, synset_str: str) -> Optional[str]:
        """
        Validate a WordNet synset string.
        
        Args:
            synset_str: Synset string like "dog.n.01"
            
        Returns:
            Definition if valid, None if invalid
        """
        try:
            # Check format: should be word.pos.number
            parts = synset_str.split('.')
            if len(parts) != 3:
                return None
                
            lemma, pos, num = parts
            if not lemma or not pos or not num.isdigit():
                return None
                
            # Try to get the synset
            synset = wn.synset(synset_str)
            return synset.definition()
        except Exception:
            return None
    
    def validate_description(self, description: str) -> bool:
        """
        Validate a description field.
        
        Args:
            description: Description text
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(description, str) and len(description.strip()) > 0
    
    def validate_label(self, label: str) -> bool:
        """
        Validate a label field.
        
        Args:
            label: Label text
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(label, str) and len(label.strip()) > 0
    
    def validate_node_structure(self, node: Any, path: str = "root") -> List[ValidationError]:
        """
        Validate the structure of a single node.
        
        Args:
            node: Node to validate
            path: Current path in taxonomy for error reporting
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check if node is a dictionary
        if not isinstance(node, dict):
            errors.append(ValidationError(
                path=path,
                error_type="type_error",
                message=f"Node must be a dictionary, got {type(node).__name__}",
                severity="error"
            ))
            return errors
        
        # Check for unknown keys
        known_keys = self.OPTIONAL_KEYS | {"embedding"} | set(self.taxonomic_features)
        unknown_keys = set(node.keys()) - known_keys
        if unknown_keys:
            errors.append(ValidationError(
                path=path,
                error_type="unknown_keys",
                message=f"Unknown keys found: {unknown_keys}",
                severity="warning"
            ))
        
        # Validate label if present
        if "label" in node:
            if not self.validate_label(node["label"]):
                errors.append(ValidationError(
                    path=path,
                    error_type="invalid_label",
                    message="Label must be a non-empty string",
                    severity="error"
                ))
        
        # Validate description if present
        if "description" in node:
            if not self.validate_description(node["description"]):
                errors.append(ValidationError(
                    path=path,
                    error_type="invalid_description", 
                    message="Description must be a non-empty string",
                    severity="error"
                ))
        
        # Validate wordnet_synsets if present
        if "wordnet_synsets" in node:
            if not isinstance(node["wordnet_synsets"], list):
                errors.append(ValidationError(
                    path=path,
                    error_type="invalid_synsets_type",
                    message="wordnet_synsets must be a list",
                    severity="error"
                ))
            else:
                for i, synset in enumerate(node["wordnet_synsets"]):
                    if not isinstance(synset, str):
                        errors.append(ValidationError(
                            path=f"{path}.wordnet_synsets[{i}]",
                            error_type="invalid_synset_type",
                            message="Synset must be a string",
                            severity="error"
                        ))
                    elif self.validate_synset(synset) is None:
                        errors.append(ValidationError(
                            path=f"{path}.wordnet_synsets[{i}]",
                            error_type="invalid_synset",
                            message=f"Invalid WordNet synset: {synset}",
                            severity="error"
                        ))
        
        # Validate children if present
        if "children" in node:
            if not isinstance(node["children"], dict):
                errors.append(ValidationError(
                    path=path,
                    error_type="invalid_children_type",
                    message="children must be a dictionary",
                    severity="error"
                ))
        
        return errors
    
    def validate_leaf_node(self, node: Dict[str, Any], path: str) -> List[ValidationError]:
        """
        Validate a leaf node (node without children).
        
        Args:
            node: Leaf node to validate
            path: Current path in taxonomy
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Leaf nodes should have a label
        if "label" not in node:
            errors.append(ValidationError(
                path=path,
                error_type="missing_label",
                message="Leaf nodes must have a 'label' field",
                severity="error"
            ))
        
        # Leaf nodes should have at least one taxonomic feature
        has_feature = any(feature in node for feature in self.taxonomic_features)
        if not has_feature and "label" in node:
            # Only warn if the node has other content
            errors.append(ValidationError(
                path=path,
                error_type="no_taxonomic_features",
                message=f"Leaf node has no taxonomic features from: {self.taxonomic_features}",
                severity="warning"
            ))
        
        return errors
    
    def validate_internal_node(self, node: Dict[str, Any], path: str) -> List[ValidationError]:
        """
        Validate an internal node (node with children).
        
        Args:
            node: Internal node to validate
            path: Current path in taxonomy
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Internal nodes should have a label for proper hierarchy navigation
        if "label" not in node:
            errors.append(ValidationError(
                path=path,
                error_type="missing_label",
                message="Internal nodes should have a 'label' field",
                severity="warning"
            ))
        
        # Check if children is empty
        if "children" in node and len(node["children"]) == 0:
            errors.append(ValidationError(
                path=path,
                error_type="empty_children",
                message="Node has empty children dictionary",
                severity="warning"
            ))
        
        return errors
    
    def validate_taxonomy(self, taxonomy: Any, path: str = "root") -> ValidationResult:
        """
        Recursively validate the entire taxonomy.
        
        Args:
            taxonomy: Taxonomy structure to validate
            path: Current path for error reporting
            
        Returns:
            ValidationResult with all errors, warnings, and info
        """
        all_errors = []
        
        # Reset visited nodes for circular reference detection
        if path == "root":
            self.visited_nodes.clear()
        
        # Check for circular references
        node_id = str(id(taxonomy))
        if node_id in self.visited_nodes:
            all_errors.append(ValidationError(
                path=path,
                error_type="circular_reference",
                message="Circular reference detected",
                severity="error"
            ))
            return ValidationResult(
                is_valid=False,
                errors=[e for e in all_errors if e.severity == "error"],
                warnings=[e for e in all_errors if e.severity == "warning"],
                info=[e for e in all_errors if e.severity == "info"]
            )
        
        self.visited_nodes.add(node_id)
        
        try:
            # Validate node structure
            all_errors.extend(self.validate_node_structure(taxonomy, path))
            
            # If not a dict, can't continue validation
            if not isinstance(taxonomy, dict):
                return ValidationResult(
                    is_valid=False,
                    errors=[e for e in all_errors if e.severity == "error"],
                    warnings=[e for e in all_errors if e.severity == "warning"],
                    info=[e for e in all_errors if e.severity == "info"]
                )
            
            # Root level must have wrapped structure
            if path == "root":
                # Root must have a "children" key - wrapped structure is required
                if "children" not in taxonomy:
                    all_errors.append(ValidationError(
                        path=path,
                        error_type="missing_children",
                        message="Root taxonomy must have a 'children' key with wrapped structure",
                        severity="error"
                    ))
                    return ValidationResult(
                        is_valid=False,
                        errors=[e for e in all_errors if e.severity == "error"],
                        warnings=[e for e in all_errors if e.severity == "warning"],
                        info=[e for e in all_errors if e.severity == "info"]
                    )
                
                # Validate as internal node (root with children)
                all_errors.extend(self.validate_internal_node(taxonomy, path))
                for key, child in taxonomy["children"].items():
                    child_path = f"{path}.children.{key}"
                    child_result = self.validate_taxonomy(child, child_path)
                    all_errors.extend(child_result.errors)
                    all_errors.extend(child_result.warnings)
                    all_errors.extend(child_result.info)
            else:
                # Non-root nodes: determine if this is a leaf or internal node
                is_leaf = "children" not in taxonomy
                
                if is_leaf:
                    all_errors.extend(self.validate_leaf_node(taxonomy, path))
                else:
                    all_errors.extend(self.validate_internal_node(taxonomy, path))
                    
                    # Recursively validate children
                    for key, child in taxonomy["children"].items():
                        child_path = f"{path}.children.{key}"
                        child_result = self.validate_taxonomy(child, child_path)
                        all_errors.extend(child_result.errors)
                        all_errors.extend(child_result.warnings)
                        all_errors.extend(child_result.info)
        
        finally:
            self.visited_nodes.discard(node_id)
        
        # Categorize errors by severity
        errors = [e for e in all_errors if e.severity == "error"]
        warnings = [e for e in all_errors if e.severity == "warning"]
        info = [e for e in all_errors if e.severity == "info"]
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )
    
    def validate_taxonomy_file(self, file_path: str) -> ValidationResult:
        """
        Validate a taxonomy from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            ValidationResult
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                taxonomy = json.load(f)
            return self.validate_taxonomy(taxonomy)
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    path="file",
                    error_type="file_not_found",
                    message=f"File not found: {file_path}",
                    severity="error"
                )],
                warnings=[],
                info=[]
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    path="file",
                    error_type="invalid_json",
                    message=f"Invalid JSON: {e}",
                    severity="error"
                )],
                warnings=[],
                info=[]
            )