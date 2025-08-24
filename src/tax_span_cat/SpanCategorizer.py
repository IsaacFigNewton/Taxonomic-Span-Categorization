from importlib.resources import files
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import spacy
from spacy.tokens import Doc, Span

from sentence_transformers import SentenceTransformer

import tax_span_cat.taxonomies
from .TaxonomyValidator import TaxonomyValidator

class SpanCategorizer:
    default_taxonomy_path = files(tax_span_cat.taxonomies).joinpath(f"SpaCy_NER.json")
    default_taxonomic_features = ["description", "wordnet_synsets"]

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 taxonomy: Optional[Dict] = None,
                 taxonomy_path: Optional[str] = None,
                 threshold: float = 0.5,
                 taxonomic_features: List[str]|None = None,
                 preserve_existing_ents: bool = False):
        # taxonomic features to include in taxonomic embeddings
        if not taxonomic_features:
            taxonomic_features = self.default_taxonomic_features
        self.taxonomic_features = taxonomic_features
        
        # whether to preserve existing entities when adding new ones
        self.preserve_existing_ents = preserve_existing_ents

        self._init_embedding_model(embedding_model)
        self._init_taxonomy(taxonomy, taxonomy_path)
        # similarity threshold for assigning a particular label
        self.threshold = threshold
    
    
    def _init_embedding_model(self, model_name: str):
        """Initialize sentence embedding model"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Failed to load embedding model: {e}, loading SpaCy en_core_web_lg backup.")
            self.embedding_model = spacy.load("en_core_web_lg")
    

    def _init_taxonomy(self, taxonomy: Optional[Dict], taxonomy_path: Optional[str]):
        """Initialize the embedding taxonomy"""
        if taxonomy:
            self.taxonomy = taxonomy
        elif taxonomy_path:
            # load the taxonomy from the provided path
            self.taxonomy = self._load_taxonomy_from_path(taxonomy_path)
        else:
            # load the taxonomy from the default path
            self.taxonomy = self._load_taxonomy_from_path(self.default_taxonomy_path)
        
        # validate the taxonomy before embedding
        validator = TaxonomyValidator(taxonomic_features=self.taxonomic_features)
        validation_result = validator.validate_taxonomy(self.taxonomy)
        
        if not validation_result.is_valid:
            error_messages = []
            for error in validation_result.errors:
                error_messages.append(f"{error.path}: {error.message}")
            
            raise ValueError(
                f"Taxonomy validation failed with {len(validation_result.errors)} error(s):\n" +
                "\n".join(error_messages)
            )
        
        # Log warnings if any
        if validation_result.has_warnings:
            for warning in validation_result.warnings:
                print(f"Taxonomy validation warning at {warning.path}: {warning.message}")
        
        # embed the taxonomy's entries
        self.taxonomy = self._embed_taxonomy(self.taxonomy)


    def _load_taxonomy_from_path(self,
            file_path: str,
            iters:int = 0
        ) -> Dict:
        """Load taxonomy from a JSON file"""
        if iters > 1:
            raise FileNotFoundError(f"Default taxonomy not found at {self.default_taxonomy_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {file_path} not found. Loading default taxonomy instead.")
            return self._load_taxonomy_from_path(self.default_taxonomy_path, iters=iters+1)
    

    def _embed(self, text: str) -> np.ndarray:
        # if it's a sentence_transformers embedding model
        if hasattr(self.embedding_model, 'encode'):
            embedding = self.embedding_model.encode([text])[0]
        # if it's a spacy model
        else:
            embedding = self.embedding_model(text).vector

        return normalize(embedding.reshape(1, -1))[0]


    def _embed_taxonomy(self, node: Dict | str) -> Dict[str, Dict]:
        """Recursively embed a taxonomy's entries"""

        # if it's a leaf; ie a text description, synset, or other info
        if isinstance(node, str):
            return {"embedding": self._embed(node)}
        
        # if it's a subtree within the taxonomy
        else:
            new_node = dict()
            subtree_centroids = list()
            
            # Get taxonomic_features with fallback for backward compatibility
            taxonomic_features = getattr(self, 'taxonomic_features', self.default_taxonomic_features)
            
            for label, subtree in node.items():
                # Process children
                if label == "children":
                    # Process children directly - don't apply filtering to child nodes
                    new_node[label] = {}
                    for child_name, child_node in subtree.items():
                        # Recursively embed each child node
                        embedded_child = self._embed_taxonomy(child_node)
                        
                        # Ensure every child has a meaningful embedding - use WordNet synset fallback for zero embeddings
                        if isinstance(embedded_child, dict) and "embedding" in embedded_child:
                            # Check if embedding is all zeros (indicates no taxonomic features were found)
                            if np.allclose(embedded_child["embedding"], 0):
                                # Use child_name as WordNet synset fallback
                                try:
                                    synset = wn.synset(child_name)
                                    lemma_names = [
                                        str(lemma.name()).replace('_', ' ')
                                        for lemma in synset.lemmas()
                                    ]
                                    if lemma_names:
                                        synset_embeddings = [
                                            self._embed(synonym)
                                            for synonym in lemma_names
                                        ]
                                        embedded_child["embedding"] = np.mean(np.array(synset_embeddings), axis=0)
                                    else:
                                        # Fallback to embedding the synset name directly
                                        embedded_child["embedding"] = self._embed(child_name.replace('_', ' '))
                                except Exception:
                                    # If WordNet lookup fails, embed the key directly
                                    embedded_child["embedding"] = self._embed(child_name.replace('_', ' '))
                        
                        new_node[label][child_name] = embedded_child
                        # Collect the child's embedding for parent centroid calculation
                        if isinstance(embedded_child, dict) and "embedding" in embedded_child:
                            subtree_centroids.append(embedded_child["embedding"])
                
                # Process taxonomic features
                elif label in taxonomic_features:
                    # handle each feature according to its format/structure
                    match label:
                        case "wordnet_synsets":
                            # use nltk.wn to expand each synset in subtree
                            expanded_synsets = []
                            for syn in subtree:
                                try:
                                    synset = wn.synset(syn)
                                    lemma_names = [
                                        str(lemma.name()).replace('_', ' ')
                                        for lemma in synset.lemmas()
                                    ]
                                    expanded_synsets.extend(lemma_names)
                                except Exception as e:
                                    print(f"Warning: Could not process synset '{syn}': {e}")
                            
                            # embed each synonym and compute centroid
                            if expanded_synsets:
                                synset_embeddings = [
                                    self._embed(synonym)
                                    for synonym in expanded_synsets
                                ]
                                synset_centroid = np.mean(np.array(synset_embeddings), axis=0)
                                new_node[label] = {"embedding": synset_centroid}
                                subtree_centroids.append(new_node[label]["embedding"])
                        case "description":
                            # embed the text content directly
                            new_node[label] = {"embedding": self._embed(subtree)}
                            subtree_centroids.append(new_node[label]["embedding"])
                # Preserve important metadata fields that aren't taxonomic features
                elif label in ["label", "id", "category", "type"]:  # Add other metadata fields as needed
                    new_node[label] = subtree
            # centroid = mean of normalized child embeddings
            if subtree_centroids:
                new_node["embedding"] = np.mean(np.array(subtree_centroids), axis=0)
            else:
                # If no embeddings available, create a zero vector
                # This should be rare but handles edge cases like root-only nodes
                embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
                try:
                    if hasattr(self.embedding_model, 'get_sentence_embedding_dimension'):
                        dim = self.embedding_model.get_sentence_embedding_dimension()
                        if isinstance(dim, int):
                            embedding_dim = dim
                    elif hasattr(self.embedding_model, 'vector_size'):
                        dim = self.embedding_model.vector_size
                        if isinstance(dim, int):
                            embedding_dim = dim
                except:
                    # Fallback to default if anything goes wrong
                    pass
                new_node["embedding"] = np.zeros(embedding_dim)
            return new_node
        

    def _semantic_search(self,
            query_vect: np.ndarray,
            corpus_vects: List[np.ndarray]
        ) -> int:
        """
        Takes in a query vector and a list of corpus vectors
        
        Returns:
        - the cosine similarity of the best match
        - the index of the best match
        """
        similarities = cosine_similarity(
            np.array([query_vect]),
            np.array(corpus_vects)
        )[0]
        highest_similarity = max(similarities)
        best_match_idx = np.argmax(similarities)
        return (
            highest_similarity,
            best_match_idx
        )


    def _extract_best_label(self, node: dict, synset_key: str) -> str:
        """
        Enhanced label extraction method that tries multiple fallback strategies.
        
        Args:
            node: The taxonomy node dictionary
            synset_key: The synset key as fallback
            
        Returns:
            Best available label string
        """
        # Safety check: if node is not a dictionary, convert to string and return
        if not isinstance(node, dict):
            return str(node)
        
        # Safety check: if node only has embedding and no other useful fields,
        # use the synset key
        if "embedding" in node and len([k for k in node.keys() if k not in ["embedding"]]) == 0:
            if synset_key:
                # Remove .n.01 style endings
                clean_key = synset_key.split('.')[0]
                # Replace underscores with spaces and titlecase
                clean_key = clean_key.replace('_', ' ').title()
                return clean_key
            return "UNKNOWN"
        
        # Try "label" field first
        if "label" in node and node["label"]:
            return node["label"]
        
        # Fall back to "description" field (truncated if too long)
        if "description" in node and node["description"]:
            description = str(node["description"])
            # Truncate long descriptions to first 50 characters
            if len(description) > 50:
                description = description[:47] + "..."
            return description
        
        # Finally clean up synset keys as last resort
        if synset_key:
            # If synset_key is already in a clean format (all caps, no dots, no underscores), return as-is
            if synset_key.isupper() and '.' not in synset_key and '_' not in synset_key:
                return synset_key
            # Otherwise, remove .n.01 style endings and clean up
            clean_key = synset_key.split('.')[0]
            # Replace underscores with spaces and titlecase
            clean_key = clean_key.replace('_', ' ').title()
            return clean_key
        
        # Ultimate fallback
        return "UNKNOWN"

    def _hierarchical_sem_search(self,
            query: str,
            current_label: str,
            current_node: dict,
            depth: int = 0,
            best_match_so_far: tuple = None
        ) -> str:
        """
        Takes in a piece of query text and performs a hierarchical semantic search through the taxonomy.
        Enhanced with improved threshold logic and better fallback strategy.

        Returns the label in the taxonomy with the highest similarity to the query.
        
        Args:
            query: The query text to search for
            current_label: Current node label
            current_node: Current taxonomy node
            depth: Current depth in hierarchy (for adaptive threshold)
            best_match_so_far: Tuple of (similarity, label) for best match found so far
        """
        # Initialize best match tracking
        if best_match_so_far is None:
            best_match_so_far = (0.0, current_label)
        
        # Check if this is a leaf node (has no children)
        if "children" not in current_node:
            # Return the label of this leaf node using enhanced extraction
            leaf_label = self._extract_best_label(current_node, current_label)
            return leaf_label

        # get an ordered list of the labelled embeddings (excluding the embedding key itself)
        children = [key for key in current_node["children"].keys() if key != "embedding"]
        
        # if there are no valid children (only embedding keys), return current label
        if not children:
            return self._extract_best_label(current_node, current_label)
            
        # get query embedding
        query_vect = self._embed(query)
        # get embeddings of taxonomic terms at the current level
        corpus_vects = []
        valid_children = []
        for child in children:
            child_node = current_node["children"][child]
            if "embedding" in child_node:
                corpus_vects.append(child_node["embedding"])
                valid_children.append(child)
        
        # Update children list to only include those with embeddings
        children = valid_children
        
        # Handle case where no children have embeddings
        if not corpus_vects:
            return self._extract_best_label(current_node, current_label)
        
        # get idx of closest match
        best_similarity, best_match_idx = self._semantic_search(query_vect, corpus_vects)
        # get synset key of closest match
        best_match_synset = children[best_match_idx]
        # get the actual label for display
        best_match_node = current_node["children"][best_match_synset]
        best_match_label = self._extract_best_label(best_match_node, best_match_synset)

        # Update best match so far if this is better
        if best_similarity > best_match_so_far[0]:
            best_match_so_far = (best_similarity, best_match_label)

        print(f"Best match for '{query}' at depth {depth} is '{best_match_label}' with similarity of {best_similarity}")
        
        # Adaptive threshold: slightly decrease threshold with depth to allow deeper exploration
        adaptive_threshold = max(0.0, self.threshold - (depth * 0.05))
        
        # Improved threshold logic: continue searching or return best match found
        if best_similarity > adaptive_threshold:
            # Continue deeper into hierarchy
            return self._hierarchical_sem_search(
                query=query,
                current_label=best_match_label,
                current_node=best_match_node,
                depth=depth + 1,
                best_match_so_far=best_match_so_far
            )
        else:
            # Below threshold - improved fallback strategy
            # For backward compatibility, if we're at the root level, similarity is very low,
            # AND current_label is "ENTITY", return it for backward compatibility
            if depth == 0 and best_similarity <= self.threshold and current_label == "ENTITY":
                return current_label
            
            # Otherwise, return the best match found in the search path
            # Only fall back to parent if no good match was found at all
            min_confidence = 0.1  # Minimum confidence threshold
            if best_match_so_far[0] >= min_confidence:
                return best_match_so_far[1]
            else:
                # Return current level's best match even if below threshold
                return best_match_label


    def __call__(self, doc: Doc) -> Doc:
        """
        Takes a SpaCy Doc, classifies noun chunks using a hierarchical semantic search process through the taxonomy, and returns a new Doc with NER applied to associated spans.
        """
        ner_doc = doc.copy()
        new_spans = []

        for chunk in doc.noun_chunks:
            # label the current chunk
            ent_label = self._hierarchical_sem_search(
                query=chunk.text,
                current_label="ENTITY",
                current_node=self.taxonomy,
            )
            span = Span(
                ner_doc,
                start=chunk.start,
                end=chunk.end,
                label=ent_label
            )
            
            # Initialize the spans list under the 'sc' key for span categorization
            if 'sc' not in ner_doc.spans:
                ner_doc.spans['sc'] = []
            # add the span to the doc's spans under the 'sc' key
            ner_doc.spans['sc'].append(span)
            
            # also add to individual label groups for backward compatibility
            if ent_label not in ner_doc.spans.keys():
                ner_doc.spans[ent_label] = list()
            ner_doc.spans[ent_label].append(span)
            
            # Collect new spans for entity resolution
            new_spans.append(span)
        
        # Handle entity conflicts based on preserve_existing_ents setting
        try:
            if self.preserve_existing_ents:
                # Get existing entities
                existing_ents = list(ner_doc.ents)
                
                # Filter out new spans that would conflict with existing entities
                non_conflicting_spans = []
                for span in new_spans:
                    has_conflict = False
                    for ent in existing_ents:
                        # Check if span overlaps with existing entity
                        if (span.start < ent.end and ent.start < span.end):
                            has_conflict = True
                            break
                    
                    if not has_conflict:
                        non_conflicting_spans.append(span)
                
                # Set entities with existing entities and non-conflicting new spans
                ner_doc.set_ents(existing_ents + non_conflicting_spans)
            else:
                # Replace all entities with our new spans only
                ner_doc.set_ents(new_spans)
                
        except Exception as e:
            print(f"WARNING: Failed to set entities: {e}")

        return ner_doc
