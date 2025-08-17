from importlib.resources import files
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import wordnet as wn

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
                 taxonomic_features: List[str]|None = None):
        # taxonomic features to include in taxonomic embeddings
        if not taxonomic_features:
            taxonomic_features = self.default_taxonomic_features
        self.taxonomic_features = taxonomic_features

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
                    # recursively embed each child and collect their embeddings
                    new_node[label] = self._embed_taxonomy(subtree)
                    # Children is a dictionary of child nodes, each with their own embedding
                    # Collect embeddings from all child nodes
                    for child_name, child_node in new_node[label].items():
                        if isinstance(child_node, dict) and "embedding" in child_node:
                            subtree_centroids.append(child_node["embedding"])
                
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


    def _hierarchical_sem_search(self,
            query: str,
            current_label: str,
            current_node: dict
        ) -> str:
        """
        Takes in a piece of query text and performs a hierarchical semantic search through the taxonomy.

        Returns the label in the taxonomy with the highest similarity to the query.
        """
        # Check if this is a leaf node (has no children)
        if "children" not in current_node:
            # Return the label of this leaf node
            return current_node.get("label", current_label)

        # get an ordered list of the labelled embeddings (excluding the embedding key itself)
        children = [key for key in current_node["children"].keys() if key != "embedding"]
        
        # if there are no valid children (only embedding keys), return current label
        if not children:
            return current_node.get("label", current_label)
            
        # get query embedding
        query_vect = self._embed(query)
        # get embeddings of taxonomic terms at the current level
        corpus_vects = [current_node["children"][child]["embedding"] for child in children]
        # get idx of closest match
        best_similarity, best_match_idx = self._semantic_search(query_vect, corpus_vects)
        # get synset key of closest match
        best_match_synset = children[best_match_idx]
        # get the actual label for display
        best_match_node = current_node["children"][best_match_synset]
        best_match_label = best_match_node.get("label", best_match_synset)

        print(f"Best match for '{query}' is '{best_match_label}' with similarity of {best_similarity}")
        if best_similarity <= self.threshold:
            return current_node.get("label", current_label)
        else:
            return self._hierarchical_sem_search(
                query=query,
                current_label=best_match_label,
                current_node=best_match_node
            )


    def __call__(self, doc: Doc) -> Doc:
        """
        Takes a SpaCy Doc, classifies noun chunks using a hierarchical semantic search process through the taxonomy, and returns a new Doc with NER applied to associated spans.
        """
        ner_doc = doc.copy()

        for chunk in doc.noun_chunks:
            # label the current chunk
            ent_label = self._hierarchical_sem_search(
                query=chunk.text,
                current_label="ENTITY",
                current_node={"children": self.taxonomy},
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
            
            try:
                ner_doc.set_ents(list(ner_doc.ents) + [span])
            except Exception as e:
                print(f"WARNING: {e}")

        return ner_doc
