from importlib.resources import files
from typing import Dict, List, Tuple
import json
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

import spacy
from spacy.tokens import Doc, Span

from sentence_transformers import SentenceTransformer

import tax_span_cat.taxonomies

class SpanCategorizer:
    default_taxonomy_path = files(tax_span_cat.taxonomies).joinpath(f"SpaCy_NER.json")

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 taxonomy_path: str|None = None,
                 threshold: float = 0.5):
        self._init_embedding_model(embedding_model)
        self._init_taxonomy(taxonomy_path)
        # similarity threshold for assigning a particular label
        self.threshold = threshold
    
    
    def _init_embedding_model(self, model_name: str):
        """Initialize sentence embedding model"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Failed to load embedding model: {e}, loading SpaCy en_core_web_lg backup.")
            self.embedding_model = spacy.load("en_core_web_lg")
    

    def _init_taxonomy(self, taxonomy_path: str|None):
        """Initialize the embedding taxonomy"""
        if not taxonomy_path:
            taxonomy_path = self.default_taxonomy_path
        # load the taxonomy
        self.taxonomy = self._load_taxonomy_from_path(taxonomy_path)
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
            for label, subtree in node.items():
                new_node[label] = self._embed_taxonomy(subtree)
                subtree_centroids.append(new_node[label]["embedding"])
            # centroid = mean of normalized child embeddings
            new_node["embedding"] = np.mean(np.array(subtree_centroids), axis=0)
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
        # this either means there's a bug in this code or in the taxonomy
        if "children" not in current_node.keys():
            raise KeyError(f"'children' were missing from the current node being checked, '{current_label}'.")

        # get an ordered list of the labelled embeddings (excluding the embedding key itself)
        children = [key for key in current_node["children"].keys() if key != "embedding"]
        
        # if there are no valid children (only embedding keys), return current label
        if not children:
            return current_label
            
        # get query embedding
        query_vect = self._embed(query)
        # get embeddings of taxonomic terms at the current level
        corpus_vects = [current_node["children"][child]["embedding"] for child in children]
        # get idx of closest match
        best_similarity, best_match_idx = self._semantic_search(query_vect, corpus_vects)
        # get label of closest match
        best_match_label = children[best_match_idx]

        print(f"Best match for '{query}' is '{best_match_label}' with similarity of {best_similarity}")
        if best_similarity <= self.threshold:
            return current_label
        else:
            return self._hierarchical_sem_search(
                query=query,
                current_label=best_match_label,
                current_node=current_node["children"][best_match_label]
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
                current_node=self.taxonomy,
            )
            # if this label isn't in the spans list, add it
            if ent_label not in ner_doc.spans.keys():
                ner_doc.spans[ent_label] = list()
            span = Span(
                ner_doc,
                start=chunk.start,
                end=chunk.end,
                label=ent_label
            )
            # add the span to the doc's spans and ents
            ner_doc.spans[ent_label].append(span)
            ner_doc.set_ents(list(ner_doc.ents) + [span])
        
        return ner_doc
