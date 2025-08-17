# SpanCategorizer

A hierarchical semantic search module for Named Entity Recognition (NER) that uses embedding-based similarity matching to classify text spans according to custom **WordNet-aligned taxonomies**.

## Overview

SpanCategorizer performs intelligent text span classification by combining:
- **Semantic embeddings** (via SentenceTransformers or spaCy)
- **Hierarchical, WordNet-based taxonomies** (JSON category trees)
- **WordNet synsets integration** for semantic expansion
- **Cosine similarity matching** for best categorical fits

The system recursively searches through taxonomic hierarchies (defined by synsets) to find the most semantically appropriate labels for noun chunks in text.

## Features

- **Hierarchical semantic search** through WordNet-aligned taxonomies
- **Multiple embedding models** (SentenceTransformers, spaCy)
- **WordNet synset expansion** for richer semantic understanding  
- **Configurable similarity thresholds** for precision control
- **SpaCy integration** for seamless NLP pipeline integration
- **Flexible taxonomy loading** with JSON-based hierarchical format

## Installation

```bash
pip install git+https://github.com/IsaacFigNewton/Taxonomic-Span-Categorization.git
````

## Quick Start

```python
from tax_span_cat import SpanCategorizer
import spacy

# Initialize the categorizer
categorizer = SpanCategorizer(
    embedding_model="all-MiniLM-L6-v2",
    threshold=0.5
)

# Load spaCy model and add categorizer to pipeline
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("Apple Inc. released the new iPhone in California.")
categorized_doc = categorizer(doc)

# Access categorized spans
for label, spans in categorized_doc.spans.items():
    print(f"{label}: {[span.text for span in spans]}")
```

## Configuration

### Parameters

* **`embedding_model`** (str): Model name for SentenceTransformers or spaCy model

  * Default: `"all-MiniLM-L6-v2"`
  * Alternatives: `"all-mpnet-base-v2"`, `"en_core_web_lg"`, etc.

* **`taxonomy_path`** (str|None): Path to custom taxonomy JSON file

  * Default: Uses built-in WordNet-aligned taxonomy (`SpaCy_NER.json`)

* **`threshold`** (float): Minimum cosine similarity for category assignment

  * Default: `0.5`
  * Range: `0.0` to `1.0`

* **`taxonomic_features`** (List\[str]): Features to include in embeddings

  * Default: `["description", "wordnet_synsets"]`

### Example Configuration

```python
categorizer = SpanCategorizer(
    embedding_model="all-mpnet-base-v2",
    taxonomy_path="./SpaCy_NER.json",
    threshold=0.7,
    taxonomic_features=["description", "wordnet_synsets", "examples"]
)
```

## Taxonomy Format

Taxonomies are JSON objects structured around **WordNet synset IDs** as keys.
Each node contains:

* **`label`**: Human-readable category label
* **`children`**: Nested subcategories, keyed by their WordNet synset

### Example

```json
{
  "physical_entity.n.01": {
    "label": "Physical_Entities",
    "children": {
      "causal_agent.n.01": {
        "label": "Agents_and_Actors",
        "children": {
          "person.n.01": {
            "label": "Individual_People",
            "children": {
              "individual.n.01": {
                "label": "PERSON"
              }
            }
          }
        }
      }
    }
  }
}
```

➡️ See [TAXONOMY.md](TAXONOMY.md) for full details.

## API Reference

### SpanCategorizer Class

#### `__init__(embedding_model, taxonomy_path, threshold, taxonomic_features)`

Initialize the categorizer with specified configuration.

#### `__call__(doc: spacy.Doc) -> spacy.Doc`

Process a spaCy document and return it with categorized spans.

**Parameters:**

* `doc`: spaCy Doc object to process

**Returns:**

* spaCy Doc object with populated `.spans` and `.ents` attributes

### Key Methods

* **`_hierarchical_sem_search(query, current_label, current_node)`**
  Recursive search through taxonomy.

* **`_semantic_search(query_vect, corpus_vects)`**
  Find best match using cosine similarity.

* **`_embed(text)`**
  Generate normalized embeddings for text.

## Examples

### Using Custom Taxonomy

```python
# Use custom WordNet-based taxonomy
categorizer = SpanCategorizer(taxonomy_path="custom_taxonomy.json")
```

### Batch Processing

```python
texts = [
    "Microsoft released Windows 11.",
    "The CEO of Tesla spoke at the conference.",
    "Scientists discovered a new species in the Amazon."
]

for text in texts:
    doc = nlp(text)
    categorized_doc = categorizer(doc)
    
    print(f"Text: {text}")
    for label, spans in categorized_doc.spans.items():
        if spans:  # Only show labels with spans
            print(f"  {label}: {[s.text for s in spans]}")
    print()
```

## Performance Considerations

* **Model Choice**: `all-MiniLM-L6-v2` is faster, `all-mpnet-base-v2` is more accurate
* **Threshold Tuning**: Higher thresholds reduce false positives but may miss valid matches
* **Taxonomy Depth**: Deeper hierarchies provide more specificity but slower processing
* **Caching**: Embeddings are precomputed for efficiency