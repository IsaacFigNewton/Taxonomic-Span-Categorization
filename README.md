# SpanCategorizer

A hierarchical semantic search module for Named Entity Recognition (NER) that uses embedding-based similarity matching to classify text spans according to custom taxonomies.

## Overview

SpanCategorizer performs intelligent text span classification by combining:
- **Semantic embeddings** (via SentenceTransformers or spaCy)
- **Hierarchical taxonomies** (JSON-based category trees)
- **WordNet integration** (for semantic expansion)
- **Cosine similarity matching** (for finding best categorical fits)

The system recursively searches through taxonomic hierarchies to find the most semantically appropriate labels for noun chunks in text.

## Features

- **Hierarchical semantic search** through custom taxonomies
- **Multiple embedding models** (SentenceTransformers, spaCy)
- **WordNet synset expansion** for richer semantic understanding  
- **Configurable similarity thresholds** for precision control
- **SpaCy integration** for seamless NLP pipeline integration
- **Flexible taxonomy loading** with fallback support

## Installation

```bash
pip install git+https://github.com/IsaacFigNewton/Taxonomic-Span-Categorization.git
```

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
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("span_categorizer", last=True, config={"categorizer": categorizer})

# Process text
doc = nlp("Apple Inc. released the new iPhone in California.")
categorized_doc = categorizer(doc)

# Access categorized spans
for label, spans in categorized_doc.spans.items():
    print(f"{label}: {[span.text for span in spans]}")
```

## Configuration

### Parameters

- **`embedding_model`** (str): Model name for SentenceTransformers or spaCy model
  - Default: `"all-MiniLM-L6-v2"`
  - Alternatives: `"all-mpnet-base-v2"`, `"en_core_web_lg"`, etc.

- **`taxonomy_path`** (str|None): Path to custom taxonomy JSON file
  - Default: Uses built-in SpaCy NER taxonomy

- **`threshold`** (float): Minimum cosine similarity for category assignment
  - Default: `0.5`
  - Range: `0.0` to `1.0`

- **`taxonomic_features`** (List[str]): Features to include in embeddings
  - Default: `["description", "wordnet_synsets"]`

### Example Configuration

```python
categorizer = SpanCategorizer(
    embedding_model="all-mpnet-base-v2",
    taxonomy_path="./custom_taxonomy.json",
    threshold=0.7,
    taxonomic_features=["description", "wordnet_synsets", "examples"]
)
```

## Taxonomy Format

For detailed information on creating and structuring taxonomies, see [TAXONOMY.md](TAXONOMY.md).

## API Reference

### SpanCategorizer Class

#### `__init__(embedding_model, taxonomy_path, threshold, taxonomic_features)`
Initialize the categorizer with specified configuration.

#### `__call__(doc: spacy.Doc) -> spacy.Doc`
Process a spaCy document and return it with categorized spans.

**Parameters:**
- `doc`: spaCy Doc object to process

**Returns:**
- spaCy Doc object with populated `.spans` and `.ents` attributes

### Key Methods

#### `_hierarchical_sem_search(query, current_label, current_node)`
Perform hierarchical semantic search through taxonomy.

#### `_semantic_search(query_vect, corpus_vects)`
Find best matching vector using cosine similarity.

#### `_embed(text)`
Generate normalized embeddings for input text.

## Examples

### Custom Taxonomy

See [TAXONOMY.md](docs/TAXONOMY.md) for detailed examples of creating custom taxonomies.

```python
# Use custom taxonomy
categorizer = SpanCategorizer(taxonomy_path="tech_taxonomy.json")
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

- **Model Choice**: `all-MiniLM-L6-v2` is faster, `all-mpnet-base-v2` is more accurate
- **Threshold Tuning**: Higher thresholds reduce false positives but may miss valid matches
- **Taxonomy Depth**: Deeper taxonomies provide more specificity but slower processing
- **Caching**: Embeddings are computed once during initialization for efficiency
