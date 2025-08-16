# Taxonomy Specification

This document describes how to create and structure taxonomies for the SpanCategorizer system.

## Overview

Taxonomies define the hierarchical category structure used for semantic classification. They are JSON files that describe categories, their relationships, and semantic information used for embedding generation.

## Basic Structure

Taxonomies are nested JSON objects with the following required structure:

```json
{
  "children": {
    "CATEGORY_NAME": {
      "description": "Category description text",
      "wordnet_synsets": ["synset.pos.nn"],
      "children": {
        // Nested subcategories follow the same structure
      }
    }
  }
}
```

## Required Fields

### `children`
A dictionary containing subcategories. Every taxonomy node must have a `children` field, even if it's empty (`{}`).

**Example:**
```json
{
  "children": {
    "PERSON": { /* category definition */ },
    "ORGANIZATION": { /* category definition */ }
  }
}
```

### Category Fields

Each category within `children` should contain:

#### `description` (optional)
A text description of the category. This is embedded and used for semantic matching.

**Guidelines:**
- Be descriptive but concise
- Use natural language
- Include key distinguishing characteristics
- Avoid overly technical jargon

**Example:**
```json
"description": "Human beings including real people and fictional characters"
```

#### `wordnet_synsets` (optional but recommended)
A list of WordNet synset identifiers that expand the semantic understanding of the category.

**Format:** `"word.part_of_speech.sense_number"`
- `word`: The lemma
- `part_of_speech`: `n` (noun), `v` (verb), `a` (adjective), `r` (adverb)
- `sense_number`: Numeric identifier (01, 02, etc.)

**Example:**
```json
"wordnet_synsets": ["person.n.01", "individual.n.01", "human.n.01"]
```

**Finding Synsets:**
```python
import nltk
from nltk.corpus import wordnet as wn

# Find synsets for a word
synsets = wn.synsets('person')
for syn in synsets:
    print(f"{syn.name()}: {syn.definition()}")
```

#### `children` (required)
Nested subcategories following the same structure. Use empty dictionary `{}` for leaf nodes.
For a complete example, see `src/tax_span_cat/taxonomies/SpaCy_NER.json`.

## Design Guidelines

### Hierarchy Depth
- **Shallow hierarchies** (2-3 levels): Faster processing, broader categories
- **Deep hierarchies** (4+ levels): More specific categorization, slower processing
- **Recommendation**: Start with 3-4 levels maximum, expand as needed

### Category Naming
- Use **UPPER_CASE** for category names
- Be **consistent** with naming conventions
- Use **descriptive** names that clearly indicate the category purpose
- Avoid **ambiguous** or **overlapping** category names

### Semantic Coverage
- Ensure **comprehensive coverage** of your domain
- Avoid **semantic gaps** between sibling categories  
- Include **sufficient synsets** for robust matching
- Test with **representative text samples** from your domain

### Balancing Specificity
- **Too broad**: Poor discrimination between different entity types
- **Too specific**: May miss valid entities due to narrow definitions
- **Sweet spot**: Categories that capture meaningful distinctions in your domain

## Taxonomy Validation

### TaxonomyValidator (coming soon)


## Advanced Features

### Custom Taxonomic Features
You can extend taxonomies with custom features by modifying the `taxonomic_features` parameter:

```python
categorizer = SpanCategorizer(
    taxonomic_features=["description", "wordnet_synsets", "examples", "aliases"]
)
```

Then include these fields in your taxonomy:
```json
{
  "description": "Technology companies and products",
  "wordnet_synsets": ["technology.n.01"],
  "examples": ["Apple, Microsoft, Google, Amazon"],
  "aliases": ["tech company", "software company", "IT firm"],
  "children": {}
}
```

## Troubleshooting

**Categories too broad:**
- Symptoms: Many unrelated entities get the same label
- Solution: Add more specific subcategories

**Categories too narrow:**
- Symptoms: Entities don't match any category (fall back to parent)
- Solution: Broaden descriptions or add more synsets

**Poor semantic coverage:**
- Symptoms: Inconsistent classification of similar entities
- Solution: Add more comprehensive synsets and examples

**Hierarchy imbalance:**
- Symptoms: Some branches much deeper than others
- Solution: Restructure for more balanced depth