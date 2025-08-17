# Taxonomy Specification

This document describes how to create and structure taxonomies for the SpanCategorizer system.

## Overview

Taxonomies define the hierarchical category structure used for semantic classification. They are JSON files that describe categories, their relationships, and semantic information used for embedding generation. The taxonomy format uses WordNet synsets as keys to provide semantic grounding for each category level.

## Basic Structure

Taxonomies are nested JSON objects where each key is a WordNet synset identifier and each value contains category information:

```json
{
  "synset.pos.nn": {
    "label": "Category_Display_Name",
    "children": {
      "child_synset.pos.nn": {
        "label": "Subcategory_Name",
        "children": {
          // Further nested subcategories or leaf nodes
        }
      }
    }
  }
}
```

## Required Fields

### Synset Keys
Category keys must be valid WordNet synset identifiers in the format `"word.part_of_speech.sense_number"`:
- `word`: The lemma (e.g., "person", "organization")
- `part_of_speech`: `n` (noun), `v` (verb), `a` (adjective), `r` (adverb)
- `sense_number`: Numeric identifier (01, 02, etc.)

**Example:**
```json
{
  "person.n.01": { /* category definition */ },
  "organization.n.01": { /* category definition */ }
}
```

### Category Fields

Each category within the taxonomy should contain:

#### `label` (required)
A human-readable display name for the category. This is the name that will be used for classification results and user interfaces.

**Guidelines:**
- Use **descriptive names** that clearly indicate the category purpose
- Follow **consistent naming conventions** (e.g., Pascal_Case, UPPER_CASE)
- Make labels **intuitive** for end users
- For leaf nodes, labels often correspond to NER tag names (e.g., "PERSON", "ORG", "GPE")

**Example:**
```json
"label": "Individual_People"
```

#### `children` (optional for leaf nodes)
Nested subcategories following the same synset-based structure. Leaf nodes (final categories) may omit the `children` field entirely.

**Example:**
```json
{
  "person.n.01": {
    "label": "Individual_People",
    "children": {
      "individual.n.01": {
        "label": "PERSON"
      }
    }
  }
}
```

## Semantic Hierarchy Design

### WordNet Integration
The taxonomy leverages WordNet's semantic hierarchy by using synsets as keys. This provides:
- **Semantic grounding** for each category level
- **Consistent hierarchical relationships** based on WordNet's hypernym/hyponym structure
- **Built-in semantic similarity** for embedding generation

### Finding Appropriate Synsets
```python
import nltk
from nltk.corpus import wordnet as wn

# Find synsets for a concept
synsets = wn.synsets('person')
for syn in synsets:
    print(f"{syn.name()}: {syn.definition()}")
    
# Explore hypernym hierarchy
person_syn = wn.synset('person.n.01')
print("Hypernyms:", person_syn.hypernyms())
print("Hyponyms:", person_syn.hyponyms())
```

### Hierarchy Structure
The taxonomy follows a top-down approach from general to specific:

1. **Root Level**: Fundamental WordNet categories (e.g., `physical_entity.n.01`, `abstraction.n.06`)
2. **Intermediate Levels**: Progressively more specific semantic categories
3. **Leaf Nodes**: Final classification labels, often corresponding to NER tags

**Example hierarchy:**
```
physical_entity.n.01 → causal_agent.n.01 → person.n.01 → individual.n.01 (PERSON)
```

## Design Guidelines

### Hierarchy Depth
- **Shallow hierarchies** (2-3 levels): Faster processing, broader categories
- **Deep hierarchies** (4+ levels): More specific categorization, may be slower
- **Current format**: Typically 4-5 levels from root synset to final label
- **Recommendation**: Balance semantic precision with processing efficiency

### Category Labeling
- **Intermediate levels**: Use descriptive, semantic category names (e.g., "Individual_People", "Formal_Organizations")
- **Leaf nodes**: Use standard NER tag conventions (e.g., "PERSON", "ORG", "GPE", "MONEY")
- **Consistency**: Maintain consistent naming patterns within semantic groups

### Synset Selection
- Choose **appropriate semantic level** synsets that represent meaningful distinctions
- Ensure **WordNet compatibility** - verify synsets exist in your WordNet version
- Use **hypernym relationships** to create logical hierarchical flow
- Balance **specificity** with **coverage** for your domain

### Semantic Coverage
- Ensure **comprehensive coverage** of your target entity types
- Map **standard NER categories** to appropriate WordNet semantic paths
- Include **domain-specific** categories as needed
- Test with **representative text samples** from your domain

## Complete Example Structure

Based on the provided SpaCy_NER.json, here's the general pattern:

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
          },
          "social_group.n.01": {
            "label": "Social_Groups",
            "children": {
              "organization.n.01": {
                "label": "Formal_Organizations",
                "children": {
                  "corporate_body.n.01": {
                    "label": "ORG"
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## Troubleshooting

**Invalid synsets:**
- Symptoms: Errors loading taxonomy, missing synset definitions
- Solution: Verify synset identifiers exist in WordNet, check spelling/format

**Semantic misalignment:**
- Symptoms: Unexpected classification results, poor semantic matching
- Solution: Review synset choices, ensure they align with intended semantics

**Hierarchy inconsistency:**
- Symptoms: Illogical parent-child relationships in results
- Solution: Follow WordNet hypernym/hyponym relationships more closely

**Missing coverage:**
- Symptoms: Entities don't match any category
- Solution: Add appropriate synset paths for missing entity types

## Validation

### Synset Validation
```python
from nltk.corpus import wordnet as wn

def validate_synset(synset_name):
    try:
        synset = wn.synset(synset_name)
        return True, synset.definition()
    except:
        return False, f"Invalid synset: {synset_name}"
```

### Hierarchy Validation
Ensure parent-child synset relationships follow WordNet semantic hierarchy where possible, though domain-specific deviations may be necessary for classification effectiveness.