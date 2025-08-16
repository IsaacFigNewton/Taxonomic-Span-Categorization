import json
from nltk.corpus import wordnet as wn

class TaxonomyValidator:
    def __init__(self):
        pass

    def validate_synset(self, synset_str):
        try:
            lemma, pos, num = synset_str.split('.')
            return wn.synset(synset_str).definition()
        except Exception as e:
            return None

    # Walk through taxonomy recursively
    def validate_taxonomy(self, node):
        results = {}
        if "wordnet_synsets" in node:
            syns = {}
            for s in node["wordnet_synsets"]:
                syns[s] = self.validate_synset(s)
            results["wordnet_synsets"] = syns
        if "children" in node:
            child_results = {}
            for key, child in node["children"].items():
                child_results[key] = self.validate_taxonomy(child)
            results["children"] = child_results
        return results