#!/usr/bin/env python3
"""
Systematic Notebook Debugging Framework
=====================================

This framework provides systematic testing of Jupyter notebooks with:
1. Sequential cell execution with state persistence
2. Error capture and analysis
3. Variable state verification
4. Clear success/failure criteria
5. Detailed debugging output
"""

import json
import sys
import traceback
from typing import Dict, List, Any, Optional, Tuple
import importlib.util
import subprocess
import os

class NotebookDebugger:
    """Systematic debugger for Jupyter notebooks"""
    
    def __init__(self, notebook_path: str):
        self.notebook_path = notebook_path
        self.cell_results = []
        self.global_namespace = {}
        self.errors = []
        
    def load_notebook(self) -> List[Dict]:
        """Load notebook cells from file"""
        try:
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            return notebook.get('cells', [])
        except Exception as e:
            raise RuntimeError(f"Failed to load notebook: {e}")
    
    def execute_cell(self, cell: Dict, cell_index: int) -> Tuple[bool, Any, str]:
        """Execute a single cell and return (success, result, error)"""
        cell_type = cell.get('cell_type', 'unknown')
        
        if cell_type == 'markdown':
            return True, "Markdown cell skipped", ""
        
        if cell_type != 'code':
            return True, f"{cell_type} cell skipped", ""
        
        # Get source code
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = str(source)
        
        if not code.strip():
            return True, "Empty cell skipped", ""
        
        print(f"\n--- Executing Cell {cell_index} ---")
        print(f"Code:\n{code[:200]}{'...' if len(code) > 200 else ''}")
        
        try:
            # Execute in the persistent global namespace
            exec(code, self.global_namespace)
            
            # Check for key variables that should exist
            success_indicators = self._check_expected_variables(cell_index)
            
            return True, success_indicators, ""
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return False, None, error_msg
    
    def _check_expected_variables(self, cell_index: int) -> Dict[str, bool]:
        """Check if expected variables exist after cell execution"""
        expected_vars = {
            0: [],  # Markdown
            1: [],  # Markdown  
            2: [],  # Markdown
            3: ['IN_COLAB', 'sys', 'os', 'warnings'],  # Environment setup
            4: [],  # Keras setup
            5: [],  # Markdown
            6: ['json', 'spacy', 'SpanCategorizer', 'nltk'],  # Basic imports
            7: ['general_ner', 'load_taxonomy'],  # Taxonomy loading
            8: ['nlp', 'ner'],  # SpaCy and SpanCategorizer
        }
        
        checks = {}
        for var in expected_vars.get(cell_index, []):
            checks[var] = var in self.global_namespace
            
        return checks
    
    def run_systematic_test(self) -> Dict[str, Any]:
        """Run complete systematic test of the notebook"""
        print("=" * 60)
        print("SYSTEMATIC NOTEBOOK DEBUGGING FRAMEWORK")
        print("=" * 60)
        
        cells = self.load_notebook()
        print(f"Loaded {len(cells)} cells from notebook")
        
        results = {
            'total_cells': len(cells),
            'executed_cells': 0,
            'successful_cells': 0,
            'failed_cells': 0,
            'cell_details': [],
            'final_test_passed': False
        }
        
        # Execute cells sequentially
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                results['executed_cells'] += 1
                
                success, result, error = self.execute_cell(cell, i)
                
                cell_result = {
                    'cell_index': i,
                    'success': success,
                    'result': result,
                    'error': error
                }
                
                results['cell_details'].append(cell_result)
                
                if success:
                    results['successful_cells'] += 1
                    print(f"[SUCCESS] Cell {i}")
                    if isinstance(result, dict):
                        for var, exists in result.items():
                            print(f"  - {var}: {'OK' if exists else 'MISSING'}")
                else:
                    results['failed_cells'] += 1
                    print(f"[FAILED] Cell {i}")
                    print(f"  Error: {error[:200]}...")
                    self.errors.append((i, error))
                    break  # Stop on first failure
        
        # Final functionality test
        if results['failed_cells'] == 0:
            results['final_test_passed'] = self._final_functionality_test()
        
        return results
    
    def _final_functionality_test(self) -> bool:
        """Test final SpanCategorizer functionality"""
        print("\n--- Final Functionality Test ---")
        
        try:
            # Check required variables exist
            required_vars = ['nlp', 'ner', 'general_ner']
            for var in required_vars:
                if var not in self.global_namespace:
                    print(f"[MISSING] Required variable: {var}")
                    return False
            
            # Extract from global namespace
            nlp = self.global_namespace['nlp']
            ner = self.global_namespace['ner']
            
            # Test on simple example
            test_text = "Tim Berners-Lee works at MIT in Boston."
            print(f"Testing: {test_text}")
            
            doc = nlp(test_text)
            ner_doc = ner(doc)
            
            entities_found = len(ner_doc.ents)
            labels = [ent.label_ for ent in ner_doc.ents]
            specific_labels = [l for l in labels if l != 'ENTITY']
            
            print(f"Entities found: {entities_found}")
            print(f"Labels: {labels}")
            print(f"Specific labels: {len(specific_labels)}/{entities_found}")
            
            # Success criteria
            if entities_found > 0 and len(specific_labels) > 0:
                print("[SUCCESS] Final test - Getting specific taxonomic labels")
                return True
            else:
                print("[FAILED] Final test - Only generic labels or no entities")
                return False
                
        except Exception as e:
            print(f"[FAILED] Final test - {e}")
            return False

def main():
    """Main debugging function"""
    notebook_path = r"C:\Users\igeek\OneDrive\Documents\GitHub\Taxonomic-Span-Categorization\Test_Taxonomic_NER.ipynb"
    
    debugger = NotebookDebugger(notebook_path)
    results = debugger.run_systematic_test()
    
    print("\n" + "=" * 60)
    print("DEBUGGING SUMMARY")
    print("=" * 60)
    print(f"Total cells: {results['total_cells']}")
    print(f"Code cells executed: {results['executed_cells']}")  
    print(f"Successful: {results['successful_cells']}")
    print(f"Failed: {results['failed_cells']}")
    print(f"Final functionality test: {'PASSED' if results['final_test_passed'] else 'FAILED'}")
    
    if results['failed_cells'] > 0:
        print(f"\nFirst failure details:")
        for cell_idx, error in debugger.errors:
            print(f"Cell {cell_idx}: {error}")
            break
    
    return results

if __name__ == "__main__":
    main()