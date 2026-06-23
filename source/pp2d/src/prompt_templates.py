# prompt_templates.py

"""
Prompt Templates for Pocket-Aware Language Model (PP2D) - Listwise Edition

This module loads diverse prompt templates from JSON files.
Templates use placeholders that will be replaced with protein token blocks:
- {ACT_BLOCK}: Will be replaced with a sequence of <protein> tokens for active pockets
- {INACT_BLOCK}: Will be replaced with a sequence of <protein> tokens for inactive pockets

Template Format:
Each template is a simple string with placeholders.
Example: "Design a molecule for {ACT_BLOCK} that avoids {INACT_BLOCK}."
"""

import json
import os
from typing import Dict, List, Union
from pathlib import Path

# Global cache for templates
_template_cache = None

def _get_template_file_path() -> str:
    """Get the path to the template JSON file."""
    # Look for template file in data directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Prefer the new listwise templates
    template_path = project_root / "data" / "prompt_templates_listwise.json"
    
    if not template_path.exists():
        # Fallback to default name if listwise specific not found
        template_path = project_root / "data" / "prompt_templates.json"
        
    if not template_path.exists():
        # Fallback: look in current directory
        template_path = current_dir / "prompt_templates.json"
        
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found. Looked for prompt_templates_listwise.json in data/")
    
    return str(template_path)

def _load_templates() -> Dict:
    """Load templates from JSON file with caching."""
    global _template_cache
    
    if _template_cache is None:
        template_file = _get_template_file_path()
        with open(template_file, 'r', encoding='utf-8') as f:
            _template_cache = json.load(f)
    
    return _template_cache

def get_templates(task_type: str = "listwise") -> List[str]:
    """
    Get prompt templates for a specific task type.
    
    Args:
        task_type: 'listwise' (default) or 'listwise_active_only'
    
    Returns:
        List of prompt template strings
    """
    templates_data = _load_templates()
    templates = templates_data.get('templates', {})
    
    if task_type not in templates:
        # Fallback to 'listwise' if specific type not found, or empty list
        return templates.get('listwise', [])
    
    return templates[task_type]

def get_all_templates() -> Dict[str, List[str]]:
    """Get all templates organized by task type."""
    templates_data = _load_templates()
    return templates_data.get('templates', {})

def reload_templates():
    """Force reload templates from file."""
    global _template_cache
    _template_cache = None
