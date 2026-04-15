"""Concept name normalisation for consistent graph node IDs."""
import re


def normalize_concept_name(name: str) -> str:
    """Normalise a concept name for use as a graph node ID suffix.

    Applies:
    1. Strip leading/trailing whitespace
    2. Lowercase
    3. Replace runs of whitespace with a single underscore
    4. Remove characters that are not alphanumeric or underscore
    5. Collapse multiple consecutive underscores to one
    6. Strip leading/trailing underscores

    Example: "Self-Attention (QKV)" -> "self_attention_qkv"
    """
    name = name.strip().lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w]', '_', name)        # \w = [a-zA-Z0-9_]
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name
