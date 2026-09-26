"""Text Normalization & Cleaning Module for Entity Resolution.

Assigned to: Person 1 (blocking & normalization)
"""

import re
import string


# Legal suffix normalization mapping
LEGAL_SUFFIXES = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bco\b": "company",
    r"\bsarl\b": "sarl",
}

# Common address abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bpo box\b": "pobox",
}


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove special characters, trim whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: str) -> str:
    """Normalize business names by handling legal suffixes, abbreviations, and noise."""
    cleaned = clean_text(name)
    for pattern, repl in LEGAL_SUFFIXES.items():
        cleaned = re.sub(pattern, repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_address(address: str) -> str:
    """Normalize address fields by expanding standard street abbreviations."""
    cleaned = clean_text(address)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
