"""Text Normalization & Cleaning Module for Entity Resolution.

Assigned to: Person 1 (blocking & normalization)
Handles: Accents, Diacritics, Legal Suffixes, DBAs, Web Domains, Address Abbreviations, Brackets.
"""

import re
import string
import unicodedata
from typing import Tuple

# Legal terms and suffixes to strip/standardize
LEGAL_PATTERNS = [
    (r"\b(d\.?b\.?a\.?|doing business as)\b.*", ""),  # Strip DBA clauses
    (r"\b(private limited|pvt ltd|pvt\.? ltd\.?)\b", "pvt_ltd"),
    (r"\b(limited liability company|l\.?l\.?c\.?|llc)\b", "llc"),
    (r"\b(limited liability partnership|l\.?l\.?p\.?|llp)\b", "llp"),
    (r"\b(incorporated|inc\.?)\b", "inc"),
    (r"\b(corporation|corp\.?)\b", "corp"),
    (r"\b(limited|ltd\.?)\b", "ltd"),
    (r"\b(company|co\.?)\b", "co"),
    (r"\b(sarl|s\.?a\.?r\.?l\.?)\b", "sarl"),
]

# Address abbreviations map
ADDRESS_ABBREVIATIONS = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bfl\.?\b": "floor",
    r"\bpo box\b": "pobox",
    r"\bnd\b": "th",  # Fix 45nd -> 45th typo
}


def strip_accents(text: str) -> str:
    """Remove diacritics and accents (e.g., Énterprises -> Enterprises, Bóral -> Boral)."""
    if not text:
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_text(text: str) -> str:
    """Basic text cleaning: accent stripping, lowercasing, bracket removal, whitespace collapse."""
    if not text or pd_isna(text):
        return ""
    text = str(text)
    # Strip accents
    text = strip_accents(text)
    # Lowercase
    text = text.lower().strip()
    # Strip brackets [[...]], (...)
    text = re.sub(r"[\[\]\(\)\{\}]", " ", text)
    # Strip web domain extensions (.com, .org, .net, .in, .co)
    text = re.sub(r"\.(com|org|net|co\.in|in|info|biz|io)\b", " ", text)
    # Replace punctuation with spaces (preserve alphanumeric and non-ASCII script)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def pd_isna(val) -> bool:
    """Safely check for NaN/None/null string values."""
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "<null>")


def normalize_business_name(name: str) -> Tuple[str, str]:
    """Normalize business name and extract legal suffix.

    Returns:
        Tuple[str, str]: (cleaned_name_without_legal, extracted_legal_suffix)
    """
    cleaned = clean_text(name)
    extracted_legal = ""

    for pattern, legal_tag in LEGAL_PATTERNS:
        if re.search(pattern, cleaned):
            if legal_tag:
                extracted_legal = legal_tag
            cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, extracted_legal


def normalize_address(address: str) -> str:
    """Normalize address fields by expanding standard street abbreviations."""
    if pd_isna(address):
        return ""
    cleaned = clean_text(address)
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

