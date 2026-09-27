"""Text Normalization & Cleaning Module for Entity Resolution.

Assigned to: Person 1 (blocking & normalization)
Handles: Accents, Diacritics, Legal Suffixes, DBAs, Web Domains, Address Abbreviations,
Unicode punctuation, and **non-Latin script safety** (Devanagari/Tamil/etc. vowel signs
must survive — see ``strip_accents``).

Ordering matters in this module: DBA markers and street-type abbreviations are resolved
on the *raw* string (where punctuation such as ``D.B.A.``, ``St,``, ``P.O. Box`` is still
visible), and only then is punctuation stripped. Stripping punctuation first (the old
behaviour) glued ``D.B.A.`` into ``d b a`` and made those rules dead code.
"""

import re
import unicodedata
from typing import Dict, List, Tuple

# Legal suffixes to strip and standardise. Ordered most-specific first, because
# ``normalize_business_name`` keeps the FIRST tag it matches (e.g. "pvt ltd" -> pvt_ltd,
# not ltd). DBA markers are handled separately in _DBA_RE (they are not a legal suffix —
# the trade name after them must be kept, not tagged).
LEGAL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(private limited|pvt\s+ltd|pvt)\b", "pvt_ltd"),
    (r"\b(limited liability company|l\s*l\s*c)\b", "llc"),
    (r"\b(limited liability partnership|l\s*l\s*p)\b", "llp"),
    (r"\b(incorporated|inc)\b", "inc"),
    (r"\b(corporation|corp)\b", "corp"),
    (r"\b(limited|ltd)\b", "ltd"),
    (r"\b(company|co)\b", "co"),
    (r"\b(sarl|s\s*a\s*r\s*l)\b", "sarl"),
]

# "doing business as" written as DBA / D.B.A. / D/B/A / doing business as.
_DBA_RE = re.compile(r"\b(?:d\s*\.?\s*b\s*\.?\s*a\s*\.?|d\s*/\s*b\s*/\s*a|doing\s+business\s+as)\b",
                     re.IGNORECASE)

# Web domain suffixes, removed before punctuation stripping (else ".com" leaves "com").
_DOMAIN_RE = re.compile(r"\.(?:com|org|net|co\.in|in|info|biz|io)\b", re.IGNORECASE)

# Legal-suffix words stripped from blocking/vectorizer text (fast path, no tag capture).
_LEGAL_SUFFIX_WORDS_RE = re.compile(
    r"\b(llc|ltd|inc|corp|co|plc|gmbh|llp|lp|sa|srl|sl|bv|nv|ag|oy|ab|as|pte|pvt|sas|kk|kg)\b\.?")

# Address street types expanded only when they are a *suffix* (followed by a comma or the
# end of the string). Without the lookahead "St Mary Road" (saint) became "street mary road".
_STREET_SUFFIX_RE = re.compile(
    r"\b(rd|st|ave|blvd|dr|ln|lane|ct|court|pl|place|hwy|sq|ter|trl)\b\.?(?=\s*(?:,|$))",
    re.IGNORECASE)
_UNIT_RE = re.compile(r"\b(apt|ste|fl|floor|suite|apartment)\b\.?", re.IGNORECASE)
_PO_BOX_RE = re.compile(r"\bp\s*\.?\s*o\s*\.?\s*box\b", re.IGNORECASE)
_ORDINAL_TYPO_RE = re.compile(r"(?<=\d)\s*nd\b", re.IGNORECASE)   # 45nd -> 45th
_BRACKETS_RE = re.compile(r"[\[\]\(\)\{\}]")
_WS_RE = re.compile(r"\s+")

# Anything outside Basic Latin .. Latin Extended-B / Latin Extended Additional (plus
# whitespace) is treated as non-Latin: NFKD folding would delete essential vowel signs.
_NON_LATIN_RE = re.compile(r"[^\x00-\u024F\u1E00-\u1EFF\s]")

STREET_TYPE_MAP: Dict[str, str] = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "lane": "lane", "ct": "court", "court": "court", "pl": "place",
    "place": "place", "hwy": "highway", "sq": "square", "ter": "terrace", "trl": "trail",
}
_UNIT_MAP: Dict[str, str] = {
    "apt": "apartment", "apartment": "apartment", "ste": "suite", "suite": "suite",
    "fl": "floor", "floor": "floor",
}


def _is_latin_text(text: str) -> bool:
    """True when ``text`` only contains Latin letters/digits/punctuation/space.

    Implemented as a single C-level regex scan (a Python char loop over 12M records is
    measurably slower) — anything outside Basic Latin .. Latin Extended-B/Additional or
    Extended Additional counts as non-Latin.
    """
    return _NON_LATIN_RE.search(text) is None


def _build_punctuation_table() -> Dict[int, str]:
    """Map every Unicode punctuation/symbol/control code point to a space (once, at import).

    ``str.translate`` keeps this C-level fast. Unlike ``re.sub(r"[^\\w\\s]", " ")`` it
    leaves **combining marks** (category ``M``) alone, which is what makes Devanagari and
    Tamil text survive: their vowel signs are marks, not ``\\w`` characters.
    """
    table: Dict[int, str] = {}
    for cp in range(0x3000):
        cat = unicodedata.category(chr(cp))
        if cat[0] in ("P", "S") or cat == "Cc":
            table[cp] = " "
    for lo, hi in ((0x3000, 0x3040), (0xFE10, 0xFE70), (0xFF00, 0xFF70)):
        for cp in range(lo, hi):
            cat = unicodedata.category(chr(cp))
            if cat[0] in ("P", "S") or cat == "Cc":
                table[cp] = " "
    return table


_PUNCTUATION_TABLE = _build_punctuation_table()


def strip_accents(text: str, keep_non_latin: bool = True) -> str:
    """Remove diacritics and accents (e.g., Énterprises -> Enterprises, Bóral -> Boral).

    NFKD + "drop combining marks" is only safe for Latin text: in Devanagari/Tamil the
    combining marks are vowel signs (े ि ं ்), so folding them silently destroys the name
    ("राम मार्केटिंग" -> "र म म रक ट ग"). Non-Latin strings are therefore returned
    unchanged by default; pass ``keep_non_latin=False`` for the old lossy behaviour.
    """
    if not text:
        return ""
    text = str(text)
    if keep_non_latin and not _is_latin_text(text):
        return text
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def strip_punctuation(text: str) -> str:
    """Replace Unicode punctuation/symbols with spaces, preserving letters and marks."""
    return str(text).translate(_PUNCTUATION_TABLE)


def strip_legal_suffixes(text: str) -> str:
    """Drop standalone legal-suffix words (llc, ltd, inc, co, pvt, ...) from a string."""
    return _WS_RE.sub(" ", _LEGAL_SUFFIX_WORDS_RE.sub(" ", text)).strip()


def strip_dba_markers(text: str) -> str:
    """Remove "doing business as" markers, keeping the trade name that follows.

    Must run **before** punctuation stripping: after ``clean_text`` the marker
    ``D.B.A.`` is already the three tokens ``d b a`` and no pattern can match it.
    """
    return _DBA_RE.sub(" ", str(text))



def clean_text(text: str) -> str:
    """Basic text cleaning: accent stripping, lowercasing, bracket removal, whitespace collapse.

    Punctuation is removed with :func:`strip_punctuation`, which keeps combining marks so
    non-Latin scripts (Devanagari, Tamil, ...) are not shredded into consonants.
    """
    if not text or pd_isna(text):
        return ""
    text = str(text)
    # Strip web domain extensions before punctuation removal (".com" would leave "com")
    text = _DOMAIN_RE.sub(" ", text)
    # Strip accents (Latin-only — see strip_accents)
    text = strip_accents(text)
    # Lowercase
    text = text.lower().strip()
    # Strip brackets [[...]], (...)
    text = _BRACKETS_RE.sub(" ", text)
    # Replace punctuation/symbols with spaces (letters + script marks preserved)
    text = strip_punctuation(text)
    # Collapse multiple whitespaces
    return _WS_RE.sub(" ", text).strip()


def pd_isna(val) -> bool:
    """Safely check for NaN/None/null string values."""
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "<null>")


def _apply_legal_patterns(cleaned: str) -> Tuple[str, str]:
    """Strip legal suffixes from already-cleaned text, returning (text, first_tag)."""
    first_tag = ""
    for pattern, legal_tag in LEGAL_PATTERNS:
        if re.search(pattern, cleaned):
            # Keep the FIRST (most specific) tag: "ABC Pvt Ltd" -> pvt_ltd, not ltd.
            if legal_tag and not first_tag:
                first_tag = legal_tag
            cleaned = re.sub(pattern, " ", cleaned)
    return _WS_RE.sub(" ", cleaned).strip(), first_tag


def normalize_business_name(name: str) -> Tuple[str, str]:
    """Normalize business name and extract legal suffix.

    Returns:
        Tuple[str, str]: (cleaned_name_without_legal, extracted_legal_suffix)

    The DBA marker is removed on the raw string so the trade name after it is kept
    ("Orelee's D.B.A. Barbershop" -> "orelee s barbershop"), and it is *not* reported as a
    legal suffix.
    """
    if pd_isna(name):
        return "", ""

    # Resolve DBA / "doing business as" before punctuation stripping, else "D.B.A."
    # becomes the tokens "d b a" and no pattern can match it.
    cleaned = clean_text(strip_dba_markers(name))
    return _apply_legal_patterns(cleaned)


def normalize_address(address: str) -> str:
    """Normalize address fields by expanding standard street abbreviations.

    Street types are only expanded when they are a *suffix* (``Elm St,`` -> ``elm street,``)
    so that "12 St Mary Road" keeps "st" (saint) instead of becoming "street mary road".
    ``P.O. Box`` and the "45nd" ordinal typo are fixed before punctuation removal.
    """
    if pd_isna(address):
        return ""

    raw = str(address)
    raw = _PO_BOX_RE.sub("pobox", raw)
    raw = _ORDINAL_TYPO_RE.sub("th", raw)   # 45nd -> 45th
    raw = _UNIT_RE.sub(lambda m: _UNIT_MAP.get(m.group(1).lower(), m.group(1)), raw)
    raw = _STREET_SUFFIX_RE.sub(
        lambda m: STREET_TYPE_MAP.get(m.group(1).lower(), m.group(1)), raw)

    return _WS_RE.sub(" ", clean_text(raw)).strip()


