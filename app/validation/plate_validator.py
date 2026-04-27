import re

PATTERNS = [
    re.compile(r"^\d{1,3}\s?TN\s?\d{3,4}$", re.IGNORECASE),
    re.compile(r"^TN-\d{3}-[A-Z]{3}$",       re.IGNORECASE),
]

def validate_tunisian_plate(text):
    if not text:
        return None
    cleaned = re.sub(r"[|\\\/\[\]{}()<>\"'`~!@#$%^&*+=]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern in PATTERNS:
        if pattern.match(cleaned):
            return cleaned.upper()
    return None

def normalize_plate(plate):
    return re.sub(r"\s+", "", plate).upper()