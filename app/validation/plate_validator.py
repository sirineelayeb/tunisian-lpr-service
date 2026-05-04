import re

# ─────────────────────────────────────────────
# Tunisian plate formats
# ─────────────────────────────────────────────
TUNISIAN_FORMATS = {

    # Civilian (Arabic)
    "civilian": re.compile(
        r"^\d{1,3}\s?تونس\s?\d{1,4}$"
    ),

    # Civilian (Latin OCR)
    "civilian_latin": re.compile(
        r"^\d{1,3}\s?TN\s?\d{1,4}$",
        re.IGNORECASE,
    ),

    # Government
    "government": re.compile(
        r"^\d{2}\s*-\s*\d{5,6}$"
    ),

    # Diplomatic
    "diplomatic_cd": re.compile(
        r"^\d{2,3}\s?CD\s?[\u0600-\u06FF\s]*\d{2,3}$",
        re.IGNORECASE,
    ),
    "diplomatic_md": re.compile(
        r"^\d{2,3}\s?MD\s?[\u0600-\u06FF\s]*\d{2,3}$",
        re.IGNORECASE,
    ),
    "diplomatic_cmd": re.compile(
        r"^\d{2,3}\s?CMD\s?[\u0600-\u06FF\s]*01$",
        re.IGNORECASE,
    ),

    # Temporary
    "temporary_arabic": re.compile(
        r"^\d{5,6}\s?نت$"
    ),
    "temporary_rs": re.compile(
        r"^RS\s?\d{5,6}$",
        re.IGNORECASE,
    ),

    # Dealer
    "dealer": re.compile(
        r"^\d{4,5}\s?عع$"
    ),

    # Military (fallback)
    "military": re.compile(
        r"^\d{5}$"
    ),
}

MIN_LENGTH = 3
MAX_LENGTH = 15


# ─────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────
def validate_plate(text: str) -> tuple[str | None, str]:
    if not text:
        return None, ""

    cleaned = _clean(text)
    stripped = cleaned.replace(" ", "")

    if len(stripped) < MIN_LENGTH or len(stripped) > MAX_LENGTH:
        return None, ""

    for plate_type, pattern in TUNISIAN_FORMATS.items():
        if pattern.match(cleaned):
            return cleaned.upper(), plate_type

    return None, ""


# ─────────────────────────────────────────────
# Normalize (for deduplication)
# ─────────────────────────────────────────────
def normalize_plate(plate: str | None) -> str:
    if not plate:
        return ""
    return re.sub(r"\s+", "", plate).upper()


# ─────────────────────────────────────────────
# Cleaner
# ─────────────────────────────────────────────
def _clean(text: str) -> str:
    noise = r"[|\\\/\[\]{}()<>\"'`~!@#$%^&*+=]"
    cleaned = re.sub(noise, "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned