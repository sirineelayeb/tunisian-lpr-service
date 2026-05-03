import re

# ══════════════════════════════════════════════════════════════
# All official Tunisian license plate formats
# ══════════════════════════════════════════════════════════════

TUNISIAN_FORMATS = {

    # ── Standard civilian (Arabic — manual input only) ────────
    # NOTE: Only reachable via manual input (e.g. /trigger endpoint).
    # Camera OCR never outputs Arabic — civilian_latin handles that path.
    "civilian": re.compile(
        r"^\d{1,3}\s?تونس\s?\d{1,4}$"
    ),

    # ── Latinized civilian (OCR output after parse_tunisian_civilian) ──
    # 123 TN 4567 / 123 TN 456 / 123 TN 45 / 123 TN 4
    "civilian_latin": re.compile(
        r"^\d{1,3}\s?TN\s?\d{1,4}$",
        re.IGNORECASE,
    ),

    # ── Government / Ministry ─────────────────────────────────
    # "03 - 012345" → two digits, optional spaces around dash, 5-6 digits
    # FIX: was \s? (0-or-1 space) — now \s* to handle "XX - XXXXXX" spacing
    "government": re.compile(
        r"^\d{2}\s*-\s*\d{5,6}$"
    ),

    # ── Diplomatic ────────────────────────────────────────────
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

    # ── Temporary / Foreign residents ─────────────────────────
    "temporary_arabic": re.compile(
        r"^\d{5,6}\s?نت$"
    ),
    "temporary_rs": re.compile(
        r"^RS\s?\d{5,6}$",
        re.IGNORECASE,
    ),

    # ── Dealer testing ────────────────────────────────────────
    "dealer": re.compile(
        r"^\d{4,5}\s?عع$"
    ),

    # ── Military — most generic, always last ──────────────────
    "military": re.compile(
        r"^\d{5}$"
    ),
}

MIN_LENGTH = 3
MAX_LENGTH = 15


def validate_plate(text: str) -> tuple[str | None, str]:
    """
    Returns (cleaned_plate, plate_type) if valid Tunisian plate.
    Returns (None, '') if not recognised.
    """
    if not text:
        return None, ""

    cleaned = _clean(text)

    stripped = cleaned.replace(" ", "")
    if len(stripped) < MIN_LENGTH:
        return None, ""
    if len(stripped) > MAX_LENGTH:
        return None, ""

    for plate_type, pattern in TUNISIAN_FORMATS.items():
        if pattern.match(cleaned):
            return cleaned.upper(), plate_type

    return None, ""


def normalize_plate(plate: str | None) -> str:
    """
    Normalize for deduplication. Safe against None input.
    """
    if not plate:
        return ""
    return re.sub(r"\s+", "", plate).upper()


def _clean(text: str) -> str:
    noise = r"[|\\\/\[\]{}()<>\"'`~!@#$%^&*+=]"
    cleaned = re.sub(noise, "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned