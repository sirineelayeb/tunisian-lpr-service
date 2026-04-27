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
        re.IGNORECASE
    ),

    # ── Government / Ministry ─────────────────────────────────
    # XX - XXXXXX
    "government": re.compile(
        r"^\d{2}\s?-\s?\d{5,6}$"
    ),

    # ── Diplomatic ────────────────────────────────────────────
    "diplomatic_cd": re.compile(
        r"^\d{2,3}\s?CD\s?[\u0600-\u06FF\s]*\d{2,3}$",
        re.IGNORECASE
    ),
    "diplomatic_md": re.compile(
        r"^\d{2,3}\s?MD\s?[\u0600-\u06FF\s]*\d{2,3}$",
        re.IGNORECASE
    ),
    "diplomatic_cmd": re.compile(
        r"^\d{2,3}\s?CMD\s?[\u0600-\u06FF\s]*01$",
        re.IGNORECASE
    ),

    # ── Temporary / Foreign residents ─────────────────────────
    "temporary_arabic": re.compile(
        r"^\d{5,6}\s?نت$"
    ),
    "temporary_rs": re.compile(
        r"^RS\s?\d{5,6}$",
        re.IGNORECASE
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


def validate_plate(text: str) -> tuple:
    """
    Returns (cleaned_plate, plate_type) if valid Tunisian plate.
    Returns (None, '') if not recognized.
    """
    if not text:
        return None, ''

    cleaned = _clean(text)

    if len(cleaned.replace(" ", "")) < MIN_LENGTH:
        return None, ''
    if len(cleaned.replace(" ", "")) > MAX_LENGTH:
        return None, ''

    for plate_type, pattern in TUNISIAN_FORMATS.items():
        if pattern.match(cleaned):
            return cleaned.upper(), plate_type

    return None, ''


def normalize_plate(plate: str) -> str:
    """Normalize for deduplication."""
    return re.sub(r"\s+", "", plate).upper()


def _clean(text: str) -> str:
    noise = r"[|\\\/\[\]{}()<>\"'`~!@#$%^&*+=]"
    cleaned = re.sub(noise, "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from app.detection.ocr import parse_tunisian_civilian

    # Test parse_tunisian_civilian
    parse_tests = [
        ("123 4567",   "123 TN 4567"),   # normal 3+4
        ("123 456",    "123 TN 456"),    # 3+3
        ("123 45",     "123 TN 45"),     # 3+2
        ("123 4",      "123 TN 4"),      # 3+1
        ("12 4567",    "12 TN 4567"),    # 2+4
        ("3 4567",     "3 TN 4567"),     # 1+4
        ("1964 63",    "63 TN 1964"),    # flipped → swap
        ("1234567",    "123 TN 4567"),   # merged 7 digits
        ("123456",     "12 TN 3456"),    # merged 6 digits
        ("34567",      "3 TN 4567"),     # merged 5 digits
        ("123TN4567",  "123 TN 4567"),   # TN present but stripped
        ("garbage",    None),
        ("",           None),
    ]

    print("parse_tunisian_civilian tests\n" + "─" * 40)
    all_passed = True
    for raw, expected in parse_tests:
        result = parse_tunisian_civilian(raw)
        passed = result == expected
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{raw}' → '{result}'  (expected: '{expected}')")

    # Test validate_plate
    plate_tests = [
        ("123 TN 4567",   "civilian_latin"),   # 3+4
        ("123 TN 456",    "civilian_latin"),   # 3+3
        ("123 TN 45",     "civilian_latin"),   # 3+2
        ("123 TN 4",      "civilian_latin"),   # 3+1
        ("12 TN 4567",    "civilian_latin"),   # 2+4
        ("3 TN 4567",     "civilian_latin"),   # 1+4
        ("03 - 012345",   "government"),
        ("15 - 123456",   "government"),
        ("29 CD 01",      "diplomatic_cd"),
        ("RS 123456",     "temporary_rs"),
        ("12345 نت",      "temporary_arabic"),
        ("12345 عع",      "dealer"),
        ("12345",         "military"),
        ("INVALID!!",     None),
        ("",              None),
    ]

    print("\nvalidate_plate tests\n" + "─" * 40)
    for plate, expected_type in plate_tests:
        result, plate_type = validate_plate(plate)
        passed = plate_type == expected_type or (result is None and expected_type is None)
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] '{plate}' → {result} (type: {plate_type})")

    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests FAILED!'}")