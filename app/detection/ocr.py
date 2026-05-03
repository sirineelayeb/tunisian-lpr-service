import logging
import re
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Civilian plate parser (COLAB-ALIGNED)
# ─────────────────────────────────────────────
def _is_valid_split(prefix: str, suffix: str) -> bool:
    return 1 <= len(prefix) <= 3 and 1 <= len(suffix) <= 4


def parse_tunisian_civilian(raw: str) -> str | None:
    if not raw:
        return None

    groups = re.findall(r"\d+", raw)

    # Case 1: exactly two groups
    if len(groups) == 2:
        prefix, suffix = groups
        if _is_valid_split(prefix, suffix):
            return f"{prefix} TN {suffix}"
        if _is_valid_split(suffix, prefix):
            return f"{suffix} TN {prefix}"
        return None

    # Case 2: one blob
    if len(groups) == 1:
        blob = groups[0]
        n = len(blob)

        if n < 2 or n > 7:
            return None

        candidates = []
        for split in range(1, n):
            p, s = blob[:split], blob[split:]
            if _is_valid_split(p, s):
                candidates.append((p, s))

        if not candidates:
            return None

        best = max(candidates, key=lambda ps: (min(len(ps[1]), 4), len(ps[0])))
        return f"{best[0]} TN {best[1]}"

    # Case 3: more than 2 groups → merge
    if len(groups) > 2:
        return parse_tunisian_civilian("".join(groups))

    return None


# ─────────────────────────────────────────────
# OCR Reader
# ─────────────────────────────────────────────
class OCRReader:
    def __init__(self):
        self.easy_reader = None
        self.easy_loaded = False

    def load(self):
        try:
            import easyocr
            import torch

            gpu = torch.cuda.is_available()
            self.easy_reader = easyocr.Reader(["en", "ar"], gpu=gpu)
            self.easy_loaded = True

            logger.info(f"EasyOCR loaded (GPU={gpu})")

        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

    def read(self, crop: np.ndarray) -> tuple[str | None, float]:
        if crop is None or crop.size == 0:
            return None, 0.0

        def _ocr(img):
            results = self.easy_reader.readtext(
                img,
                detail=1,
                allowlist=None,
                width_ths=0.7,
                paragraph=False,
            )

            if not results:
                return None, 0.0

            texts = [r[1] for r in results]
            confs = [r[2] for r in results]
            avg_conf = sum(confs) / len(confs)

            if avg_conf < 0.1:
                return None, 0.0

            return " ".join(texts).strip(), avg_conf

        # ── 1. Gray ───────────────────────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        if h < 64:
            scale = 64 / h
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Light denoise
        gray = cv2.medianBlur(gray, 3)

        # ── 2. OCR ────────────────────────────────
        text, conf = _ocr(gray)

        # ── 3. Threshold ──────────────────────────
        if text is None:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            text, conf = _ocr(thresh)

        # ── 4. Inverted ───────────────────────────
        if text is None:
            inverted = cv2.bitwise_not(gray)
            text, conf = _ocr(inverted)

        # ── Post-processing ───────────────────────
        if text:
            text = re.sub(r'[\u0600-\u06FF]+', ' TN ', text)
            text = re.sub(r'[^0-9A-Z\s\-]', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'\s+TN\s+TN\s+', ' TN ', text)

        # ── Parse civilian ────────────────────────
        if text:
            parsed = parse_tunisian_civilian(text)
            if parsed:
                return parsed, conf

        return text, conf


ocr_reader = OCRReader()