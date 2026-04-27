import logging
import re
import numpy as np
import cv2
from app.config import config

logger = logging.getLogger(__name__)


def parse_tunisian_civilian(raw_ocr: str) -> str | None:
    if not raw_ocr:
        return None

    all_digits = re.sub(r"[^0-9]", "", raw_ocr)

    if not all_digits:
        return None

    if len(all_digits) == 7:
        return f"{all_digits[:3]} TN {all_digits[3:]}"

    if len(all_digits) > 7:
        logger.warning(f"Extra digits ({len(all_digits)}), using first 3 + last 4")
        return f"{all_digits[:3]} TN {all_digits[-4:]}"

    if 5 <= len(all_digits) <= 6:
        groups = re.findall(r'\d+', raw_ocr)
        if len(groups) == 2:
            prefix, suffix = groups
            if 1 <= len(prefix) <= 3 and len(suffix) == 4:
                return f"{prefix} TN {suffix}"

        if len(all_digits) == 6:
            return f"{all_digits[:2]} TN {all_digits[2:]}"
        if len(all_digits) == 5:
            return f"{all_digits[:1]} TN {all_digits[1:]}"

    logger.debug(f"Could not parse civilian plate from: '{raw_ocr}'")
    return None


class OCRReader:
    def __init__(self):
        self.easy_reader = None
        self.tess_loaded = False
        self.easy_loaded = False

    def load(self):
        try:
            import easyocr
            self.easy_reader = easyocr.Reader(['en'], gpu=False)
            self.easy_loaded = True
            logger.info("EasyOCR loaded (en)")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {e}")

        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            pytesseract.get_tesseract_version()
            self.tess_loaded = True
            logger.info("Tesseract loaded")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

    def read(self, crop: np.ndarray) -> tuple:
        processed = self._preprocess(crop)

        if self.easy_loaded:
            text, confidence = self._read_easyocr(processed)
            if text:
                parsed = parse_tunisian_civilian(text)
                if parsed:
                    logger.debug(f"EasyOCR (civilian parsed): '{parsed}' ({confidence:.2f})")
                    return parsed, confidence
                if confidence >= 0.4:
                    logger.debug(f"EasyOCR (raw): '{text}' ({confidence:.2f})")
                    return text, confidence

        if self.tess_loaded:
            text = self._read_tesseract(processed)
            if text:
                parsed = parse_tunisian_civilian(text)
                if parsed:
                    logger.debug(f"Tesseract (civilian parsed): '{parsed}'")
                    return parsed, 0.6
                return text, 0.6

        return None, 0.0

    def _read_easyocr(self, image: np.ndarray) -> tuple:
        try:
            best_text = None
            best_conf = 0.0

            results = self.easy_reader.readtext(
                image,
                detail=1,
                allowlist='0123456789 TN',
                width_ths=0.9,
                paragraph=False
            )
            if results:
                texts     = [r[1] for r in results]
                confs     = [r[2] for r in results]
                best_text = " ".join(texts).strip()
                best_conf = sum(confs) / len(confs)

            # Second attempt inverted — handles dark background plates
            if best_conf < 0.6:
                inverted = cv2.bitwise_not(image)
                results2 = self.easy_reader.readtext(
                    inverted,
                    detail=1,
                    allowlist='0123456789 TN',
                    width_ths=0.9,
                    paragraph=False
                )
                if results2:
                    texts2 = [r[1] for r in results2]
                    confs2 = [r[2] for r in results2]
                    text2  = " ".join(texts2).strip()
                    conf2  = sum(confs2) / len(confs2)
                    if conf2 > best_conf:
                        best_text = text2
                        best_conf = conf2

            return best_text or None, best_conf

        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return None, 0.0

    def _read_tesseract(self, image: np.ndarray) -> str | None:
        try:
            import pytesseract
            from PIL import Image
            pil_image = Image.fromarray(image)
            text = pytesseract.image_to_string(
                pil_image,
                config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            return text.strip() or None
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale + sharpen only.
        Tested: adaptive threshold destroys plate text — do NOT add it back.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Always upscale — most critical step (tested: gives conf=0.99)
        h, w = gray.shape
        scale = max(120 / h, 2.0)
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

        # Sharpen
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        gray = cv2.filter2D(gray, -1, kernel)

        return gray


ocr_reader = OCRReader()