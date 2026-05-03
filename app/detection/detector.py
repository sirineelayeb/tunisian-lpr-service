import logging
import numpy as np
import cv2
from app.config import config

logger = logging.getLogger(__name__)


class PlateDetector:
    def __init__(self):
        self.model  = None
        self.loaded = False

    def load(self):
        try:
            import torch
            from ultralytics import YOLO
            torch.serialization.add_safe_globals([])
            self.model  = YOLO(config.YOLO_MODEL_PATH)
            self.loaded = True
            logger.info(f"YOLO model loaded: {config.YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")

    def detect(self, frame: np.ndarray) -> list:
        """Run YOLO on full frame, return all boxes above YOLO confidence threshold.
        
        FIX: Use a dedicated YOLO threshold (0.4), not the OCR threshold (0.7).
        YOLO detection confidence and OCR read confidence are different things.
        """
        if not self.loaded or self.model is None:
            return []

        results = self.model(frame, verbose=False)
        boxes = [
            box for box in results[0].boxes
            if float(box.conf) >= config.YOLO_CONFIDENCE_THRESHOLD
        ]
        logger.debug(f"YOLO detected {len(boxes)} plate(s)")
        return boxes

    def crop(self, frame: np.ndarray, box) -> np.ndarray:
        """Crop detected region from frame with small padding."""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 12
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2]

    def detect_by_contours(self, frame: np.ndarray) -> list:
        """
        Fallback: find plate-like rectangles using contour detection.
        Used when YOLO misses the plate.
        Returns list of cropped plate images.

        FIX: Widened aspect ratio from (2.0–5.5) to (1.5–7.0) to handle
        partial views, angles, and non-standard crops.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur    = cv2.bilateralFilter(gray, 11, 17, 17)
        edges   = cv2.Canny(blur, 30, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plates = []
        for c in contours:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ratio = w / h
                # FIX: was 2.0–5.5, too strict for angled/partial plates
                if 1.5 <= ratio <= 7.0 and w > 60:
                    plates.append(frame[y:y+h, x:x+w])
                    logger.debug(f"Contour plate found: {w}x{h} ratio={ratio:.2f}")

        return plates


detector = PlateDetector()