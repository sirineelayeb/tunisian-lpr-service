import asyncio
import logging
import time
import cv2
import numpy as np
from app.config import config
from app.validation.plate_validator import validate_plate, normalize_plate
from app.client.backend_client import backend_client

logger = logging.getLogger(__name__)


class StreamProcessor:
    def __init__(self, rtsp_url: str, direction: str, camera_id: str):
        self.rtsp_url  = rtsp_url
        self.direction = direction
        self.camera_id = camera_id
        self.running   = False

        self._last_plate:      str | None = None
        self._last_plate_time: float      = 0

    def _is_duplicate(self, plate: str) -> bool:
        if normalize_plate(plate) == normalize_plate(self._last_plate or ""):
            elapsed = time.time() - self._last_plate_time
            if elapsed < config.DUPLICATE_COOLDOWN_SEC:
                return True
        return False

    async def start(self):
        self.running = True
        logger.info(f"Stream processor started: {self.camera_id} ({self.direction})")

        from app.detection.detector import detector
        from app.detection.ocr import ocr_reader

        if not detector.loaded:
            detector.load()
        if not ocr_reader.easy_loaded:
            ocr_reader.load()

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error(f"Cannot open stream: {self.rtsp_url}")
            return

        while self.running:
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Lost stream {self.camera_id}, retrying in 3s...")
                cap.release()                          # release before reconnecting
                await asyncio.sleep(3)
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            await self._process_frame(frame, detector, ocr_reader)
            await asyncio.sleep(config.FRAME_INTERVAL)

        cap.release()
        logger.info(f"Stream processor stopped: {self.camera_id}")

    async def _process_frame(self, frame, detector, ocr_reader):
        """Full pipeline: detect → crop → OCR → validate → send."""
        try:
            # Step 1: YOLO detection
            boxes = detector.detect(frame)
            crops = [detector.crop(frame, box) for box in boxes]

            # Step 2: Fallback to contour detection
            if not crops:
                crops = detector.detect_by_contours(frame)

            if not crops:
                return

            # Step 3: OCR each crop
            for crop in crops:
                text, confidence = ocr_reader.read(crop)
                if not text:
                    continue

                # Step 4: Validate Tunisian format
                plate, plate_type = validate_plate(text)
                if not plate:
                    logger.debug(f"OCR output failed validation: '{text}'")
                    continue

                # Step 5: Skip duplicates
                if self._is_duplicate(plate):
                    logger.debug(f"Duplicate skipped: {plate}")
                    continue

                logger.info(f"Valid plate detected: {plate} ({plate_type}) [{self.direction}]")

                # Step 6: Send to Node.js backend
                result = await backend_client.send_detection(
                    plate_number=plate,
                    direction=self.direction,
                    camera_id=self.camera_id,
                    confidence=confidence,
                )

                if result:
                    self._last_plate      = plate
                    self._last_plate_time = time.time()

        except Exception as e:
            logger.error(f"Frame processing error on {self.camera_id}: {e}", exc_info=True)

    def stop(self):
        self.running = False