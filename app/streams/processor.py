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
        self._loop = None

    def _is_duplicate(self, plate: str) -> bool:
        if normalize_plate(plate) == normalize_plate(self._last_plate or ""):
            elapsed = time.time() - self._last_plate_time
            if elapsed < config.DUPLICATE_COOLDOWN_SEC:
                return True
        return False

    def _read_frame(self, cap: cv2.VideoCapture):
        """Blocking frame read — runs in thread pool, not event loop."""
        return cap.read()

    async def start(self):
        self.running = True
        self._loop = asyncio.get_event_loop()
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

        retry_count = 0
        max_retries = 10  # give up after 10 consecutive failures

        try:
            while self.running:
                # Run blocking cap.read() in a thread so the event loop stays free
                ret, frame = await self._loop.run_in_executor(
                    None, self._read_frame, cap
                )

                if not ret:
                    retry_count += 1
                    logger.warning(
                        f"Lost stream {self.camera_id} "
                        f"(attempt {retry_count}/{max_retries}), retrying in 3s..."
                    )
                    cap.release()

                    if retry_count >= max_retries:
                        logger.error(f"Stream {self.camera_id} failed after {max_retries} retries. Stopping.")
                        break

                    await asyncio.sleep(3)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

                retry_count = 0  # reset on successful frame
                await self._process_frame(frame, detector, ocr_reader)
                await asyncio.sleep(config.FRAME_INTERVAL)

        finally:
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

            # Step 3: OCR each crop — stop at first valid plate
            for crop in crops:
                if crop.size == 0:  # guard against empty crops
                    continue

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

                break  # one confirmed plate per frame is enough

        except Exception as e:
            logger.error(f"Frame processing error on {self.camera_id}: {e}", exc_info=True)

    def stop(self):
        self.running = False