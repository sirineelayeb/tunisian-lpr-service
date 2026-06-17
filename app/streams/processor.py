import asyncio
import logging
import time
import cv2
import numpy as np
from difflib import SequenceMatcher
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
        self._highlighted_plate: dict | None = None
        self._loop = None

        # Baseline frame used for scene-change detection (grayscale, downscaled).
        # Only updated when a frame is actually processed, so slow/gradual
        # changes still accumulate against a fixed reference instead of
        # resetting every frame.
        self._last_gray_frame: np.ndarray | None = None

    def _is_duplicate(self, plate: str) -> bool:
        """Check if plate is duplicate using fuzzy matching (70% similarity threshold for OCR tolerance)."""
        if not self._last_plate:
            return False
        
        norm_current = normalize_plate(plate)
        norm_last = normalize_plate(self._last_plate)
        
        # Fuzzy match: if 70%+ similar, it's the same plate (tolerates OCR variations)
        similarity = SequenceMatcher(None, norm_current, norm_last).ratio()
        return similarity >= 0.70

    def _read_frame(self, cap: cv2.VideoCapture):
        """Blocking frame read — runs in thread pool, not event loop."""
        return cap.read()

    def _rotate_frame(self, frame: np.ndarray, degrees: int) -> np.ndarray:
        if degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _draw_plate_overlay(self, frame: np.ndarray):
        if not self._highlighted_plate:
            return

        text = self._highlighted_plate["text"]
        box = self._highlighted_plate.get("box")
        overlay_time = self._highlighted_plate.get("timestamp", 0)
        # if time.time() - overlay_time > 5.0:
        #     self._highlighted_plate = None
        #     return

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_position = (x1, max(30, y1 - 10))
        else:
            text_position = (10, 30)

        text_label = f"Plate: {text}"
        text_size, _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
        cv2.rectangle(
            frame,
            (text_position[0] - 5, text_position[1] - text_h - 10),
            (text_position[0] + text_w + 5, text_position[1] + 5),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text_label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    def _frame_changed(self, frame: np.ndarray) -> bool:
        """
        Lightweight scene-change detector.

        Downscales + blurs the frame to ignore sensor noise, diffs it against
        the last *processed* frame, and reports whether enough of the frame
        changed to be worth running detection + OCR on. The baseline only
        resets when we actually decide to process, so small frame-to-frame
        drift accumulates correctly instead of being "absorbed" every tick.
        """
        small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(small, (160, 90), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        if self._last_gray_frame is None:
            self._last_gray_frame = small
            return True  # first frame — always process

        diff = cv2.absdiff(small, self._last_gray_frame)
        _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_ratio = cv2.countNonZero(diff_thresh) / diff_thresh.size

        if changed_ratio >= config.FRAME_CHANGE_THRESHOLD:
            self._last_gray_frame = small
            return True

        return False

    async def start(self):
        if not self.rtsp_url:
            logger.warning(f"No RTSP URL configured for {self.camera_id}; skipping stream")
            return

        self.running = True
        self._loop = asyncio.get_running_loop()   # was get_event_loop() — deprecated
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
        max_retries = 10
        window_name = f"LPR - {self.camera_id}"

        try:
            while self.running:
                ret, frame = await self._loop.run_in_executor(None, self._read_frame, cap)

                if not ret:
                    retry_count += 1
                    logger.warning(f"Lost stream {self.camera_id} (attempt {retry_count}/{max_retries}), retrying in 3s...")
                    cap.release()
                    if retry_count >= max_retries:
                        logger.error(f"Stream {self.camera_id} failed after {max_retries} retries. Stopping.")
                        break
                    await asyncio.sleep(3)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

                if config.FRAME_ROTATION:
                    frame = self._rotate_frame(frame, config.FRAME_ROTATION)

                retry_count = 0

                if self._frame_changed(frame):
                    await self._process_frame(frame, detector, ocr_reader)
                else:
                    logger.debug(f"Scene unchanged [{self.camera_id}] — skipping detection")

                self._draw_plate_overlay(frame)

                # Step 1: just prove the feed is alive — no boxes yet
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info(f"'q' pressed — stopping {self.camera_id}")
                    self.running = False
                    break

                await asyncio.sleep(config.FRAME_INTERVAL)
        finally:
            cap.release()
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
            logger.info(f"Stream processor stopped: {self.camera_id}")

    async def _process_frame(self, frame, detector, ocr_reader):
        """Full pipeline: detect → crop → OCR → validate → send."""
        try:
            # Step 1: YOLO detection
            boxes = detector.detect(frame)
            plate_box = None
            crops = [detector.crop(frame, box) for box in boxes]
            if boxes:
                logger.info(f"YOLO found {len(boxes)} box(es) in frame [{self.direction}]")
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                plate_box = (x1, y1, x2, y2)

            # Step 2: Fallback to contour detection
            if not crops:
                contour_results = detector.detect_by_contours(frame)
                if contour_results:
                    crops = [crop for crop, _ in contour_results]
                    plate_box = contour_results[0][1]
                    logger.info(f"Contour detection found {len(crops)} plate(s) [{self.direction}]")

            if not crops:
                logger.debug(f"No plates detected in frame [{self.direction}]")
                return

            # Step 3: OCR each crop — stop at first valid plate
            for crop in crops:
                if crop.size == 0:  # guard against empty crops
                    continue

                text, confidence = ocr_reader.read(crop)
                if not text:
                    continue

                logger.info(f"OCR candidate: '{text}' confidence={confidence:.4f} [{self.direction}]")

                if confidence < config.CONFIDENCE_THRESHOLD:
                    logger.warning(
                        f"OCR confidence below threshold: {confidence:.4f} < {config.CONFIDENCE_THRESHOLD} for '{text}'"
                    )
                    continue

                # Step 4: Validate Tunisian format
                plate, plate_type = validate_plate(text)
                if not plate:
                    logger.warning(f"OCR output invalid for Tunisian plate: '{text}'")
                    continue

                # Step 5: Skip duplicates (fuzzy match with 70% similarity)
                if self._is_duplicate(plate):
                    norm_current = normalize_plate(plate)
                    norm_last = normalize_plate(self._last_plate)
                    similarity = SequenceMatcher(None, norm_current, norm_last).ratio()
                    logger.info(f"Duplicate blocked: '{plate}' (similarity={similarity:.2%} to '{self._last_plate}') [{self.direction}]")
                    continue

                logger.info(f"Valid plate detected: {plate} ({plate_type}) [{self.direction}]")

                self._highlighted_plate = {
                    "text": plate,
                    "box": plate_box,
                    "timestamp": time.time(),
                }

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