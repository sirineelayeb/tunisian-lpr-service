import logging
import asyncio
import numpy as np
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from typing import List
import os

from app.config import config
from app.client.backend_client import backend_client
from app.streams.processor import StreamProcessor

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Stream processors ────────────────────────────────────────
entry_processor = StreamProcessor(
    rtsp_url=config.ENTRY_CAMERA_RTSP,
    direction="entry",
    camera_id="cam_entry_01"
)
exit_processor = StreamProcessor(
    rtsp_url=config.EXIT_CAMERA_RTSP,
    direction="exit",
    camera_id="cam_exit_01"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LPR service...")

    from app.detection.detector import detector
    from app.detection.ocr import ocr_reader
    detector.load()
    ocr_reader.load()

    ok = await backend_client.health_check()
    if ok:
        logger.info("Backend connection: OK")
    else:
        logger.warning("Backend connection: FAILED — detections will retry automatically")

    # camera processors
    # asyncio.create_task(entry_processor.start())
    # asyncio.create_task(exit_processor.start())
    logger.info("Camera streams: disabled (no cameras connected yet)")

    logger.info("LPR service ready")
    yield

    logger.info("Shutting down LPR service...")
    entry_processor.stop()
    exit_processor.stop()
    await backend_client.close()
    logger.info("LPR service stopped")


app = FastAPI(
    title="LPR Service",
    description="License Plate Recognition service for Tunisian fleet management",
    version="1.0.0",
    lifespan=lifespan
)


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "lpr",
        "streams": {
            "entry": entry_processor.running,
            "exit":  exit_processor.running,
        }
    }


@app.get("/status")
async def status():
    return {
        "entry_stream": {
            "url":     config.ENTRY_CAMERA_RTSP or "not configured",
            "running": entry_processor.running,
        },
        "exit_stream": {
            "url":     config.EXIT_CAMERA_RTSP or "not configured",
            "running": exit_processor.running,
        },
        "backend_url":          config.NODE_BACKEND_URL,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "frame_interval":       config.FRAME_INTERVAL,
    }


@app.post("/trigger")
async def manual_trigger(plate: str, direction: str, confidence: float = 1.0):
    from app.validation.plate_validator import validate_plate

    validated, plate_type = validate_plate(plate)
    if not validated:
        return {"success": False, "error": f"Invalid Tunisian plate format: {plate}"}

    result = await backend_client.send_detection(
        plate_number=validated,
        direction=direction,
        camera_id="manual_trigger",
        confidence=confidence,
    )
    return {"success": result is not None, "data": result}


@app.post("/test-image")
async def test_image(file: UploadFile = File(...)):
    from app.detection.detector import detector
    from app.detection.ocr import ocr_reader
    from app.validation.plate_validator import validate_plate

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    boxes = detector.detect(frame)
    crops = [detector.crop(frame, box) for box in boxes]

    if not crops:
        crops = detector.detect_by_contours(frame)

    if not crops:
        return {"error": "No plate detected in image"}

    results = []
    for i, crop in enumerate(crops):
        text, confidence = ocr_reader.read(crop)
        if text is None:
            continue
        plate, plate_type = validate_plate(text)
        results.append({
            "crop_index": i,
            "raw_ocr":    text,
            "confidence": round(confidence, 4),
            "plate":      plate,
            "plate_type": plate_type,
            "valid":      bool(plate is not None),
        })

    if not results:
        return {"error": "No plate text could be read"}

    return {"detections": results}


@app.post("/test-images")
async def test_images(files: List[UploadFile] = File(...)):
    from app.detection.detector import detector
    from app.detection.ocr import ocr_reader
    from app.validation.plate_validator import validate_plate

    all_results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            all_results.append({"filename": file.filename, "error": "Could not decode image"})
            continue

        boxes = detector.detect(frame)
        crops = [detector.crop(frame, box) for box in boxes]

        if not crops:
            crops = detector.detect_by_contours(frame)

        if not crops:
            all_results.append({"filename": file.filename, "error": "No plate detected"})
            continue

        detections = []
        for i, crop in enumerate(crops):
            text, confidence = ocr_reader.read(crop)
            if text is None:
                continue
            plate, plate_type = validate_plate(text)
            detections.append({
                "crop_index": i,
                "raw_ocr":    text,
                "confidence": round(confidence, 4),
                "plate":      plate,
                "plate_type": plate_type,
                "valid":      bool(plate is not None),
            })

        all_results.append({"filename": file.filename, "detections": detections})
        save_dir = "debug_crops"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/{file.filename}_crop{i}.jpg", crop)
        logger.info(f"Saved crop: {save_dir}/{file.filename}_crop{i}.jpg  shape={crop.shape}")

    return {"results": all_results}