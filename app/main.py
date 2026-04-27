import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

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

    # Load models
    from app.detection.detector import detector
    from app.detection.ocr import ocr_reader
    detector.load()
    ocr_reader.load()

    # Check backend connectivity
    ok = await backend_client.health_check()
    if ok:
        logger.info("Backend connection: OK")
    else:
        logger.warning("Backend connection: FAILED — detections will retry automatically")

    # Start camera streams (uncomment when cameras are available)
    # asyncio.create_task(entry_processor.start())
    # asyncio.create_task(exit_processor.start())
    logger.info("Camera streams: disabled (no cameras connected yet)")

    logger.info("LPR service ready")
    yield

    # Shutdown
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
        "backend_url":            config.NODE_BACKEND_URL,
        "confidence_threshold":   config.CONFIDENCE_THRESHOLD,
        "frame_interval":         config.FRAME_INTERVAL,
    }