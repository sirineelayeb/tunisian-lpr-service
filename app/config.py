import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── Camera streams ──────────────────────────────────────
    ENTRY_CAMERA_RTSP: str = os.getenv("ENTRY_CAMERA_RTSP", "")
    EXIT_CAMERA_RTSP:  str = os.getenv("EXIT_CAMERA_RTSP",  "")

    # ── Node.js backend ─────────────────────────────────────
    NODE_BACKEND_URL: str = os.getenv("NODE_BACKEND_URL", "http://localhost:5000/api/lpr/detect")
    API_SECRET_KEY:   str = os.getenv("API_SECRET_KEY",   "")

    # ── Detection settings ──────────────────────────────────
    FRAME_INTERVAL:         float = float(os.getenv("FRAME_INTERVAL",         "0.5"))
    CONFIDENCE_THRESHOLD:   float = float(os.getenv("CONFIDENCE_THRESHOLD",   "0.7"))
    # OCR confidence
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4")) 
    DUPLICATE_COOLDOWN_SEC: int   = int(os.getenv("DUPLICATE_COOLDOWN_SEC",   "10"))

    # ── Paths ───────────────────────────────────────────────
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/best.pt")

    # ── Tesseract — override in .env on Windows ─────────────
    # Windows example: C:\Program Files\Tesseract-OCR\tesseract.exe
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

    # ── Logging ─────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self):
        missing = []
        if not self.NODE_BACKEND_URL:
            missing.append("NODE_BACKEND_URL")
        # Camera URLs are optional until hardware is connected
        # if not self.ENTRY_CAMERA_RTSP: missing.append("ENTRY_CAMERA_RTSP")
        # if not self.EXIT_CAMERA_RTSP:  missing.append("EXIT_CAMERA_RTSP")
        if missing:
            raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
config = Config()