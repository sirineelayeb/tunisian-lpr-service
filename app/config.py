# app/config.py
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
    FRAME_INTERVAL:            float = float(os.getenv("FRAME_INTERVAL",            "0.5"))
    CONFIDENCE_THRESHOLD:      float = float(os.getenv("CONFIDENCE_THRESHOLD",      "0.45"))
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4"))
    DUPLICATE_COOLDOWN_SEC:    int   = int(os.getenv("DUPLICATE_COOLDOWN_SEC",      "10"))

    # ── Paths ───────────────────────────────────────────────
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "models/best.pt")

    # ── Logging ─────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self):
        if not self.NODE_BACKEND_URL:
            raise EnvironmentError("Missing required env var: NODE_BACKEND_URL")


config = Config()
config.validate()  # fail fast on bad config