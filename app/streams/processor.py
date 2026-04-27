import logging

logger = logging.getLogger(__name__)


class StreamProcessor:
    def __init__(self, rtsp_url: str, direction: str, camera_id: str):
        self.rtsp_url  = rtsp_url
        self.direction = direction
        self.camera_id = camera_id
        self.running   = False

    async def start(self):
        self.running = True
        logger.info(f"Stream processor ready: {self.camera_id} (camera not connected yet)")

    def stop(self):
        self.running = False