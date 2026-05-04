import httpx
import logging
from datetime import datetime
from app.config import config

logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(self):
        self.url     = config.NODE_BACKEND_URL
        self.headers = {"Content-Type": "application/json"}
        if config.API_SECRET_KEY:
            self.headers["Authorization"] = f"Bearer {config.API_SECRET_KEY}"
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send_detection(self, plate_number, direction, camera_id, confidence, loading_zone=None):
        if self._client.is_closed:
            logger.warning("HTTP client is closed, skipping send")
            return None

        payload = {
            "plateNumber": plate_number,
            "direction":   direction,
            "cameraId":    camera_id,
            "confidence":  round(confidence, 4),
            "source":      "camera",
            "timestamp":   datetime.utcnow().isoformat() + "Z",
        }
        if loading_zone:
            payload["loadingZone"] = loading_zone

        logger.info(f"Sending to {self.url} → {payload}")

        try:
            response = await self._client.post(self.url, json=payload, headers=self.headers)
            logger.info(f"Response: {response.status_code} — {response.text}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"Detection sent: {plate_number} [{direction}] authorized={data.get('data', {}).get('isAuthorized')}")
            return data
        except httpx.TimeoutException:
            logger.error(f"Timeout sending detection for {plate_number}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Backend returned {e.response.status_code} for {plate_number}")
        except httpx.RequestError as e:
            logger.error(f"Network error: {e}")
        return None

    async def health_check(self):
        if self._client.is_closed:
            return False
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            base_url = f"{parsed.scheme}://{parsed.netloc}/health"
            response = await self._client.get(base_url, timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Backend health check failed: {e}")
            return False

    async def close(self):
        await self._client.aclose()


backend_client = BackendClient()