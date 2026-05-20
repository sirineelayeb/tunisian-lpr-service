# Tunisian LPR Service

A License Plate Recognition (LPR) system built for Tunisian vehicle plates, designed to detect and read plate numbers from images or live RTSP camera streams using YOLO and EasyOCR.

Built as part of a fleet management platform — can be integrated into parking systems, security checkpoints, and access control gates.

---

## Features

- Real-time plate detection via YOLO (custom-trained model)
- OCR with EasyOCR (Arabic + Latin support)
- Tunisian plate format validation (civilian, government, diplomatic, military, temporary, dealer)
- RTSP stream processing with async frame pipeline
- Contour-based fallback detection when YOLO misses
- Duplicate suppression with configurable cooldown
- REST API (FastAPI) with `/test-image`, `/trigger`, `/health`, `/status` endpoints
- Structured logging throughout

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI + Uvicorn |
| Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| Vision | OpenCV (headless) |
| HTTP client | HTTPX (async) |
| Config | python-dotenv |

---

## Project Structure

```
tunisian-lpr-service/
├── app/
│   ├── client/
│   │   └── backend_client.py    # HTTP client for Node.js backend
│   ├── detection/
│   │   ├── detector.py          # YOLO + contour plate detection
│   │   └── ocr.py               # EasyOCR + Tunisian plate parser
│   ├── streams/
│   │   └── processor.py         # Async RTSP stream processor
│   ├── validation/
│   │   └── plate_validator.py   # Tunisian plate format validation
│   ├── config.py                # Environment config
│   └── main.py                  # FastAPI app + lifespan
├── models/
│   └── best.pt                  # Custom YOLO model (not tracked in git)
├── requirements.txt
├── .env                         # Local config (not tracked in git)
└── .gitignore
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sirineelayeb/tunisian-lpr-service.git
cd tunisian-lpr-service
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is not in `requirements.txt`. Install it separately for your platform:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Add your YOLO model

Place your trained model at `models/best.pt` (not included in the repo).

### 5. Configure environment

Create a `.env` file in the project root:

```env
NODE_BACKEND_URL=http://localhost:5000/api/lpr/detect
API_SECRET_KEY=your_secret_key_here

ENTRY_CAMERA_RTSP=rtsp://192.168.1.100:554/stream1
EXIT_CAMERA_RTSP=rtsp://192.168.1.101:554/stream1

YOLO_MODEL_PATH=models/best.pt
YOLO_CONFIDENCE_THRESHOLD=0.4
CONFIDENCE_THRESHOLD=0.45
DUPLICATE_COOLDOWN_SEC=10
FRAME_INTERVAL=0.5

LOG_LEVEL=INFO
```

### 6. Run the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/status` | Stream and config status |
| `POST` | `/test-image` | Test a single image upload |
| `POST` | `/test-images` | Test multiple image uploads |
| `POST` | `/trigger` | Manually trigger a plate detection |

### Example: Test an image

```bash
curl -X POST http://localhost:8000/test-image \
  -F "file=@plate.jpg"
```

Response:
```json
{
  "detections": [
    {
      "crop_index": 0,
      "raw_ocr": "123 TN 4567",
      "confidence": 0.91,
      "plate": "123 TN 4567",
      "plate_type": "civilian_latin",
      "valid": true
    }
  ]
}
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `NODE_BACKEND_URL` | Node.js backend endpoint | `http://localhost:5000/api/lpr/detect` |
| `API_SECRET_KEY` | Optional Bearer token for backend auth | — |
| `ENTRY_CAMERA_RTSP` | Entry camera RTSP URL | — |
| `EXIT_CAMERA_RTSP` | Exit camera RTSP URL | — |
| `YOLO_MODEL_PATH` | Path to trained YOLO `.pt` model | `models/best.pt` |
| `YOLO_CONFIDENCE_THRESHOLD` | Minimum YOLO detection confidence | `0.4` |
| `CONFIDENCE_THRESHOLD` | Minimum OCR confidence | `0.45` |
| `DUPLICATE_COOLDOWN_SEC` | Seconds before re-reporting same plate | `10` |
| `FRAME_INTERVAL` | Seconds between processed frames | `0.5` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Supported Plate Formats

| Type | Example |
|---|---|
| Civilian (Latin OCR) | `123 TN 4567` |
| Civilian (Arabic) | `١٢٣ تونس ٤٥٦٧` |
| Government | `12-123456` |
| Diplomatic (CD/MD/CMD) | `12 CD 34` |
| Temporary | `RS 12345` |
| Military | `12345` |
| Dealer | `1234 عع` |

---

## Roadmap

- [ ] Improve YOLO model accuracy on angled / partial plates
- [ ] Web dashboard for live detection feed
- [ ] Support for additional camera protocols

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Open a Pull Request

---

## License

MIT License

---

## Author

**Syrine Elayeb** — PFE internship project
