### 🚗 Tunisian LPR Service

## 📌 Overview ##
The Tunisian LPR Service is a License Plate Recognition system designed 
to detect and extract Tunisian vehicle plate numbers from images or video streams using AI and computer vision.

It can be integrated into parking systems, security checkpoints, and fleet management platforms.

## ⚙️ Features ##
- 🚗 Tunisian plate detection
- 🔤 Optical Character Recognition (OCR)
- 🎥 Image & video processing
- 📡 REST API integration
- 🧠 AI-ready architecture (YOLO / OpenCV)
- 🗂️ Logging & tracking system

## 🧱 Tech Stack ##
- Language: Python
- Computer Vision: OpenCV
- AI Detection: YOLO / custom detector
- OCR: EasyOCR / Tesseract

## 📁 Project Structure ##

   ```bash
tunisian-lpr-service/
│── app/
│   ├── client/
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py        # Plate detection logic
│   │   └── ocr.py             # OCR processing
│   │
│   ├── streams/
│   │   ├── __init__.py
│   │   └── processor.py       # Stream/video processing
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── plate_validator.py # Tunisian plate validation
│   │
│   ├── __init__.py
│   └── config.py              # App configuration
│
│── main.py                    # Entry point
│── test_pipeline.py           # Testing pipeline
│── requirements.txt
│── .gitignore
```

## 🚀 Getting Started ##
1. Clone the repository
 ```bash
git clone https://github.com/sirineelayeb/tunisian-lpr-service.git
cd tunisian-lpr-service
```
2. Create a virtual environment
 ```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
## 🔐 Environment Configuration (.env) ##
```bash
NODE_BACKEND_URL=
API_SECRET_KEY=

ENTRY_CAMERA_RTSP=
EXIT_CAMERA_RTSP=

TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

FRAME_INTERVAL=0.5
CONFIDENCE_THRESHOLD=0.7
DUPLICATE_COOLDOWN_SEC=10
LOG_LEVEL=INFO
YOLO_MODEL_PATH=models/best.pt
```
🔑 Environment Variables Explained 
- NODE_BACKEND_URL → Node.js backend endpoint
- API_SECRET_KEY → Optional security key
- ENTRY_CAMERA_RTSP → Entry camera stream URL
- EXIT_CAMERA_RTSP → Exit camera stream URL
- TESSERACT_CMD → Path to Tesseract OCR executable
- FRAME_INTERVAL → Time between frame processing (seconds)
- CONFIDENCE_THRESHOLD → Minimum detection confidence
- DUPLICATE_COOLDOWN_SEC → Time to ignore duplicate plates
- LOG_LEVEL → Logging level (INFO, DEBUG, etc.)
- YOLO_MODEL_PATH → Path to trained YOLO model

4. Run the application
```bash
python main.py
```
API will be available at: 
```bash 
http://localhost:8000
```

## 🛠️ Future Improvements ##
🤖 Improve YOLO model accuracy

## 🤝 Contributing ##
- 1- Fork the repo
- 2- Create a branch (feature/...)
- 3- Commit changes
- 4- Open Pull Request

## 📄 License ##
MIT License

## 👩‍💻 Author ##
Syrine Elayeb
