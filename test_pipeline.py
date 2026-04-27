# test_pipeline.py
import cv2
from app.detection.detector import detector
from app.detection.ocr import ocr_reader
from app.validation.plate_validator import validate_plate

detector.load()
ocr_reader.load()

img = cv2.imread("test_car.jpg")   # put any Tunisian car photo here

# Detect
boxes = detector.detect(img)
print(f"Plates found: {len(boxes)}")

if not boxes:
    print("YOLO missed it — trying contour fallback...")
    crops = detector.detect_by_contours(img)
else:
    crops = [detector.crop(img, box) for box in boxes]

for i, crop in enumerate(crops):
    text, conf = ocr_reader.read(crop)
    plate, plate_type = validate_plate(text or "")
    print(f"\nCrop {i+1}:")
    print(f"  OCR    : '{text}' (conf={conf:.2f})")
    print(f"  Plate  : '{plate}' [{plate_type}]")