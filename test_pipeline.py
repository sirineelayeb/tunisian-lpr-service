import cv2
import asyncio
from app.detection.detector import detector
from app.detection.ocr import ocr_reader
from app.validation.plate_validator import validate_plate

async def test_image(path):
    print(f"\n📸 Testing image: {path}")

    # Load models
    detector.load()
    ocr_reader.load()

    # Read image
    frame = cv2.imread(path)
    if frame is None:
        print("❌ Failed to load image")
        return

    # Step 1: Detect plates
    boxes = detector.detect(frame)
    print(f"🔍 YOLO detections: {len(boxes)}")

    crops = [detector.crop(frame, box) for box in boxes]

    # Fallback if YOLO fails
    if not crops:
        print("⚠️ YOLO failed → trying contour detection...")
        crops = detector.detect_by_contours(frame)

    if not crops:
        print("❌ No plate detected")
        return

    # Step 2: OCR each crop
    for i, crop in enumerate(crops):
        print(f"\n➡️ Plate #{i+1}")

        cv2.imshow(f"crop_{i}", crop)

        text, conf = ocr_reader.read(crop)
        print(f"   OCR Raw: {text}")
        print(f"   Confidence: {conf:.2f}")

        if not text:
            continue

        # Step 3: Validate
        plate, plate_type = validate_plate(text)

        if plate:
            print(f"   ✅ FINAL: {plate} ({plate_type})")
        else:
            print("   ❌ Invalid Tunisian plate")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(test_image("test.jpg"))