import cv2
import numpy as np
import paddleocr

# Create a simple test image with text
test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(test_img, "TEST TEXT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Initialize PaddleOCR with GPU
print("Initializing PaddleOCR with GPU...")
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)

# Try OCR detection
print("Running OCR detection...")
try:
    result = ocr.ocr(test_img, cls=True)
    print("OCR successful! Results:", len(result[0]) if result and result[0] else 0, "detections")
    if result and result[0]:
        for line in result[0]:
            print("Text:", line[1][0], "Confidence:", line[1][1])
    else:
        print("No text detected")
except Exception as e:
    print("OCR Error:", str(e))
    import traceback

    traceback.print_exc()
