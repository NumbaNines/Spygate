import cv2

# Change to a video file path if you want to test with a file
VIDEO_SOURCE = 0  # 0 for webcam, or 'path/to/video.mp4'

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Test Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 