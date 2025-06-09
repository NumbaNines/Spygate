import sys

from ultralytics import YOLO


def predict_images(image_path):
    # Load the trained model
    model = YOLO("runs/detect/train5/weights/best.pt")

    # Run prediction
    results = model.predict(
        source=image_path,  # Path to image file or directory
        save=True,  # Save results
        save_txt=True,  # Save results in YOLO format
        conf=0.25,  # Confidence threshold
        save_conf=True,  # Save confidence scores
        project="predictions",  # Save results to predictions/
        name="detect",  # Name of the results folder
    )

    print("\nPredictions saved to 'predictions/detect' folder")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image_or_folder")
    else:
        predict_images(sys.argv[1])
