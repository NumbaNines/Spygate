#!/usr/bin/env python3
"""Live detection with all SpygateAI systems: YOLO HUD detection, OCR, and game state tracking."""

import argparse
import signal
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# Import SpygateAI components
from spygate.ml.enhanced_ocr import EnhancedOCR
from spygate.ml.game_state import GameState
from spygate.ml.visualization_engine import DetectionVisualizer, VisualizationConfig

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class ScreenCapture:
    """Screen capture with multiple backends."""

    def __init__(self, monitor=1):
        self.sct = mss()
        self.monitor = self.sct.monitors[monitor]  # Get the specified monitor

    def capture_frame(self):
        screenshot = self.sct.grab(self.monitor)
        # Convert RGBA to RGB
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


class LiveGameAnalyzer:
    """Real-time game analyzer combining all SpygateAI systems."""

    def __init__(self, model_path: str, conf_threshold: float = 0.15):
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Initialize OCR system with performance optimizations
        self.ocr_system = EnhancedOCR()

        # Initialize visualizer with custom config
        vis_config = VisualizationConfig(
            show_confidence=True,
            show_bounding_boxes=True,
            show_labels=True,
            show_triangles=True,
            show_ocr=True,
            box_thickness=2,
            font_scale=0.7,
            font_thickness=2,
        )
        self.visualizer = DetectionVisualizer(config=vis_config)

        # Initialize tracking
        self.fps_queue = deque(maxlen=50)
        self.frame_count = 0

        # Track scores and yard line separately
        self.away_score = None
        self.home_score = None
        self.yard_line = None

    def process_ocr_results(self, frame: np.ndarray, detections: list[dict]) -> dict:
        """Process OCR results based on detection regions."""
        ocr_results = {}

        for detection in detections:
            label = detection["label"]
            box = detection["box"]

            # Extract region for OCR
            x1, y1, x2, y2 = map(int, box)
            region = frame[y1:y2, x1:x2]

            if label == "territory_triangle_area":
                # Only look for yard line in territory area
                text = self.ocr_system.read_text(region)
                if text:
                    # Clean up the text and extract numbers
                    numbers = "".join(filter(str.isdigit, text))
                    # Take only the last two digits if more than 2 digits
                    if len(numbers) >= 2:
                        numbers = numbers[-2:]
                    # Convert to integer and validate range
                    try:
                        yard_line = int(numbers)
                        if 0 <= yard_line <= 50:  # Valid yard line range
                            self.yard_line = yard_line
                            ocr_results["Yard Line"] = str(yard_line)
                    except ValueError:
                        pass

            elif label == "possession_triangle_area":
                # Look for scores in possession area
                text = self.ocr_system.read_text(region)
                if text and text.isdigit() and 0 <= int(text) <= 99:
                    # Assume first number is away score, second is home score
                    numbers = [int(n) for n in text.split() if n.isdigit()]
                    if len(numbers) >= 2:
                        self.away_score = numbers[0]
                        self.home_score = numbers[1]
                        ocr_results["Score"] = f"{numbers[0]}-{numbers[1]}"

        return ocr_results

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Process a single frame."""
        start_time = time.time()

        # Run YOLO detection
        results = self.model(frame)[0]
        detections = []

        # Process detections
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            label = results.names[int(cls)]
            detections.append(
                {"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": conf, "label": label}
            )

        # Run OCR on specific regions
        ocr_results = {}
        for det in detections:
            if det["label"] == "territory_triangle_area":  # Only process territory triangle
                x1, y1, x2, y2 = det["bbox"]
                region = frame[y1:y2, x1:x2]
                text = self.ocr_system.read_text(region)
                ocr_results[det["label"]] = {"text": text, "bbox": det["bbox"]}
                # Print territory OCR result with clear formatting
                print(f"\n🎯 Territory Yard Line: {text}")
                print("-" * 40)

        # Create visualization
        vis_frame = frame.copy()

        # Draw detections and OCR results
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            color = (0, 255, 0) if label == "territory_triangle_area" else (255, 0, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis_frame,
                f"{label}: {det['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Add FPS counter
        processing_time = time.time() - start_time
        self.fps_queue.append(processing_time)
        avg_fps = len(self.fps_queue) / sum(self.fps_queue)

        cv2.putText(
            vis_frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Add frame counter
        self.frame_count += 1
        cv2.putText(
            vis_frame,
            f"Frame: {self.frame_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return vis_frame, ocr_results

    def get_fps(self):
        return len(self.fps_queue) / sum(self.fps_queue)


class VisualizationConfig:
    def __init__(
        self,
        show_confidence=True,
        show_bounding_boxes=True,
        show_labels=True,
        show_triangles=True,
        show_ocr=True,
        box_thickness=2,
        font_scale=0.6,
        font_thickness=2,
    ):
        self.show_confidence = show_confidence
        self.show_bounding_boxes = show_bounding_boxes
        self.show_labels = show_labels
        self.show_triangles = show_triangles
        self.show_ocr = show_ocr
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness


class EnhancedOCR:
    def __init__(self):
        import easyocr

        self.reader = easyocr.Reader(["en"])

    def read_text(self, image: np.ndarray) -> str:
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhance contrast
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

            # Run OCR
            results = self.reader.readtext(image)

            # Get all detected text
            texts = [text for _, text, conf in results if conf > 0.3]

            # Return combined text
            return " ".join(texts) if texts else ""
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""


class DetectionVisualizer:
    def __init__(self, config: VisualizationConfig):
        self.config = config


@contextmanager
def create_window(window_name):
    """Create a window and ensure it's destroyed properly."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        yield
    finally:
        cv2.destroyWindow(window_name)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Stopping analysis...")
    cv2.destroyAllWindows()
    sys.exit(0)


def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1, help="Monitor to capture")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO model")
    args = parser.parse_args()

    # Initialize screen capture
    capture = ScreenCapture(monitor=args.monitor)

    # Initialize analyzer with custom model
    analyzer = LiveGameAnalyzer(args.model, conf_threshold=args.conf)

    print("✅ Using MSS for screen capture (fastest)")
    print(f"🔍 Loading custom YOLO model: {args.model}")
    print("\n🎮 Starting Live Game Analysis...")
    print("=" * 60)
    print("Features:")
    print("  🔍 HUD Region Detection (every frame)")
    print("  📝 Enhanced OCR System (30 FPS)")
    print("  🎯 Game State Tracking")
    print("=" * 60)
    print("Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 'f' to show current FPS")
    print("")

    try:
        while True:
            frame = capture.capture_frame()
            vis_frame, ocr_results = analyzer.process_frame(frame)

            cv2.imshow("Live Game Analysis", vis_frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("f"):
                print(f"Current FPS: {analyzer.get_fps():.1f}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping analysis...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
