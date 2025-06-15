import logging
from pathlib import Path

import cv2
import numpy as np

from ..core.hardware import HardwareDetector
from .enhanced_game_analyzer import EnhancedGameAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_screenshot(image_path: str, debug: bool = True):
    """Analyze a single screenshot and show triangle detection."""

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create debug output directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)

    # Initialize analyzer with debug output
    analyzer = EnhancedGameAnalyzer(
        hardware=HardwareDetector(), debug_output_dir=debug_dir if debug else None
    )

    # Run analysis
    game_state = analyzer.analyze_frame(frame)

    # Save debug visualization
    if debug:
        debug_frame = frame.copy()

        # Draw YOLO detections
        for detection in game_state.get("detections", []):
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            cls = detection["class"]

            # Draw box
            cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Add label
            cv2.putText(
                debug_frame,
                f"{cls} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Save visualization
        cv2.imwrite(str(debug_dir / "analyzed_frame.png"), debug_frame)

    return game_state


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m spygate.ml.analyze_screenshot <image_path>")
        sys.exit(1)

    result = analyze_screenshot(sys.argv[1])
    print("Analysis result:", result)
