#!/usr/bin/env python3
"""
Diagnostic script to sample video at different timestamps and extract down/distance regions.
This will help us see if the video actually contains different downs or if OCR is over-correcting.
"""

import os
import sys

import cv2
import numpy as np

sys.path.append("src")

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def extract_frame_at_time(video_path, timestamp_seconds):
    """Extract frame at specific timestamp."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def save_down_distance_region(frame, analyzer, timestamp, output_name):
    """Extract and save down/distance region from frame."""
    if frame is None:
        print(f"❌ No frame at {timestamp}s")
        return None

    # Resize to burst sampling resolution
    frame = cv2.resize(frame, (854, 480))

    # Run YOLO detection
    results = analyzer.model.predict(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = analyzer.model.names[class_id]
                conf = float(box.conf[0])

                if class_name == "down_distance_area" and conf > 0.8:
                    # Extract region
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    region = frame[y1:y2, x1:x2]

                    # Save original region
                    cv2.imwrite(f"diagnostic_{output_name}_original.png", region)

                    # Save processed region
                    processed = analyzer._preprocess_region_for_ocr(region)
                    cv2.imwrite(f"diagnostic_{output_name}_processed.png", processed)

                    # Test OCR
                    import pytesseract

                    raw_text = pytesseract.image_to_string(
                        processed, config=r"--oem 3 --psm 7"
                    ).strip()
                    corrected_text = analyzer._apply_down_distance_corrections(raw_text)

                    print(
                        f"⏰ {timestamp}s: Raw: '{raw_text}' → Corrected: '{corrected_text}' (conf: {conf:.3f})"
                    )

                    return {
                        "timestamp": timestamp,
                        "raw_text": raw_text,
                        "corrected_text": corrected_text,
                        "confidence": conf,
                        "region_shape": region.shape,
                    }

    print(f"❌ No down_distance_area found at {timestamp}s")
    return None


def main():
    """Sample video at different timestamps to diagnose the issue."""
    video_path = "1 min 30 test clip.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    print("🔍 DIAGNOSTIC: Sampling video at different timestamps")
    print("=" * 60)

    # Initialize analyzer
    analyzer = EnhancedGameAnalyzer()

    # Sample timestamps spread across the video
    timestamps = [10, 25, 40, 55, 70, 85]  # Every ~15 seconds

    results = []

    for timestamp in timestamps:
        print(f"\n📍 Sampling at {timestamp}s...")
        frame = extract_frame_at_time(video_path, timestamp)
        result = save_down_distance_region(frame, analyzer, timestamp, f"{timestamp}s")
        if result:
            results.append(result)

    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY:")
    print("=" * 60)

    if not results:
        print("❌ No down/distance regions detected in any samples")
        return

    # Analyze patterns
    raw_texts = [r["raw_text"] for r in results]
    corrected_texts = [r["corrected_text"] for r in results]

    print(f"📈 Samples collected: {len(results)}")
    print(f"📝 Unique raw texts: {len(set(raw_texts))}")
    print(f"📝 Unique corrected texts: {len(set(corrected_texts))}")

    print("\n🔍 Raw OCR Results:")
    for i, text in enumerate(set(raw_texts)):
        count = raw_texts.count(text)
        print(f"   '{text}' → {count} times")

    print("\n🔧 Corrected Results:")
    for i, text in enumerate(set(corrected_texts)):
        count = corrected_texts.count(text)
        print(f"   '{text}' → {count} times")

    # Diagnosis
    if len(set(raw_texts)) > 1 and len(set(corrected_texts)) == 1:
        print("\n🚨 DIAGNOSIS: OCR over-correction detected!")
        print("   → Raw OCR shows variety, but correction makes everything the same")
        print("   → Solution: Make corrections less aggressive")
    elif len(set(raw_texts)) == 1:
        print("\n🤔 DIAGNOSIS: Video might actually show same down/distance")
        print("   → All raw OCR reads the same text")
        print("   → Check if video contains different game situations")
    else:
        print("\n✅ DIAGNOSIS: OCR working correctly")
        print("   → Both raw and corrected show variety")


if __name__ == "__main__":
    main()
