#!/usr/bin/env python3
"""Test the trained YOLO model on video files with real-time detection visualization."""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import argparse

def load_trained_model():
    """Load the latest trained model."""
    # Look for the latest training run
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("❌ No training runs found!")
        return None
    
    # Find the latest spygate_hud_detection_fast run
    model_dirs = list(runs_dir.glob("spygate_hud_detection_fast*"))
    if not model_dirs:
        print("❌ No spygate_hud_detection_fast runs found!")
        return None
    
    # Get the latest one (by modification time)
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Model weights not found at {model_path}")
        return None
    
    print(f"✅ Loading model from: {model_path}")
    return YOLO(str(model_path))

def process_video(video_path, model, output_path=None, conf_threshold=0.15):
    """Process video and detect HUD elements."""
    
    # Class names and colors
    class_names = ["hud", "qb_position", "left_hash_mark", "right_hash_mark", "preplay", "playcall"]
    colors = [
        (0, 0, 255),    # hud - red
        (0, 255, 0),    # qb_position - green
        (255, 0, 0),    # left_hash_mark - blue
        (0, 255, 255),  # right_hash_mark - yellow
        (255, 0, 255),  # preplay - magenta
        (255, 255, 0),  # playcall - cyan
    ]
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if output path specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"📝 Output will be saved to: {output_path}")
    
    # Detection statistics
    detection_stats = {name: 0 for name in class_names}
    frame_count = 0
    total_inference_time = 0
    
    print(f"\n🎬 Processing video...")
    print("=" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            results = model(frame, conf=conf_threshold, iou=0.45, verbose=False)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Draw detections
            detections_this_frame = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for j in range(len(boxes)):
                        # Get box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                        cls_id = int(boxes.cls[j].item())
                        conf = boxes.conf[j].item()
                        
                        if cls_id < len(class_names):
                            class_name = class_names[cls_id]
                            color = colors[cls_id]
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label with background
                            label = f"{class_name}: {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Count detection
                            detection_stats[class_name] += 1
                            detections_this_frame.append(f"{class_name}({conf:.2f})")
            
            # Add frame info overlay
            info_text = f"Frame: {frame_count}/{total_frames} | FPS: {1/inference_time:.1f} | Detections: {len(detections_this_frame)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection summary overlay
            y_offset = 60
            for i, (class_name, count) in enumerate(detection_stats.items()):
                if count > 0:
                    color = colors[i]
                    text = f"{class_name}: {count}"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 25
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
                print(f"Progress: {progress:.1f}% | Avg FPS: {avg_fps:.1f} | Detections this batch: {detections_this_frame}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n📊 FINAL DETECTION STATISTICS:")
    print("=" * 60)
    print(f"📹 Processed {frame_count} frames")
    print(f"⚡ Average inference time: {total_inference_time/frame_count*1000:.1f}ms per frame")
    print(f"🎯 Average FPS: {frame_count/total_inference_time:.1f}")
    
    print(f"\n🎯 DETECTIONS BY CLASS:")
    print("=" * 40)
    total_detections = sum(detection_stats.values())
    for class_name, count in detection_stats.items():
        if count > 0:
            percentage = (count / total_detections) * 100 if total_detections > 0 else 0
            status = "✅" if count > frame_count * 0.1 else "⚠️" if count > 0 else "❌"
            print(f"{status} {class_name:<15}: {count:>4} detections ({percentage:.1f}%)")
        else:
            print(f"❌ {class_name:<15}: {count:>4} detections")
    
    print(f"\n🏆 PERFORMANCE ASSESSMENT:")
    print("=" * 40)
    if detection_stats["hud"] > frame_count * 0.3:
        print("✅ HUD Detection: EXCELLENT")
    elif detection_stats["hud"] > frame_count * 0.1:
        print("⚠️  HUD Detection: GOOD")
    else:
        print("❌ HUD Detection: POOR")
    
    hashmarks = detection_stats["left_hash_mark"] + detection_stats["right_hash_mark"]
    if hashmarks > frame_count * 0.1:
        print("✅ Hash Mark Detection: GOOD")
    elif hashmarks > 0:
        print("⚠️  Hash Mark Detection: FAIR")
    else:
        print("❌ Hash Mark Detection: POOR")
    
    if detection_stats["qb_position"] > frame_count * 0.05:
        print("✅ QB Position Detection: GOOD")
    elif detection_stats["qb_position"] > 0:
        print("⚠️  QB Position Detection: FAIR")
    else:
        print("❌ QB Position Detection: POOR")
    
    game_state = detection_stats["preplay"] + detection_stats["playcall"]
    if game_state > 0:
        print(f"✅ Game State Detection: WORKING ({game_state} detections)")
    else:
        print("❌ Game State Detection: NOT WORKING")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Test YOLO model on video")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Path to save output video with detections")
    parser.add_argument("--conf", "-c", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    
    args = parser.parse_args()
    
    print("🏈 SpygateAI Video Detection Test")
    print("=" * 60)
    
    # Check if video file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Load model
    model = load_trained_model()
    if model is None:
        print("❌ Could not load trained model!")
        return
    
    # Process video
    output_path = Path(args.output) if args.output else None
    process_video(video_path, model, output_path, args.conf)
    
    print("\n🎉 Video processing complete!")
    if output_path:
        print(f"📹 Output saved to: {output_path}")

if __name__ == "__main__":
    main() 