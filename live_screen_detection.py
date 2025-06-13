#!/usr/bin/env python3
"""Live screen capture with real-time YOLO triangle detection for SpygateAI."""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import argparse
import threading
from collections import deque

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("⚠️  mss not available, trying alternative screen capture...")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

def load_trained_model(model_path=None):
    """Load the latest trained triangle detection model."""
    # If custom model path provided, use it
    if model_path and Path(model_path).exists():
        print(f"✅ Loading custom model from: {model_path}")
        return YOLO(str(model_path))
    
    # First try the latest HUD regions model
    hud_model_path = Path("hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt")
    if hud_model_path.exists():
        print(f"✅ Loading NEW HUD MODEL from: {hud_model_path}")
        return YOLO(str(hud_model_path))
    
    # Fallback to any recent spygate model
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("❌ No training runs found!")
        return None
    
    model_dirs = list(runs_dir.glob("spygate*"))
    if not model_dirs:
        print("❌ No spygate detection runs found!")
        return None
    
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Model weights not found at {model_path}")
        return None
    
    print(f"✅ Loading fallback model from: {model_path}")
    return YOLO(str(model_path))

class ScreenCapture:
    """Screen capture class with multiple backends."""
    
    def __init__(self, monitor=1, region=None):
        self.monitor = monitor
        self.region = region
        self.method = None
        self.sct = None
        
        # Try different capture methods
        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
                self.method = "mss"
                print("✅ Using MSS for screen capture (fastest)")
            except Exception as e:
                print(f"❌ MSS failed: {e}")
        
        if self.method is None and PYAUTOGUI_AVAILABLE:
            try:
                # Test pyautogui
                test_img = pyautogui.screenshot()
                self.method = "pyautogui"
                print("✅ Using PyAutoGUI for screen capture")
            except Exception as e:
                print(f"❌ PyAutoGUI failed: {e}")
        
        if self.method is None:
            raise RuntimeError("❌ No screen capture method available! Install: pip install mss pyautogui")
    
    def get_monitors(self):
        """Get available monitors."""
        if self.method == "mss" and self.sct:
            return self.sct.monitors
        else:
            # Fallback - assume single monitor
            import tkinter as tk
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return [{"top": 0, "left": 0, "width": width, "height": height}]
    
    def capture(self):
        """Capture screen and return as OpenCV image."""
        if self.method == "mss" and self.sct:
            # MSS capture
            monitors = self.sct.monitors
            if self.monitor < len(monitors):
                monitor = monitors[self.monitor]
                if self.region:
                    # Use specific region
                    monitor = {
                        "top": self.region[1],
                        "left": self.region[0],
                        "width": self.region[2],
                        "height": self.region[3]
                    }
                
                sct_img = self.sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        
        elif self.method == "pyautogui":
            # PyAutoGUI capture
            if self.region:
                img = pyautogui.screenshot(region=self.region)
            else:
                img = pyautogui.screenshot()
            
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        
        return None

def live_detection(model, conf_threshold=0.15, monitor=1, region=None, save_video=None, show_preview=True):
    """Run live detection on screen capture with TRIANGLE DETECTION!"""
    
    # Updated class names and colors for HUD region detection
    class_names = [
        "hud",                         # Main HUD bar
        "possession_triangle_area",    # LEFT side triangle area
        "territory_triangle_area",     # RIGHT side triangle area
        "preplay_indicator",          # Pre-play UI element
        "play_call_screen"            # Play selection UI
    ]
    
    colors = [
        (0, 0, 255),      # hud - red
        (0, 255, 128),    # possession_triangle_area - lime green
        (255, 128, 0),    # territory_triangle_area - orange
        (255, 0, 255),    # preplay_indicator - magenta
        (255, 255, 0),    # play_call_screen - cyan
    ]
    
    # Initialize screen capture
    try:
        capture = ScreenCapture(monitor=monitor, region=region)
    except RuntimeError as e:
        print(e)
        return
    
    # Detection statistics
    detection_stats = {name: 0 for name in class_names}
    frame_count = 0
    fps_queue = deque(maxlen=30)  # Keep last 30 frame times for FPS calculation
    
    # Triangle detection tracking
    triangle_stats = {
        "possession_detected": 0,
        "territory_detected": 0,
        "both_detected": 0
    }
    
    # Video writer setup
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # We'll initialize this after getting the first frame to know dimensions
    
    print(f"\n🎬 Starting live TRIANGLE detection...")
    print("=" * 60)
    print("🔺 NEW FEATURES:")
    print("  🔺 Possession Indicator Detection (LEFT triangle)")
    print("  🔺 Territory Indicator Detection (RIGHT triangle)")
    print("  🎯 All 8 HUD classes supported!")
    print("=" * 60)
    print("Controls:")
    print("  🎯 Press 'q' to quit")
    print("  📊 Press 's' to show statistics")
    print("  🎥 Press 'r' to reset statistics")
    print("  🔺 Press 't' to show triangle stats")
    if show_preview:
        print("  👁️  Preview window will show detections")
    print("=" * 60)
    
    try:
        while True:
            start_time = time.time()
            
            # Capture screen
            frame = capture.capture()
            if frame is None:
                print("❌ Failed to capture screen")
                break
            
            frame_count += 1
            
            # Initialize video writer with actual frame dimensions
            if save_video and out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(save_video, fourcc, 10.0, (width, height))
                print(f"📹 Recording to: {save_video}")
            
            # Run detection
            detection_start = time.time()
            results = model(frame, conf=conf_threshold, iou=0.45, verbose=False)
            detection_time = time.time() - detection_start
            
            # Track triangles in this frame
            possession_detected = False
            territory_detected = False
            
            # Draw detections
            detections_this_frame = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                        cls_id = int(boxes.cls[j].item())
                        conf = boxes.conf[j].item()
                        
                        if cls_id < len(class_names):
                            class_name = class_names[cls_id]
                            color = colors[cls_id]
                            
                            # Special handling for triangles
                            if class_name == "possession_triangle_area":
                                possession_detected = True
                                # Extra thick border for triangles
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                                # Add triangle symbol
                                label = f"🔺 POSSESSION: {conf:.2f}"
                            elif class_name == "territory_triangle_area":
                                territory_detected = True
                                # Extra thick border for triangles
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                                # Add triangle symbol
                                label = f"🔺 TERRITORY: {conf:.2f}"
                            else:
                                # Normal bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                label = f"{class_name}: {conf:.2f}"
                            
                            # Draw label with background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1-label_size[1]-15), (x1+label_size[0]+10, y1), color, -1)
                            cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Count detection
                            detection_stats[class_name] += 1
                            detections_this_frame.append(f"{class_name}({conf:.2f})")
            
            # Update triangle stats
            if possession_detected:
                triangle_stats["possession_detected"] += 1
            if territory_detected:
                triangle_stats["territory_detected"] += 1
            if possession_detected and territory_detected:
                triangle_stats["both_detected"] += 1
            
            # Calculate FPS
            frame_time = time.time() - start_time
            fps_queue.append(frame_time)
            avg_fps = len(fps_queue) / sum(fps_queue) if fps_queue else 0
            
            # Add performance overlay
            height, width = frame.shape[:2]
            overlay_y = 40
            
            # FPS and detection info
            info_text = f"FPS: {avg_fps:.1f} | Detection: {detection_time*1000:.1f}ms | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            overlay_y += 35
            
            # Triangle status
            triangle_text = f"🔺 Triangles: "
            if possession_detected and territory_detected:
                triangle_text += "BOTH DETECTED!"
                triangle_color = (0, 255, 0)  # Green
            elif possession_detected:
                triangle_text += "POSSESSION ONLY"
                triangle_color = (0, 255, 128)  # Lime
            elif territory_detected:
                triangle_text += "TERRITORY ONLY"
                triangle_color = (255, 128, 0)  # Orange
            else:
                triangle_text += "NONE DETECTED"
                triangle_color = (128, 128, 128)  # Gray
            
            cv2.putText(frame, triangle_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, triangle_color, 2)
            overlay_y += 35
            
            # Current detections
            if detections_this_frame:
                det_text = f"Current: {', '.join(detections_this_frame[:4])}"  # Show first 4
                cv2.putText(frame, det_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                overlay_y += 30
            
            # Detection counters (only show non-zero)
            for i, (class_name, count) in enumerate(detection_stats.items()):
                if count > 0:
                    color = colors[i]
                    # Special display for triangles
                    if "triangle_area" in class_name:
                        text = f"🔺 {class_name}: {count}"
                    else:
                        text = f"{class_name}: {count}"
                    cv2.putText(frame, text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    overlay_y += 25
            
            # Save frame to video
            if out:
                out.write(frame)
            
            # Show preview window
            if show_preview:
                # Resize for display if too large
                display_frame = frame
                if width > 1920 or height > 1080:
                    scale = min(1920/width, 1080/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imshow('SpygateAI Live TRIANGLE Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopping live detection...")
                    break
                elif key == ord('s'):
                    print_statistics(detection_stats, frame_count, avg_fps, triangle_stats)
                elif key == ord('t'):
                    print_triangle_statistics(triangle_stats, frame_count)
                elif key == ord('r'):
                    print("\n🔄 Resetting statistics...")
                    detection_stats = {name: 0 for name in class_names}
                    triangle_stats = {"possession_detected": 0, "territory_detected": 0, "both_detected": 0}
                    frame_count = 0
                    fps_queue.clear()
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                total_detections = sum(detection_stats.values())
                triangle_rate = (triangle_stats["both_detected"] / frame_count) * 100 if frame_count > 0 else 0
                print(f"Frame {frame_count} | FPS: {avg_fps:.1f} | Total: {total_detections} | 🔺 Both triangles: {triangle_rate:.1f}%")
    
    except KeyboardInterrupt:
        print("\n⏹️  Live detection interrupted by user")
    
    finally:
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print_final_statistics(detection_stats, frame_count, fps_queue, triangle_stats)

def print_triangle_statistics(triangle_stats, frame_count):
    """Print triangle-specific statistics."""
    print(f"\n🔺 TRIANGLE DETECTION STATISTICS:")
    print("=" * 50)
    
    possession_rate = (triangle_stats["possession_detected"] / frame_count) * 100 if frame_count > 0 else 0
    territory_rate = (triangle_stats["territory_detected"] / frame_count) * 100 if frame_count > 0 else 0
    both_rate = (triangle_stats["both_detected"] / frame_count) * 100 if frame_count > 0 else 0
    
    print(f"🔺 Possession Triangle: {triangle_stats['possession_detected']} frames ({possession_rate:.1f}%)")
    print(f"🔺 Territory Triangle: {triangle_stats['territory_detected']} frames ({territory_rate:.1f}%)")
    print(f"🔺 Both Triangles: {triangle_stats['both_detected']} frames ({both_rate:.1f}%)")

def print_statistics(detection_stats, frame_count, avg_fps, triangle_stats=None):
    """Print current statistics."""
    print(f"\n📊 CURRENT STATISTICS (Frame {frame_count}):")
    print("=" * 50)
    print(f"⚡ FPS: {avg_fps:.1f}")
    
    # Triangle stats first (if available)
    if triangle_stats:
        print_triangle_statistics(triangle_stats, frame_count)
        print("=" * 50)
    
    total_detections = sum(detection_stats.values())
    for class_name, count in detection_stats.items():
        if count > 0:
            rate = (count / frame_count) * 100 if frame_count > 0 else 0
            symbol = "🔺" if "triangle_area" in class_name else "🎯"
            print(f"{symbol} {class_name}: {count} ({rate:.1f}% of frames)")
    print(f"📈 Total detections: {total_detections}")

def print_final_statistics(detection_stats, frame_count, fps_queue, triangle_stats=None):
    """Print final statistics."""
    avg_fps = len(fps_queue) / sum(fps_queue) if fps_queue else 0
    
    print(f"\n📊 FINAL LIVE TRIANGLE DETECTION STATISTICS:")
    print("=" * 60)
    print(f"📹 Processed {frame_count} frames")
    print(f"🎯 Average FPS: {avg_fps:.1f}")
    
    # Triangle performance first
    if triangle_stats:
        print(f"\n🔺 TRIANGLE DETECTION PERFORMANCE:")
        print("=" * 40)
        possession_rate = (triangle_stats["possession_detected"] / frame_count) * 100 if frame_count > 0 else 0
        territory_rate = (triangle_stats["territory_detected"] / frame_count) * 100 if frame_count > 0 else 0
        both_rate = (triangle_stats["both_detected"] / frame_count) * 100 if frame_count > 0 else 0
        
        status_p = "✅" if possession_rate > 20 else "⚠️" if possession_rate > 5 else "❌"
        status_t = "✅" if territory_rate > 20 else "⚠️" if territory_rate > 5 else "❌"
        status_b = "✅" if both_rate > 10 else "⚠️" if both_rate > 2 else "❌"
        
        print(f"{status_p} Possession Triangle: {triangle_stats['possession_detected']} frames ({possession_rate:.1f}%)")
        print(f"{status_t} Territory Triangle: {triangle_stats['territory_detected']} frames ({territory_rate:.1f}%)")
        print(f"{status_b} Both Triangles: {triangle_stats['both_detected']} frames ({both_rate:.1f}%)")
    
    print(f"\n🎯 DETECTIONS BY CLASS:")
    print("=" * 40)
    total_detections = sum(detection_stats.values())
    
    for class_name, count in detection_stats.items():
        if count > 0:
            rate = (count / frame_count) * 100 if frame_count > 0 else 0
            if "triangle_area" in class_name:
                status = "✅" if rate > 20 else "⚠️" if count > 0 else "❌"
                symbol = "🔺"
            else:
                status = "✅" if rate > 10 else "⚠️" if count > 0 else "❌"
                symbol = ""
            print(f"{status} {symbol} {class_name:<20}: {count:>4} detections ({rate:.1f}% of frames)")
        else:
            symbol = "🔺" if "triangle_area" in class_name else ""
            print(f"❌ {symbol} {class_name:<20}: {count:>4} detections")
    
    print(f"\n🏆 OVERALL PERFORMANCE ASSESSMENT:")
    print("=" * 40)
    hud_rate = (detection_stats["hud"] / frame_count) * 100 if frame_count > 0 else 0
    if hud_rate > 30:
        print("✅ HUD Detection: EXCELLENT")
    elif hud_rate > 10:
        print("⚠️  HUD Detection: GOOD")
    else:
        print("❌ HUD Detection: POOR")
    
    if triangle_stats:
        both_rate = (triangle_stats["both_detected"] / frame_count) * 100 if frame_count > 0 else 0
        if both_rate > 10:
            print("✅ Triangle Detection: EXCELLENT")
        elif both_rate > 2:
            print("⚠️  Triangle Detection: GOOD")
        else:
            print("❌ Triangle Detection: NEEDS IMPROVEMENT")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Live screen detection with YOLO")
    parser.add_argument("--conf", "-c", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    parser.add_argument("--monitor", "-m", type=int, default=1, help="Monitor number to capture (default: 1)")
    parser.add_argument("--region", "-r", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), 
                       help="Capture region: x y width height")
    parser.add_argument("--save", "-s", help="Save video to file")
    parser.add_argument("--no-preview", action="store_true", help="Don't show preview window")
    parser.add_argument("--model", type=str, help="Path to custom model weights")
    
    args = parser.parse_args()
    
    print("🏈 SpygateAI Live Screen Detection")
    print("=" * 60)
    
    # Load model with custom path if provided
    model = load_trained_model(args.model)
    if model is None:
        print("❌ Could not load trained model!")
        return
    
    # Check dependencies
    if not MSS_AVAILABLE and not PYAUTOGUI_AVAILABLE:
        print("❌ No screen capture library available!")
        print("Install with: pip install mss pyautogui")
        return
    
    # Start live detection
    live_detection(
        model=model,
        conf_threshold=args.conf,
        monitor=args.monitor,
        region=args.region,
        save_video=args.save,
        show_preview=not args.no_preview
    )
    
    print("\n🎉 Live detection complete!")

if __name__ == "__main__":
    main() 