#!/usr/bin/env python3

"""
Debug Video Analysis - Fix 30% Freeze Issue
==========================================

This script identifies and fixes the video analysis freeze at 30%.
"""

import os
import sys
import threading
import time
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtWidgets import *
    print("✅ PyQt6 and OpenCV imported successfully")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)

class DebugAutoClipDetector(QObject):
    """Debug version of AutoClipDetector with extensive logging."""
    
    analysis_progress = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.is_analyzing = False
        
    def analyze_video(self, video_path: str):
        """Debug video analysis with detailed logging."""
        print(f"🔍 DEBUG: Starting analysis of {video_path}")
        
        try:
            # Test video file opening
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ ERROR: Cannot open video file: {video_path}")
                return False
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📹 Video info:")
            print(f"  - Total frames: {total_frames}")
            print(f"  - FPS: {fps}")
            print(f"  - Duration: {duration:.2f} seconds")
            
            if total_frames == 0:
                print("❌ ERROR: Video has 0 frames")
                return False
                
            self.is_analyzing = True
            frame_count = 0
            frames_processed = 0
            
            while cap.isOpened() and self.is_analyzing:
                ret, frame = cap.read()
                if not ret:
                    print(f"📹 End of video reached at frame {frame_count}")
                    break
                    
                frame_count += 1
                
                # Process every 30th frame to simulate the analysis
                if frame_count % 30 == 0:
                    frames_processed += 1
                    
                    # Calculate progress
                    progress = int((frame_count / total_frames) * 100)
                    
                    print(f"🔄 Frame {frame_count}/{total_frames} - Progress: {progress}%")
                    
                    # Emit progress signal
                    self.analysis_progress.emit(progress, f"Processing frame {frame_count}")
                    
                    # Check for the 30% point specifically
                    if 25 <= progress <= 35:
                        print(f"🎯 CRITICAL ZONE: {progress}% - This is where freeze might occur")
                        time.sleep(0.1)  # Small delay to check for threading issues
                        
                    # Simulate some processing
                    if frame is not None:
                        # Basic frame processing to test memory/CPU
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        _ = np.mean(gray)  # Simple computation
                        
                    # Yield control to other threads
                    QApplication.processEvents()
                    
                # Stop if we reach 50% for debugging
                if frame_count > total_frames * 0.5:
                    print("🛑 Stopping at 50% for debugging")
                    break
                    
            cap.release()
            self.analysis_complete.emit(True)
            print("✅ Analysis completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ ERROR in video analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.is_analyzing = False

class DebugWindow(QMainWindow):
    """Debug window to test video analysis."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Debug Video Analysis - Fix 30% Freeze")
        self.setGeometry(100, 100, 800, 600)
        self.detector = DebugAutoClipDetector()
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.browse_btn = QPushButton("Browse Video")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Analyze button
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        central_widget.setLayout(layout)
        
    def connect_signals(self):
        self.detector.analysis_progress.connect(self.update_progress)
        self.detector.analysis_complete.connect(self.analysis_finished)
        
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected: {Path(file_path).name}")
            self.analyze_btn.setEnabled(True)
            self.log(f"File selected: {file_path}")
            
    def start_analysis(self):
        if not hasattr(self, 'file_path'):
            return
            
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log("Starting analysis...")
        
        # Start analysis in thread to prevent UI freeze
        self.analysis_thread = threading.Thread(
            target=self.detector.analyze_video, 
            args=(self.file_path,)
        )
        self.analysis_thread.daemon = True  # Important: make thread daemon
        self.analysis_thread.start()
        
    def update_progress(self, value: int, message: str):
        """Update progress - this should work from thread via Qt's signal system."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.log(f"Progress: {value}% - {message}")
        
        # Force UI update
        QApplication.processEvents()
        
    def analysis_finished(self, success: bool):
        self.analyze_btn.setEnabled(True)
        if success:
            self.log("✅ Analysis completed successfully!")
            self.status_label.setText("Analysis complete")
        else:
            self.log("❌ Analysis failed!")
            self.status_label.setText("Analysis failed")
            
    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

def main():
    """Main function to run the debug application."""
    print("🔍 DEBUG: Starting Video Analysis Debug Tool")
    
    app = QApplication(sys.argv)
    
    # Test basic Qt functionality
    print("✅ QApplication created successfully")
    
    window = DebugWindow()
    window.show()
    
    print("✅ Debug window created and shown")
    print("📋 Instructions:")
    print("  1. Click 'Browse Video' to select a video file")
    print("  2. Click 'Start Analysis' to begin")
    print("  3. Watch for freeze around 30% progress")
    print("  4. Check console output for detailed logging")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 