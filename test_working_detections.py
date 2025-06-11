#!/usr/bin/env python3
"""
Working Detection System Testing on Unannotated Images
======================================================

This script tests the working detection systems on unannotated images and saves 
visual results to organized folders so you can see what everything is picking up.

Focus on systems that are currently working:
1. YOLOv8 HUD Region Detection (working perfectly)
2. Basic OCR text extraction  
3. Triangle detection via YOLO regions
"""

import cv2
import numpy as np
import json
from pathlib import Path
import time
import random
import os
from datetime import datetime

# Import detection systems
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available")
    YOLO_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    print("⚠️ EasyOCR not available")
    OCR_AVAILABLE = False


class WorkingDetectionTester:
    """Tests working detection systems on unannotated images."""
    
    def __init__(self, output_base_dir="working_detection_results"):
        self.output_base_dir = Path(output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"session_{self.timestamp}"
        
        # Create organized output directories
        self.dirs = {
            'yolo': self.session_dir / "yolo_hud_detection",
            'ocr': self.session_dir / "ocr_text_extraction",
            'combined': self.session_dir / "combined_analysis",
            'summary': self.session_dir / "summary_reports"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize detection systems
        self.init_detectors()
        
        # Results tracking
        self.results_summary = {
            'total_images': 0,
            'processing_times': {},
            'detection_counts': {},
            'errors': []
        }
    
    def init_detectors(self):
        """Initialize all available detection systems."""
        print("🔧 Initializing working detection systems...")
        
        self.detectors = {}
                
        # YOLO HUD Detection
        if YOLO_AVAILABLE:
            try:
                model_paths = [
                    "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt",
                    "yolov8_model.pt",
                    "best.pt"
                ]
                model_loaded = False
                for model_path in model_paths:
                    if Path(model_path).exists():
                        self.detectors['yolo'] = YOLO(model_path)
                        print(f"   ✅ YOLO HUD Detector initialized with {model_path}")
                        model_loaded = True
                        break
                if not model_loaded:
                    print(f"   ⚠️ YOLO model not found at any expected paths")
            except Exception as e:
                print(f"   ❌ YOLO HUD Detector failed: {e}")
                
        # OCR Text Detection
        if OCR_AVAILABLE:
            try:
                self.detectors['ocr'] = easyocr.Reader(['en'], gpu=True)
                print("   ✅ EasyOCR text detector initialized")
            except Exception as e:
                print(f"   ❌ EasyOCR failed: {e}")
    
    def find_test_images(self, max_images=20):
        """Find unannotated test images."""
        print("📂 Finding test images...")
        
        image_dirs = [
            Path("madden 6111"),
            Path("NEW MADDEN DATA"), 
            Path(".")
        ]
        
        all_images = []
        for img_dir in image_dirs:
            if img_dir.exists():
                patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
                for pattern in patterns:
                    found_imgs = list(img_dir.glob(pattern))
                    all_images.extend(found_imgs)
                    if not found_imgs:
                        # Try recursive search
                        found_imgs = list(img_dir.rglob(pattern))
                        all_images.extend(found_imgs)
        
        # Remove duplicate paths and sort
        unique_images = list(set(all_images))
        
        # Filter out annotation files
        filtered_images = []
        for img in unique_images:
            if not any(x in str(img).lower() for x in ["annotation", "label", "mask", "result"]):
                filtered_images.append(img)
        
        # Randomly sample for testing
        if len(filtered_images) > max_images:
            test_images = random.sample(filtered_images, max_images)
        else:
            test_images = filtered_images
            
        print(f"   Found {len(unique_images)} total images")
        print(f"   Filtered to {len(filtered_images)} non-annotation images")
        print(f"   Selected {len(test_images)} for testing")
        
        return test_images
    
    def test_yolo_detection(self, image, image_name):
        """Test YOLO HUD detection."""
        if 'yolo' not in self.detectors:
            return None, None
            
        try:
            start_time = time.time()
            results = self.detectors['yolo'](image)
            processing_time = time.time() - start_time
            
            # Create visualization
            vis_image = image.copy()
            
            detection_count = 0
            detection_details = []
            
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Get class names
                    class_names = result.names
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        if conf > 0.3:  # Lower confidence threshold for visibility
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = class_names[int(cls)]
                            
                            # Draw bounding box
                            color = self.get_class_color(class_name)
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(vis_image, f"{class_name}: {conf:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            detection_count += 1
                            
                            detection_details.append({
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
            
            self.add_detection_info_overlay(vis_image, {
                'system': 'YOLO HUD Detection',
                'detections_found': detection_count,
                'processing_time': f"{processing_time:.3f}s"
            })
            
            # Save result
            output_path = self.dirs['yolo'] / f"{image_name}_yolo.png"
            cv2.imwrite(str(output_path), vis_image)
            
            return {'detections': detection_details, 'count': detection_count}, processing_time
            
        except Exception as e:
            print(f"   ❌ YOLO detection failed: {e}")
            self.results_summary['errors'].append(f"YOLO - {image_name}: {e}")
            return None, None
    
    def get_class_color(self, class_name):
        """Get consistent colors for different detection classes."""
        colors = {
            'hud': (255, 0, 0),  # Blue
            'possession_triangle_area': (0, 255, 255),  # Yellow
            'territory_triangle_area': (0, 255, 0),  # Green
            'preplay_indicator': (255, 0, 255),  # Magenta
            'play_call_screen': (255, 165, 0)  # Orange
        }
        return colors.get(class_name, (128, 128, 128))  # Gray default
    
    def test_ocr_detection(self, image, image_name):
        """Test OCR text detection."""
        if 'ocr' not in self.detectors:
            return None, None
            
        try:
            start_time = time.time()
            
            # Run OCR on full image
            ocr_results = self.detectors['ocr'].readtext(image)
            processing_time = time.time() - start_time
            
            # Create visualization
            vis_image = image.copy()
            
            text_detections = []
            for i, (bbox, text, confidence) in enumerate(ocr_results):
                if confidence > 0.5:  # Filter low confidence text
                    # Convert bbox to integer coordinates
                    bbox = np.array(bbox).astype(int)
                    
                    # Draw bounding box around text
                    cv2.polylines(vis_image, [bbox], True, (0, 255, 255), 2)
                    
                    # Add text label
                    x, y = bbox[0]
                    cv2.putText(vis_image, f"'{text}' ({confidence:.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    text_detections.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox.tolist()
                    })
            
            self.add_detection_info_overlay(vis_image, {
                'system': 'OCR Text Detection',
                'text_found': len(text_detections),
                'processing_time': f"{processing_time:.3f}s"
            })
            
            # Save result
            output_path = self.dirs['ocr'] / f"{image_name}_ocr.png"
            cv2.imwrite(str(output_path), vis_image)
            
            return {'text_detections': text_detections, 'count': len(text_detections)}, processing_time
            
        except Exception as e:
            print(f"   ❌ OCR detection failed: {e}")
            self.results_summary['errors'].append(f"OCR - {image_name}: {e}")
            return None, None
    
    def create_combined_analysis(self, image, image_name, all_results):
        """Create a combined visualization showing all detection results."""
        vis_image = image.copy()
        
        # Add title
        cv2.putText(vis_image, f"Combined Analysis: {image_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        
        # Draw YOLO detections
        if 'yolo' in all_results and all_results['yolo'][0]:
            yolo_data = all_results['yolo'][0]
            for detection in yolo_data.get('detections', []):
                bbox = detection['bbox']
                class_name = detection['class']
                conf = detection['confidence']
                
                color = self.get_class_color(class_name)
                cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(vis_image, f"{class_name}: {conf:.2f}", 
                          (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw OCR detections
        if 'ocr' in all_results and all_results['ocr'][0]:
            ocr_data = all_results['ocr'][0]
            for detection in ocr_data.get('text_detections', []):
                bbox = np.array(detection['bbox']).astype(int)
                text = detection['text']
                conf = detection['confidence']
                
                cv2.polylines(vis_image, [bbox], True, (0, 255, 255), 1)
                x, y = bbox[0]
                cv2.putText(vis_image, f"'{text}'", 
                          (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add summary statistics
        y_offset = 70
        total_detections = 0
        
        for system, results in all_results.items():
            if results and results[0]:
                if system == 'yolo':
                    count = results[0].get('count', 0)
                    cv2.putText(vis_image, f"YOLO HUD Regions: {count}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    total_detections += count
                elif system == 'ocr':
                    count = results[0].get('count', 0)
                    cv2.putText(vis_image, f"OCR Text Elements: {count}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    total_detections += count
                
                y_offset += 30
        
        # Add total detections summary
        cv2.putText(vis_image, f"Total Detections: {total_detections}", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save combined result
        output_path = self.dirs['combined'] / f"{image_name}_combined.png"
        cv2.imwrite(str(output_path), vis_image)
        
        return vis_image
    
    def add_detection_info_overlay(self, image, info):
        """Add detection system information overlay to image."""
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Add dark background for text
        cv2.rectangle(overlay, (image.shape[1]-450, 10), (image.shape[1]-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add detection info
        y_start = 35
        x_start = image.shape[1] - 440
        cv2.putText(image, info['system'], (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_start += 25
        
        for key, value in info.items():
            if key != 'system':
                text = f"{key.replace('_', ' ').title()}: {value}"
                cv2.putText(image, text, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_start += 18
    
    def test_all_systems_on_image(self, image_path):
        """Test all detection systems on a single image."""
        image_name = Path(image_path).stem
        print(f"   🔍 Testing: {image_name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ❌ Could not load {image_path}")
            return
        
        all_results = {}
        total_time = 0
        
        # Test each system
        systems = [
            ('yolo', self.test_yolo_detection),
            ('ocr', self.test_ocr_detection)
        ]
        
        for system_name, test_func in systems:
            if system_name in self.detectors:
                print(f"     🧪 Testing {system_name}...")
                results, proc_time = test_func(image, image_name)
                all_results[system_name] = (results, proc_time)
                if proc_time:
                    total_time += proc_time
        
        # Create combined visualization
        self.create_combined_analysis(image, image_name, all_results)
        
        # Update summary
        self.results_summary['processing_times'][image_name] = total_time
        
        print(f"     ✅ Completed in {total_time:.3f}s")
        return all_results
    
    def generate_summary_report(self):
        """Generate a summary report of all testing."""
        report_path = self.dirs['summary'] / "testing_summary.json"
        
        # Calculate statistics
        if self.results_summary['processing_times']:
            avg_time = np.mean(list(self.results_summary['processing_times'].values()))
            total_time = sum(self.results_summary['processing_times'].values())
        else:
            avg_time = 0
            total_time = 0
        
        summary = {
            'session_timestamp': self.timestamp,
            'total_images_tested': self.results_summary['total_images'],
            'systems_tested': list(self.detectors.keys()),
            'average_processing_time': avg_time,
            'total_processing_time': total_time,
            'errors_encountered': len(self.results_summary['errors']),
            'error_details': self.results_summary['errors'],
            'output_directories': {k: str(v) for k, v in self.dirs.items()}
        }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run_comprehensive_test(self):
        """Run comprehensive testing on all systems."""
        print("🚀 Starting Working Detection System Testing")
        print("=" * 60)
        
        # Find test images
        test_images = self.find_test_images()
        if not test_images:
            print("❌ No test images found!")
            return
        
        self.results_summary['total_images'] = len(test_images)
        
        print(f"🎯 Testing {len(test_images)} images with {len(self.detectors)} detection systems")
        print(f"📁 Results will be saved to: {self.session_dir}")
        print("-" * 60)
        
        # Test each image
        for i, image_path in enumerate(test_images, 1):
            print(f"\n📸 Image {i}/{len(test_images)}: {Path(image_path).name}")
            self.test_all_systems_on_image(image_path)
        
        # Generate summary report
        print("\n📊 Generating summary report...")
        summary = self.generate_summary_report()
        
        # Print final results
        print("\n" + "=" * 60)
        print("🎉 WORKING DETECTION TESTING COMPLETE!")
        print("=" * 60)
        print(f"📊 **SUMMARY:**")
        print(f"   🖼️  Images tested: {summary['total_images_tested']}")
        print(f"   🧪 Systems tested: {len(summary['systems_tested'])}")
        print(f"   ⚡ Average processing time: {summary['average_processing_time']:.3f}s")
        print(f"   🕒 Total processing time: {summary['total_processing_time']:.3f}s")
        print(f"   ❌ Errors encountered: {summary['errors_encountered']}")
        print(f"\n📁 **RESULTS FOLDERS:**")
        for folder_name, folder_path in summary['output_directories'].items():
            print(f"   {folder_name}: {folder_path}")
        print(f"\n✅ Summary report: {self.dirs['summary'] / 'testing_summary.json'}")
        print("=" * 60)


def main():
    """Main testing function."""
    print("🎯 Working Detection System Testing")
    print("Testing working detection systems on unannotated images")
    print("Results will be saved with visual overlays for easy review\n")
    
    # Set random seed for reproducible image selection
    random.seed(42)
    
    # Create and run tester
    tester = WorkingDetectionTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main() 