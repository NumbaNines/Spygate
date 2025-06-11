#!/usr/bin/env python3
"""
Comprehensive Testing of All Detection Systems on Unannotated Images
===================================================================

This script tests all available detection systems on unannotated images from the 
madden 6111 folder and saves visual results to organized folders so you can see 
what everything is picking up.

Detection Systems Tested:
1. Production Game Analyzer (complete analysis pipeline)
2. Game Situation Analysis (HUD OCR + situation detection)  
3. YOLOv8 HUD Region Detection
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
    from game_situation_analyzer import GameSituationAnalyzer
    SITUATION_AVAILABLE = True
except ImportError:
    print("⚠️ Game Situation Analyzer not available")  
    SITUATION_AVAILABLE = False

try:
    from production_game_analyzer import ProductionGameAnalyzer
    PRODUCTION_AVAILABLE = True
except ImportError:
    print("⚠️ Production Game Analyzer not available")
    PRODUCTION_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available")
    YOLO_AVAILABLE = False


class ComprehensiveDetectionTester:
    """Tests all available detection systems on unannotated images."""
    
    def __init__(self, output_base_dir="detection_test_results"):
        self.output_base_dir = Path(output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"session_{self.timestamp}"
        
        # Create organized output directories
        self.dirs = {
            'situation': self.session_dir / "game_situation_analysis", 
            'yolo': self.session_dir / "yolo_hud_detection",
            'production': self.session_dir / "production_analysis",
            'combined': self.session_dir / "combined_overlay",
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
        print("🔧 Initializing detection systems...")
        
        self.detectors = {}
                
        # Production Game Analyzer (most comprehensive)
        if PRODUCTION_AVAILABLE:
            try:
                self.detectors['production'] = ProductionGameAnalyzer()
                print("   ✅ Production Game Analyzer initialized")
            except Exception as e:
                print(f"   ❌ Production Game Analyzer failed: {e}")
                
        # Game Situation Analyzer
        if SITUATION_AVAILABLE:
            try:
                self.detectors['situation'] = GameSituationAnalyzer()
                print("   ✅ Game Situation Analyzer initialized")
            except Exception as e:
                print(f"   ❌ Game Situation Analyzer failed: {e}")
                
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
    
    def find_test_images(self, max_images=15):
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
    
    def test_production_analysis(self, image, image_name):
        """Test production game analyzer."""
        if 'production' not in self.detectors:
            return None, None
            
        try:
            start_time = time.time()
            results = self.detectors['production'].analyze_game_frame(image)
            processing_time = time.time() - start_time
            
            # Create visualization
            vis_image = image.copy()
            
            if results:
                # Draw production analysis results
                text_y = 50
                if 'possession_analysis' in results:
                    poss = results['possession_analysis']
                    text = f"Possession: {poss.get('team', 'Unknown')} ({poss.get('confidence', 0)*100:.1f}%)"
                    cv2.putText(vis_image, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    text_y += 30
                
                if 'game_situation' in results:
                    sit = results['game_situation']
                    if 'down_distance' in sit and sit['down_distance']:
                        text = f"Down & Distance: {sit['down_distance']}"
                        cv2.putText(vis_image, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        text_y += 30
                    if 'score_away' in sit and 'score_home' in sit:
                        text = f"Score: {sit.get('score_away', '?')}-{sit.get('score_home', '?')}"
                        cv2.putText(vis_image, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        text_y += 30
                    if 'clock' in sit and sit['clock']:
                        text = f"Clock: {sit['clock']}"
                        cv2.putText(vis_image, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        text_y += 30
                
                if 'strategic_context' in results:
                    context = results['strategic_context']
                    for i, ctx in enumerate(context.get('special_situations', [])[:3]):
                        text = f"Context: {ctx}"
                        cv2.putText(vis_image, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        text_y += 25
                
                # Draw triangles if detected
                if 'triangle_detections' in results:
                    for triangle in results['triangle_detections']:
                        if triangle.get('confidence', 0) > 0.3:
                            center = (int(triangle.get('center_x', 0)), int(triangle.get('center_y', 0)))
                            cv2.circle(vis_image, center, 8, (255, 0, 255), 3)
                            label = f"{triangle.get('type', 'unknown')}: {triangle.get('confidence', 0):.2f}"
                            cv2.putText(vis_image, label, (center[0]-50, center[1]-15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            confidence = results.get('overall_confidence', 0) if results else 0
            
            self.add_detection_info_overlay(vis_image, {
                'system': 'Production Game Analyzer',
                'overall_confidence': f"{confidence:.2f}",
                'processing_time': f"{processing_time:.3f}s"
            })
            
            # Save result
            output_path = self.dirs['production'] / f"{image_name}_production.png"
            cv2.imwrite(str(output_path), vis_image)
            
            return results, processing_time
            
        except Exception as e:
            print(f"   ❌ Production analysis failed: {e}")
            self.results_summary['errors'].append(f"Production - {image_name}: {e}")
            return None, None
    
    def test_situation_analysis(self, image, image_name):
        """Test game situation analysis."""
        if 'situation' not in self.detectors:
            return None, None
            
        try:
            start_time = time.time()
            results = self.detectors['situation'].analyze_frame(image)
            processing_time = time.time() - start_time
            
            # Create visualization
            vis_image = image.copy()
            
            # Draw detected triangles from situation analysis
            if results and 'triangles' in results:
                for triangle in results['triangles']:
                    if triangle['confidence'] > 0.3:
                        # Draw triangle detection
                        center = (int(triangle['center_x']), int(triangle['center_y']))
                        cv2.circle(vis_image, center, 10, (0, 255, 255), 2)
                        cv2.putText(vis_image, f"{triangle['type']}: {triangle['confidence']:.2f}", 
                                  (center[0]-50, center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Add detected text information
            if results and 'ocr_results' in results:
                y_offset = 50
                for text_detection in results['ocr_results'][:5]:  # Top 5 text detections
                    text = f"OCR: {text_detection.get('text', 'N/A')}"
                    cv2.putText(vis_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
            
            self.add_detection_info_overlay(vis_image, {
                'system': 'Game Situation Analysis',
                'triangles_found': len(results.get('triangles', [])) if results else 0,
                'ocr_detections': len(results.get('ocr_results', [])) if results else 0,
                'processing_time': f"{processing_time:.3f}s"
            })
            
            # Save result
            output_path = self.dirs['situation'] / f"{image_name}_situation.png"
            cv2.imwrite(str(output_path), vis_image)
            
            return results, processing_time
            
        except Exception as e:
            print(f"   ❌ Situation analysis failed: {e}")
            self.results_summary['errors'].append(f"Situation - {image_name}: {e}")
            return None, None
    
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
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(vis_image, f"{class_name}: {conf:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            detection_count += 1
            
            self.add_detection_info_overlay(vis_image, {
                'system': 'YOLO HUD Detection',
                'detections_found': detection_count,
                'processing_time': f"{processing_time:.3f}s"
            })
            
            # Save result
            output_path = self.dirs['yolo'] / f"{image_name}_yolo.png"
            cv2.imwrite(str(output_path), vis_image)
            
            return results, processing_time
            
        except Exception as e:
            print(f"   ❌ YOLO detection failed: {e}")
            self.results_summary['errors'].append(f"YOLO - {image_name}: {e}")
            return None, None
    
    def create_combined_overlay(self, image, image_name, all_results):
        """Create a combined visualization showing all detection results."""
        vis_image = image.copy()
        
        # Add title
        cv2.putText(vis_image, f"Combined Detection Results: {image_name}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add summary of all detections
        y_offset = 70
        total_detections = 0
        
        for system, results in all_results.items():
            if results and results[0]:
                if system == 'situation':
                    count = len(results[0].get('triangles', []))
                    cv2.putText(vis_image, f"Situation Triangles: {count}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    total_detections += count
                elif system == 'yolo':
                    # Count YOLO detections
                    count = 0
                    if results[0] and len(results[0]) > 0:
                        result = results[0][0]
                        if result.boxes is not None:
                            count = len(result.boxes)
                    cv2.putText(vis_image, f"YOLO HUD Regions: {count}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    total_detections += count
                elif system == 'production':
                    conf = results[0].get('overall_confidence', 0) if results[0] else 0
                    cv2.putText(vis_image, f"Production Confidence: {conf:.2f}", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
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
            ('production', self.test_production_analysis),
            ('situation', self.test_situation_analysis), 
            ('yolo', self.test_yolo_detection)
        ]
        
        for system_name, test_func in systems:
            if system_name in self.detectors:
                print(f"     🧪 Testing {system_name}...")
                results, proc_time = test_func(image, image_name)
                all_results[system_name] = (results, proc_time)
                if proc_time:
                    total_time += proc_time
        
        # Create combined visualization
        self.create_combined_overlay(image, image_name, all_results)
        
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
        print("🚀 Starting Comprehensive Detection System Testing")
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
        print("🎉 COMPREHENSIVE TESTING COMPLETE!")
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
    print("🎯 Comprehensive Detection System Testing")
    print("Testing all available detection systems on unannotated images")
    print("Results will be saved with visual overlays for easy review\n")
    
    # Set random seed for reproducible image selection
    random.seed(42)
    
    # Create and run tester
    tester = ComprehensiveDetectionTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main() 