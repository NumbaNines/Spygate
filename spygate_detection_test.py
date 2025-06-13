#!/usr/bin/env python3
"""
Comprehensive SpygateAI Detection Test
Shows all detection components working: YOLO + OCR + Game State + Hardware
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import time
from ultralytics import YOLO
import json
from datetime import datetime
import pytesseract

# Set Tesseract path explicitly
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# SpygateAI imports
from src.spygate.ml.enhanced_ocr import EnhancedOCR
from src.spygate.core.hardware import HardwareDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colors for visualization
COLORS = {
    'hud': (0, 255, 0),  # Green
    'possession_triangle_area': (255, 0, 0),  # Blue
    'territory_triangle_area': (0, 0, 255),  # Red
    'play_call_screen': (255, 255, 0),  # Cyan
    'preplay_indicator': (255, 0, 255)  # Magenta
}

def draw_detections(image, detections, ocr_results=None):
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    
    # Draw YOLO detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class']
        color = COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw OCR results if available
    if ocr_results:
        for result in ocr_results:
            x1, y1, x2, y2 = result['bbox']
            text_info = result['text']
            
            # Draw OCR region
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # Draw extracted text
            y_offset = y1 - 5
            for key, value in text_info.items():
                if value is not None:
                    text = f"{key}: {value}"
                    cv2.putText(img, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset -= 15
    
    return img

def save_results(results_dir: Path, image_name: str, results: dict):
    """Save detection results to JSON file."""
    results_file = results_dir / f"{image_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

def comprehensive_detection_test():
    """Test all SpygateAI detection components."""
    
    print("🚀 COMPREHENSIVE SPYGATE DETECTION TEST")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("detection_test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create visualizations directory
    vis_dir = results_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Track overall statistics
    stats = {
        'total_images': 0,
        'successful_detections': 0,
        'total_hud_elements': 0,
        'total_triangles': 0,
        'total_text_regions': 0,
        'total_time': 0,
        'start_time': datetime.now().isoformat()
    }
    
    # 1. Hardware Detection Test
    print("\n1. 🔧 HARDWARE DETECTION:")
    try:
        hw_detector = HardwareDetector()
        tier = hw_detector.detect_tier()
        print(f"   ✅ Hardware Tier: {tier.name}")
        print(f"   ✅ CPU Cores: {hw_detector.cpu_count}")
        print(f"   ✅ RAM: {hw_detector.total_ram:.1f}GB")
        print(f"   ✅ CUDA Available: {hw_detector.has_cuda}")
        if hw_detector.has_cuda:
            print(f"   ✅ GPU: {hw_detector.gpu_name}")
            print(f"   ✅ GPU Memory: {hw_detector.gpu_memory:.1f}GB")
            
        stats['hardware'] = {
            'tier': tier.name,
            'cpu_cores': hw_detector.cpu_count,
            'ram_gb': round(hw_detector.total_ram, 1),
            'has_cuda': hw_detector.has_cuda,
            'gpu_name': hw_detector.gpu_name if hw_detector.has_cuda else None,
            'gpu_memory_gb': round(hw_detector.gpu_memory, 1) if hw_detector.has_cuda else None
        }
    except Exception as e:
        print(f"   ❌ Hardware Detection Error: {e}")
        return
    
    # 2. YOLO Custom HUD Model Test
    print("\n2. 🎯 CUSTOM HUD YOLO MODEL:")
    try:
        model_path = "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
        model = YOLO(model_path)
        model.conf = 0.25
        print(f"   ✅ Custom HUD Model Loaded: {model_path}")
        print(f"   ✅ Model Classes: {len(model.names)}")
        print(f"   ✅ Class Names: {list(model.names.values())}")
        print(f"   ✅ Confidence Threshold: {model.conf}")
        
        stats['model'] = {
            'path': model_path,
            'classes': len(model.names),
            'class_names': list(model.names.values()),
            'confidence_threshold': model.conf
        }
    except Exception as e:
        print(f"   ❌ YOLO Model Error: {e}")
        return
    
    # 3. Enhanced OCR Test  
    print("\n3. 📝 ENHANCED OCR ENGINE:")
    try:
        ocr_engine = EnhancedOCR(hardware=tier)
        print(f"   ✅ Enhanced OCR Initialized")
        print(f"   ✅ Hardware Tier: {tier.name}")
        print(f"   ✅ Has EasyOCR Reader: {hasattr(ocr_engine, 'reader') and ocr_engine.reader is not None}")
        print(f"   ✅ Validation Config: Min Confidence {ocr_engine.validation.min_confidence}")
        
        stats['ocr'] = {
            'hardware_tier': tier.name,
            'has_easyocr': hasattr(ocr_engine, 'reader') and ocr_engine.reader is not None,
            'min_confidence': ocr_engine.validation.min_confidence
        }
    except Exception as e:
        print(f"   ❌ OCR Engine Error: {e}")
        return
    
    # 4. Image Analysis Test
    print("\n4. 📷 IMAGE ANALYSIS TEST:")
    image_dir = Path("Madden AYOUTUB")
    if not image_dir.exists():
        print(f"   ❌ Image directory not found: {image_dir}")
        return
    
    images = list(image_dir.glob("*.png"))
    if not images:
        print(f"   ❌ No PNG images found in {image_dir}")
        return
    
    print(f"\nProcessing {len(images)} images from {image_dir}...")
    stats['total_images'] = len(images)
    
    for img_num, image_path in enumerate(images, 1):
        print(f"\n📸 Processing Image {img_num}/{len(images)}: {image_path.name}")
        image_results = {
            'filename': image_path.name,
            'detections': [],
            'ocr_results': [],
            'timing': {}
        }
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"   ❌ Could not load image")
                continue
            
            print(f"   ✅ Image Shape: {image.shape}")
            image_results['shape'] = list(image.shape)
            
            # YOLO Detection Analysis
            print("\n   🎯 YOLO HUD DETECTION:")
            start_time = time.time()
            yolo_results = model(image, verbose=False)
            yolo_time = time.time() - start_time
            image_results['timing']['yolo'] = yolo_time
            
            yolo_detections = 0
            detection_details = []
            
            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    yolo_detections = len(boxes)
                    
                    print(f"      ✅ YOLO Detections: {yolo_detections} HUD elements")
                    print(f"      ⏱️  YOLO Time: {yolo_time:.3f}s")
                    
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i].cpu().numpy())
                        conf = float(boxes.conf[i].cpu().numpy())
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        
                        if cls < len(model.names):
                            class_name = model.names[cls]
                            detection = {
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            }
                            detection_details.append(detection)
                            print(f"      - {class_name}: {conf:.3f} at [{x1},{y1},{x2},{y2}]")
                            
                    image_results['detections'] = detection_details
            else:
                print(f"      ⚠️  YOLO: No detections found")
            
            # OCR Detection Analysis
            print("\n   📝 OCR TEXT DETECTION:")
            start_time = time.time()
            try:
                # Process each detected HUD region with OCR
                ocr_results = []
                for detection in detection_details:
                    if detection['class'] == 'hud':
                        # Extract the region from the image
                        x1, y1, x2, y2 = detection['bbox']
                        region = image[y1:y2, x1:x2]
                        
                        # Process the region
                        region_results = ocr_engine.process_region(region, region_type='hud')
                        if region_results:
                            result = {
                                'bbox': detection['bbox'],
                                'text': region_results,
                                'confidence': detection['confidence'],
                                'method': 'enhanced_ocr'
                            }
                            ocr_results.append(result)
                
                ocr_time = time.time() - start_time
                image_results['timing']['ocr'] = ocr_time
                image_results['ocr_results'] = ocr_results
                
                if ocr_results:
                    print(f"      ✅ OCR Detections: {len(ocr_results)} text regions")
                    print(f"      ⏱️  OCR Time: {ocr_time:.3f}s")
                    
                    for i, result in enumerate(ocr_results):
                        bbox = result['bbox']
                        text_info = result['text']
                        
                        print(f"      - Region {i+1} at {bbox}:")
                        for key, value in text_info.items():
                            if value is not None:
                                print(f"         {key}: {value}")
                else:
                    print(f"      ⚠️  OCR: No text detected")
                    
            except Exception as e:
                print(f"      ❌ OCR Detection Error: {e}")
                import traceback
                traceback.print_exc()
            
            # Draw detections on image
            annotated_image = draw_detections(image, detection_details, ocr_results)
            
            # Save annotated image
            output_path = vis_dir / f"{image_path.stem}_annotated.png"
            cv2.imwrite(str(output_path), annotated_image)
            
            # Performance Summary
            total_time = yolo_time + ocr_time
            image_results['timing']['total'] = total_time
            
            print("\n   📊 PERFORMANCE SUMMARY:")
            print(f"      Total Analysis Time: {total_time:.3f}s")
            print(f"      YOLO Performance: {yolo_detections/yolo_time:.1f} detections/sec" if yolo_time > 0 else "      YOLO Performance: N/A")
            print(f"      OCR Performance: {len(ocr_results)/ocr_time:.1f} regions/sec" if ocr_time > 0 and ocr_results else "      OCR Performance: N/A")
            
            # Analysis Results Summary
            print("\n   🎯 DETECTION ANALYSIS:")
            hud_elements = [d for d in detection_details if 'hud' in d['class']]
            triangles = [d for d in detection_details if 'triangle' in d['class']]
            indicators = [d for d in detection_details if 'indicator' in d['class'] or 'screen' in d['class']]
            
            print(f"      HUD Elements: {len(hud_elements)}")
            print(f"      Triangle Areas: {len(triangles)}")
            print(f"      Game Indicators: {len(indicators)}")
            
            image_results['summary'] = {
                'hud_elements': len(hud_elements),
                'triangles': len(triangles),
                'indicators': len(indicators)
            }
            
            if triangles:
                print("      Triangle Details:")
                for triangle in triangles:
                    print(f"        - {triangle['class']}: {triangle['confidence']:.3f}")
            
            # Hardware Performance Rating
            fps_estimate = 1.0 / total_time if total_time > 0 else 0
            print(f"\n   ⚡ PERFORMANCE:")
            print(f"      Estimated FPS: {fps_estimate:.2f}")
            
            if fps_estimate >= 2.0:
                rating = "EXCELLENT (Real-time capable)"
            elif fps_estimate >= 1.0:
                rating = "GOOD (Near real-time)"
            elif fps_estimate >= 0.5:
                rating = "FAIR (Batch processing)"
            else:
                rating = "SLOW (Optimization needed)"
            
            print(f"      Performance Rating: {rating}")
            
            image_results['performance'] = {
                'fps': fps_estimate,
                'rating': rating
            }
            
            # Update overall statistics
            stats['successful_detections'] += 1
            stats['total_hud_elements'] += len(hud_elements)
            stats['total_triangles'] += len(triangles)
            stats['total_text_regions'] += len(ocr_results)
            stats['total_time'] += total_time
            
            # Save individual image results
            save_results(results_dir, image_path.stem, image_results)
            
        except Exception as e:
            print(f"   ❌ Image Analysis Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall statistics
    stats['end_time'] = datetime.now().isoformat()
    stats['average_time_per_image'] = stats['total_time'] / stats['total_images'] if stats['total_images'] > 0 else 0
    stats['average_fps'] = stats['total_images'] / stats['total_time'] if stats['total_time'] > 0 else 0
    
    print("\n📊 OVERALL STATISTICS:")
    print(f"Total Images Processed: {stats['total_images']}")
    print(f"Successful Detections: {stats['successful_detections']}")
    print(f"Total HUD Elements: {stats['total_hud_elements']}")
    print(f"Total Triangles: {stats['total_triangles']}")
    print(f"Total Text Regions: {stats['total_text_regions']}")
    print(f"Average Time per Image: {stats['average_time_per_image']:.3f}s")
    print(f"Average FPS: {stats['average_fps']:.2f}")
    
    # Save overall statistics
    save_results(results_dir, "overall_statistics", stats)
    
    print("\n🎉 COMPREHENSIVE DETECTION TEST COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print(f"Visualizations saved to: {vis_dir}")
    print("=" * 60)

if __name__ == "__main__":
    comprehensive_detection_test() 