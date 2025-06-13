#!/usr/bin/env python3
"""
Comprehensive SpygateAI Detection Analysis Tool
Shows all detection components: YOLO + OCR + Game State + Hardware + Triangles
"""

import cv2
import numpy as np
import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from ultralytics import YOLO
import torch

# Import SpygateAI components
from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.ml.enhanced_ocr import EnhancedOCR
from src.spygate.core.hardware import HardwareDetector, HardwareTier

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveSpygateAnalyzer:
    """Complete SpygateAI detection pipeline with full component visualization."""
    
    def __init__(self, output_dir: str = "comprehensive_analysis_results"):
        """Initialize all SpygateAI detection components."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize hardware detection first
        logger.info("🔧 Initializing Hardware Detection...")
        self.hardware_detector = HardwareDetector()
        self.hardware_tier = self.hardware_detector.detect_tier()
        logger.info(f"✅ Hardware Tier: {self.hardware_tier.name}")
        
        # Initialize custom HUD YOLO model
        logger.info("🎯 Loading Custom HUD YOLO Model...")
        self.yolo_model = YOLO("hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt")
        self.yolo_model.conf = 0.25
        logger.info("✅ Custom HUD YOLO Model Loaded")
        
        # Initialize OCR engines
        logger.info("📝 Initializing OCR Engines...")
        self.ocr_engine = EnhancedOCR(hardware=self.hardware_tier)
        logger.info("✅ OCR Engines Ready (EasyOCR + Tesseract)")
        
        # Initialize full game analyzer
        logger.info("🧠 Initializing Enhanced Game Analyzer...")
        self.game_analyzer = EnhancedGameAnalyzer(
            hardware=self.hardware_detector
        )
        logger.info("✅ Enhanced Game Analyzer Ready")
        
        # Custom HUD classes
        self.hud_classes = [
            "hud",
            "possession_triangle_area", 
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen"
        ]
        
        # Colors for different detection types
        self.colors = {
            # YOLO detections
            "hud": (0, 255, 0),  # Green
            "possession_triangle_area": (255, 0, 0),  # Blue
            "territory_triangle_area": (0, 0, 255),  # Red
            "preplay_indicator": (255, 255, 0),  # Cyan
            "play_call_screen": (255, 0, 255),  # Magenta
            # OCR detections
            "ocr_text": (0, 255, 255),  # Yellow
            "ocr_region": (255, 255, 255),  # White
            # Game state
            "game_state": (128, 255, 128),  # Light green
        }
        
        logger.info("🚀 Comprehensive SpygateAI Analyzer Ready!")
    
    def analyze_image_comprehensive(self, image_path: str) -> Dict[str, Any]:
        """Run complete SpygateAI analysis on a single image."""
        logger.info(f"🔍 Analyzing: {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        analysis_start = time.time()
        results = {
            "image_path": image_path,
            "image_shape": image.shape,
            "analysis_time": 0,
            "hardware_tier": self.hardware_tier.name,
            "detections": {}
        }
        
        # 1. YOLO HUD Detection
        logger.info("  🎯 Running YOLO HUD Detection...")
        yolo_start = time.time()
        yolo_results = self.yolo_model(image, verbose=False)
        yolo_detections = self.process_yolo_results(yolo_results)
        results["detections"]["yolo"] = {
            "detection_time": time.time() - yolo_start,
            "detections": yolo_detections,
            "count": len(yolo_detections)
        }
        logger.info(f"    ✅ Found {len(yolo_detections)} HUD elements")
        
        # 2. OCR Text Detection
        logger.info("  📝 Running OCR Text Detection...")
        ocr_start = time.time()
        ocr_results = self.run_ocr_analysis(image)
        results["detections"]["ocr"] = {
            "detection_time": time.time() - ocr_start,
            "detections": ocr_results,
            "count": len(ocr_results)
        }
        logger.info(f"    ✅ Found {len(ocr_results)} text regions")
        
        # 3. Game State Analysis
        logger.info("  🧠 Running Game State Analysis...")
        game_start = time.time()
        game_state = self.game_analyzer.analyze_frame(image)
        game_analysis = self.process_game_state(game_state)
        results["detections"]["game_state"] = {
            "analysis_time": time.time() - game_start,
            "state": game_analysis
        }
        logger.info(f"    ✅ Game state analyzed")
        
        # 4. Triangle Detection (specialized)
        logger.info("  🔺 Running Triangle Detection...")
        triangle_start = time.time()
        triangle_results = self.detect_triangles(image, yolo_detections)
        results["detections"]["triangles"] = {
            "detection_time": time.time() - triangle_start,
            "detections": triangle_results,
            "count": len(triangle_results)
        }
        logger.info(f"    ✅ Found {len(triangle_results)} triangles")
        
        results["analysis_time"] = time.time() - analysis_start
        logger.info(f"  ⏱️ Total analysis time: {results['analysis_time']:.2f}s")
        
        return results
    
    def process_yolo_results(self, yolo_results) -> List[Dict[str, Any]]:
        """Process YOLO detection results."""
        detections = []
        
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    cls = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    if cls < len(self.hud_classes):
                        detection = {
                            "class": self.hud_classes[cls],
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "area": (x2 - x1) * (y2 - y1),
                            "center": [(x1 + x2) // 2, (y1 + y2) // 2]
                        }
                        detections.append(detection)
        
        return detections
    
    def run_ocr_analysis(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run comprehensive OCR analysis."""
        try:
            # Use SpygateAI's enhanced OCR
            ocr_results = self.ocr_engine.extract_text(image)
            
            processed_results = []
            if ocr_results:
                for result in ocr_results:
                    if isinstance(result, dict):
                        processed_results.append({
                            "text": result.get("text", ""),
                            "confidence": result.get("confidence", 0.0),
                            "bbox": result.get("bbox", [0, 0, 0, 0]),
                            "method": result.get("method", "unknown")
                        })
                    elif isinstance(result, tuple) and len(result) >= 2:
                        # Handle tuple format (bbox, text, confidence)
                        bbox, text = result[0], result[1]
                        conf = result[2] if len(result) > 2 else 0.0
                        
                        processed_results.append({
                            "text": text,
                            "confidence": conf,
                            "bbox": bbox if isinstance(bbox, list) else [0, 0, 0, 0],
                            "method": "enhanced_ocr"
                        })
            
            return processed_results
            
        except Exception as e:
            logger.warning(f"OCR analysis failed: {e}")
            return []
    
    def process_game_state(self, game_state) -> Dict[str, Any]:
        """Process game state analysis results."""
        state_dict = {
            "down": getattr(game_state, 'down', None),
            "distance": getattr(game_state, 'distance', None),
            "possession_team": getattr(game_state, 'possession_team', None),
            "field_position": getattr(game_state, 'field_position', None),
            "quarter": getattr(game_state, 'quarter', None),
            "time_remaining": getattr(game_state, 'time_remaining', None),
            "score_away": getattr(game_state, 'score_away', None),
            "score_home": getattr(game_state, 'score_home', None),
            "confidence": getattr(game_state, 'confidence', 0.0),
            "is_play_active": getattr(game_state, 'is_play_active', False),
            "game_situation": getattr(game_state, 'game_situation', None)
        }
        
        # Filter out None values
        return {k: v for k, v in state_dict.items() if v is not None}
    
    def detect_triangles(self, image: np.ndarray, yolo_detections: List[Dict]) -> List[Dict[str, Any]]:
        """Specialized triangle detection within triangle areas."""
        triangle_detections = []
        
        # Find triangle areas from YOLO
        triangle_areas = [d for d in yolo_detections 
                         if 'triangle' in d['class']]
        
        for area in triangle_areas:
            bbox = area['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract region
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Convert to grayscale for triangle detection
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Look for triangle-like shapes
            triangles = self.find_triangle_shapes(gray_roi)
            
            for triangle in triangles:
                # Adjust coordinates back to full image
                adj_triangle = []
                for point in triangle:
                    adj_point = [point[0] + x1, point[1] + y1]
                    adj_triangle.append(adj_point)
                
                triangle_detections.append({
                    "type": area['class'],
                    "area_bbox": bbox,
                    "triangle_points": adj_triangle,
                    "confidence": area['confidence']
                })
        
        return triangle_detections
    
    def find_triangle_shapes(self, gray_image: np.ndarray) -> List[List[List[int]]]:
        """Find triangle shapes using contour detection."""
        triangles = []
        
        try:
            # Apply threshold
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's triangle-like (3-4 vertices)
                if len(approx) == 3 or len(approx) == 4:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Minimum area threshold
                        triangle_points = [[int(point[0][0]), int(point[0][1])] for point in approx]
                        triangles.append(triangle_points)
        
        except Exception as e:
            logger.warning(f"Triangle detection failed: {e}")
        
        return triangles
    
    def create_comprehensive_visualization(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Create comprehensive visualization with all detection types."""
        vis_image = image.copy()
        
        # 1. Draw YOLO detections
        for detection in results["detections"]["yolo"]["detections"]:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            class_name = detection["class"]
            conf = detection["confidence"]
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"YOLO: {class_name} ({conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 2. Draw OCR detections
        for detection in results["detections"]["ocr"]["detections"]:
            if "bbox" in detection and detection["bbox"]:
                bbox = detection["bbox"]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    text = detection.get("text", "")
                    conf = detection.get("confidence", 0)
                    
                    # Draw OCR bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), self.colors["ocr_text"], 1)
                    
                    # Draw OCR text
                    if text.strip():
                        label = f"OCR: {text[:20]}... ({conf:.2f})"
                        cv2.putText(vis_image, label, (x1, y2 + 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["ocr_text"], 1)
        
        # 3. Draw triangles
        for triangle in results["detections"]["triangles"]["detections"]:
            points = triangle["triangle_points"]
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.polylines(vis_image, [pts], True, (0, 255, 255), 2)
                
                # Label triangle
                center_x = sum(p[0] for p in points) // len(points)
                center_y = sum(p[1] for p in points) // len(points)
                cv2.putText(vis_image, f"△ {triangle['type']}", (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 4. Add game state overlay
        game_state = results["detections"]["game_state"]["state"]
        y_offset = 30
        
        for key, value in game_state.items():
            if value is not None:
                text = f"{key}: {value}"
                cv2.putText(vis_image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["game_state"], 2)
                y_offset += 25
        
        # 5. Add performance info
        analysis_time = results.get("analysis_time", 0)
        hardware_tier = results.get("hardware_tier", "Unknown")
        
        perf_text = f"Analysis: {analysis_time:.2f}s | Hardware: {hardware_tier}"
        cv2.putText(vis_image, perf_text, (10, vis_image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def analyze_directory(self, input_dir: str):
        """Analyze all images in directory with comprehensive SpygateAI detection."""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_path}")
        
        # Find all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(input_path.glob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"No image files found in {input_path}")
            return
        
        logger.info(f"🎯 Starting comprehensive analysis of {len(image_files)} images")
        
        total_stats = {
            "total_images": len(image_files),
            "total_yolo_detections": 0,
            "total_ocr_detections": 0,
            "total_triangles": 0,
            "total_analysis_time": 0,
            "hardware_tier": self.hardware_tier.name
        }
        
        for i, image_file in enumerate(image_files, 1):
            try:
                logger.info(f"📸 Processing {i}/{len(image_files)}: {image_file.name}")
                
                # Run comprehensive analysis
                results = self.analyze_image_comprehensive(str(image_file))
                
                # Create visualization
                image = cv2.imread(str(image_file))
                vis_image = self.create_comprehensive_visualization(image, results)
                
                # Save results
                output_path = self.output_dir / f"{image_file.stem}_comprehensive_analysis.jpg"
                cv2.imwrite(str(output_path), vis_image)
                
                # Update stats
                total_stats["total_yolo_detections"] += results["detections"]["yolo"]["count"]
                total_stats["total_ocr_detections"] += results["detections"]["ocr"]["count"]
                total_stats["total_triangles"] += results["detections"]["triangles"]["count"]
                total_stats["total_analysis_time"] += results["analysis_time"]
                
                logger.info(f"✅ Saved: {output_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_file}: {e}")
        
        # Print final statistics
        avg_time = total_stats["total_analysis_time"] / total_stats["total_images"]
        
        logger.info("📊 COMPREHENSIVE ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Images Processed: {total_stats['total_images']}")
        logger.info(f"Total YOLO Detections: {total_stats['total_yolo_detections']}")
        logger.info(f"Total OCR Detections: {total_stats['total_ocr_detections']}")
        logger.info(f"Total Triangles Found: {total_stats['total_triangles']}")
        logger.info(f"Average Analysis Time: {avg_time:.2f}s per image")
        logger.info(f"Hardware Tier: {total_stats['hardware_tier']}")
        logger.info(f"Total Processing Time: {total_stats['total_analysis_time']:.2f}s")
        logger.info("=" * 60)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive SpygateAI Detection Analysis")
    parser.add_argument("input_dir", help="Directory containing images to analyze")
    parser.add_argument("--output-dir", default="comprehensive_analysis_results", 
                       help="Directory to save analysis results")
    args = parser.parse_args()
    
    try:
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveSpygateAnalyzer(args.output_dir)
        
        # Run analysis
        analyzer.analyze_directory(args.input_dir)
        
        logger.info(f"🎉 Analysis complete! Check results in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 