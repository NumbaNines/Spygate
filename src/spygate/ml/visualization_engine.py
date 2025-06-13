"""
Visualization engine for SpygateAI detections.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

from src.spygate.ml.game_state import GameState

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    show_confidence: bool = True
    show_bounding_boxes: bool = True
    show_labels: bool = True
    show_triangles: bool = True
    show_ocr: bool = True
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    
    # Colors (BGR format)
    colors: Dict[str, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "hud": (0, 255, 0),  # Green
                "possession_triangle_area": (255, 0, 0),  # Blue
                "territory_triangle_area": (0, 0, 255),  # Red
                "preplay_indicator": (255, 255, 0),  # Cyan
                "play_call_screen": (0, 255, 255),  # Yellow
                "ocr_text": (255, 255, 255),  # White
                "default": (128, 128, 128)  # Gray
            }

class DetectionVisualizer:
    """Visualizes detections from YOLOv8 and OCR."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with config."""
        self.config = config or VisualizationConfig()
        
    def draw_detection(self, image: np.ndarray, bbox: List[int], 
                      class_name: str, confidence: float) -> np.ndarray:
        """Draw a single detection on the image."""
        x, y, w, h = bbox
        color = self.config.colors.get(class_name, self.config.colors["default"])
        
        # Draw bounding box
        if self.config.show_bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, self.config.box_thickness)
            
        # Draw label with confidence
        if self.config.show_labels:
            label = class_name
            if self.config.show_confidence:
                label += f" {confidence:.2f}"
                
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label, font, 
                                                 self.config.font_scale, 
                                                 self.config.font_thickness)
            
            # Draw label background
            cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y), 
                         color, -1)
            
            # Draw text
            cv2.putText(image, label, (x, y - 4), font, 
                       self.config.font_scale, (0, 0, 0), 
                       self.config.font_thickness)
            
        return image
        
    def draw_ocr_result(self, image: np.ndarray, text: str, 
                       bbox: List[int], confidence: float) -> np.ndarray:
        """Draw OCR result on the image."""
        if not self.config.show_ocr:
            return image
            
        x, y, w, h = bbox
        color = self.config.colors["ocr_text"]
        
        # Draw bounding box
        if self.config.show_bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            
        # Draw text with confidence
        label = f"{text} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (x, y - 4), font, 
                   self.config.font_scale * 0.8, color, 1)
                   
        return image
        
    def draw_triangle_detection(self, image: np.ndarray, bbox: List[int], 
                              direction: str, confidence: float) -> np.ndarray:
        """Draw triangle detection with direction indicator."""
        if not self.config.show_triangles:
            return image
            
        x, y, w, h = bbox
        color = self.config.colors["possession_triangle_area"]
        
        # Draw triangle outline
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        
        # Draw direction indicator
        if direction == "right":
            cv2.arrowedLine(image, (x + w//4, y + h//2), 
                          (x + 3*w//4, y + h//2), color, 2)
        elif direction == "left":
            cv2.arrowedLine(image, (x + 3*w//4, y + h//2), 
                          (x + w//4, y + h//2), color, 2)
        elif direction == "up":
            cv2.arrowedLine(image, (x + w//2, y + 3*h//4), 
                          (x + w//2, y + h//4), color, 2)
        elif direction == "down":
            cv2.arrowedLine(image, (x + w//2, y + h//4), 
                          (x + w//2, y + 3*h//4), color, 2)
                          
        return image
        
    def create_visualization(self, image: np.ndarray, 
                           detections: List[Dict], 
                           ocr_results: Optional[List[Dict]] = None,
                           triangle_detections: Optional[List[Dict]] = None) -> np.ndarray:
        """Create full visualization with all detections."""
        # Make a copy to avoid modifying original
        vis_image = image.copy()
        
        # Draw YOLOv8 detections
        for det in detections:
            vis_image = self.draw_detection(
                vis_image,
                det["bbox"],
                det["class_name"],
                det["confidence"]
            )
            
        # Draw OCR results
        if ocr_results and self.config.show_ocr:
            for ocr in ocr_results:
                vis_image = self.draw_ocr_result(
                    vis_image,
                    ocr["text"],
                    ocr["bbox"],
                    ocr["confidence"]
                )
                
        # Draw triangle detections
        if triangle_detections and self.config.show_triangles:
            for tri in triangle_detections:
                vis_image = self.draw_triangle_detection(
                    vis_image,
                    tri["bbox"],
                    tri["direction"],
                    tri["confidence"]
                )
                
        return vis_image
    
    def create_detection_overlay(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Create overlay showing all detections with confidence scores."""
        overlay = frame.copy()
        
        if self.config.show_bounding_boxes:
            for class_name, boxes in detections.items():
                color = self.config.colors.get(class_name, self.config.colors["default"])
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = box[4] if len(box) > 4 else 1.0
                    
                    # Draw box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Show confidence if enabled
                    if self.config.show_confidence:
                        conf_text = f"{class_name}: {conf:.2f}"
                        cv2.putText(overlay, conf_text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return overlay
    
    def visualize_triangle_geometry(self, frame: np.ndarray, 
                                  triangle_points: List[np.ndarray],
                                  validation_results: List[Tuple[bool, float]]) -> np.ndarray:
        """Show triangle detection geometry and validation."""
        overlay = frame.copy()
        
        if self.config.show_triangles:
            for points, (is_valid, confidence) in zip(triangle_points, validation_results):
                color = (0, 255, 0) if is_valid else (0, 0, 255)
                
                # Draw triangle outline
                cv2.polylines(overlay, [points], True, color, 2)
                
                # Draw geometric validation overlay
                if is_valid:
                    # Show angles and aspect ratio
                    centroid = np.mean(points, axis=0).astype(int)
                    cv2.putText(overlay, f"Conf: {confidence:.2f}", 
                              tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 2)
                    
                    # Draw validation metrics
                    self._draw_validation_metrics(overlay, points, confidence)
        
        return overlay
    
    def visualize_ocr_regions(self, frame: np.ndarray, 
                            ocr_results: Dict[str, Dict]) -> np.ndarray:
        """Show OCR detection regions and results."""
        overlay = frame.copy()
        
        if self.config.show_ocr:
            for region_name, result in ocr_results.items():
                bbox = result.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = bbox
                    color = self.config.colors['ocr_text']
                    
                    # Draw OCR region
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Show detected text and confidence
                    text = result.get('text', '')
                    conf = result.get('confidence', 0)
                    cv2.putText(overlay, f"{text} ({conf:.2f})",
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 2)
        
        return overlay
    
    def visualize_hud_analysis(self, frame: np.ndarray, 
                             game_state: 'GameState') -> np.ndarray:
        """Show HUD analysis results and game state extraction."""
        overlay = frame.copy()
        
        # Create info panel
        info_panel = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
        
        # Add game state information
        lines = [
            f"Down: {game_state.down}",
            f"Distance: {game_state.distance}",
            f"Yard Line: {game_state.yard_line}",
            f"Possession: {game_state.possession_team}",
            f"Territory: {game_state.territory}",
            f"Confidence: {game_state.confidence:.2f}"
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(info_panel, line, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine frame and info panel
        result = np.vstack([overlay, info_panel])
    
        return result
    
    def _draw_validation_metrics(self, overlay: np.ndarray, 
                               points: np.ndarray, 
                               confidence: float) -> None:
        """Draw geometric validation metrics for triangle detection."""
        # Calculate and show angles
        angles = self._calculate_angles(points)
        for i, angle in enumerate(angles):
            pt = points[i][0]
            cv2.putText(overlay, f"{angle:.1f}°", 
                       tuple(pt), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1)
    
    def _calculate_angles(self, points: np.ndarray) -> List[float]:
        """Calculate angles of triangle."""
        angles = []
        for i in range(3):
            pt1 = points[i][0]
            pt2 = points[(i+1)%3][0]
            pt3 = points[(i+2)%3][0]
            
            v1 = pt2 - pt1
            v2 = pt3 - pt1
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)
            
        return angles
    
    def create_visualization_pipeline(self, frame: np.ndarray,
                                   detections: Dict,
                                   triangle_data: List[Tuple[np.ndarray, Tuple[bool, float]]],
                                   ocr_results: Dict[str, Dict],
                                   game_state: 'GameState') -> List[np.ndarray]:
        """
        Create complete visualization pipeline showing each detection layer.
        
        Returns:
            List of frames showing each layer of detection:
            1. Raw frame
            2. Detection overlay
            3. Triangle geometry analysis
            4. OCR region detection
            5. Final HUD analysis
        """
        visualizations = []
        
        # 1. Raw frame
        visualizations.append(frame.copy())
        
        # 2. Detection overlay
        detection_viz = self.create_detection_overlay(frame, detections)
        visualizations.append(detection_viz)
        
        # 3. Triangle geometry
        triangle_points = [t[0] for t in triangle_data]
        validation_results = [t[1] for t in triangle_data]
        triangle_viz = self.visualize_triangle_geometry(frame, triangle_points, validation_results)
        visualizations.append(triangle_viz)
        
        # 4. OCR regions
        ocr_viz = self.visualize_ocr_regions(frame, ocr_results)
        visualizations.append(ocr_viz)
        
        # 5. HUD analysis
        hud_viz = self.visualize_hud_analysis(frame, game_state)
        visualizations.append(hud_viz)
        
        return visualizations
    
    def save_visualizations(self, visualizations: List[np.ndarray], 
                          base_path: Path) -> None:
        """Save visualization frames to disk."""
        if not base_path.exists():
            base_path.mkdir(parents=True)
            
        for i, viz in enumerate(visualizations):
            layer_name = [
                "raw_frame",
                "detections",
                "triangle_geometry",
                "ocr_regions",
                "hud_analysis"
            ][i]
            
            output_path = base_path / f"layer_{i}_{layer_name}.png"
            cv2.imwrite(str(output_path), viz)
            logger.info(f"Saved visualization layer {i} to {output_path}")
    
    def display_visualizations(self, visualizations: List[np.ndarray]) -> None:
        """Display visualization frames using matplotlib."""
        n_viz = len(visualizations)
        fig, axes = plt.subplots(1, n_viz, figsize=(4*n_viz, 4))
        
        titles = [
            "Raw Frame",
            "Detections",
            "Triangle Geometry",
            "OCR Regions",
            "HUD Analysis"
        ]
        
        for i, (viz, title) in enumerate(zip(visualizations, titles)):
            if n_viz > 1:
                ax = axes[i]
            else:
                ax = axes
            
            ax.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show() 