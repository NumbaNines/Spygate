"""
Comprehensive False Positive Reduction Strategies for 8-Class YOLOv8 Model.
Multiple approaches to improve precision and reduce unwanted detections.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json

class FalsePositiveReducer:
    """
    Comprehensive false positive reduction system for YOLOv8 HUD detection.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.strategies = {
            "confidence_tuning": self.tune_confidence_thresholds,
            "nms_optimization": self.optimize_nms_settings,
            "training_improvements": self.improve_training_config,
            "data_augmentation": self.enhance_data_quality,
            "post_processing": self.add_post_processing_filters,
            "geometric_validation": self.add_geometric_constraints
        }
    
    def tune_confidence_thresholds(self) -> Dict:
        """
        Strategy 1: Optimize confidence thresholds per class.
        Different HUD elements may need different confidence levels.
        """
        
        # Class-specific confidence thresholds based on your results
        optimized_thresholds = {
            'hud': 0.6,                      # Main HUD - higher confidence needed
            'possession_triangle_area': 0.7,  # Critical for game state - be strict
            'territory_triangle_area': 0.7,   # Critical for game state - be strict  
            'preplay_indicator': 0.5,         # Can be subtle - moderate confidence
            'play_call_screen': 0.8,          # Very distinct - high confidence
            'down_distance_area': 0.6,        # Important text - moderate-high confidence
            'game_clock_area': 0.5,           # Can vary in appearance - moderate
            'play_clock_area': 0.5            # Can vary in appearance - moderate
        }
        
        print("🎯 Optimized Confidence Thresholds:")
        for class_name, threshold in optimized_thresholds.items():
            print(f"  {class_name}: {threshold}")
        
        return optimized_thresholds
    
    def optimize_nms_settings(self) -> Dict:
        """
        Strategy 2: Optimize Non-Maximum Suppression to reduce overlapping detections.
        """
        
        nms_config = {
            'iou': 0.4,          # Lower IoU = more aggressive NMS (reduces overlaps)
            'agnostic': False,   # Class-aware NMS (better for distinct HUD elements)
            'max_det': 20,       # Limit total detections per image
            'classes': None,     # Detect all classes
            'retina_masks': False
        }
        
        print("🔧 Optimized NMS Settings:")
        for setting, value in nms_config.items():
            print(f"  {setting}: {value}")
        
        return nms_config
    
    def improve_training_config(self) -> Dict:
        """
        Strategy 3: Enhanced training configuration to reduce false positives.
        """
        
        improved_config = {
            # Loss function adjustments
            'box': 7.5,              # Box regression loss weight
            'cls': 1.0,              # INCREASED: Classification loss weight (reduces FP)
            'dfl': 1.5,              # Distribution focal loss weight
            
            # Confidence and objectness
            'obj': 1.0,              # Objectness loss weight
            'obj_pw': 1.0,           # Objectness positive weight
            'iou_t': 0.20,           # IoU training threshold
            'anchor_t': 4.0,         # Anchor-multiple threshold
            
            # Data augmentation (reduce overfitting)
            'hsv_h': 0.015,          # HSV-Hue augmentation
            'hsv_s': 0.7,            # HSV-Saturation augmentation  
            'hsv_v': 0.4,            # HSV-Value augmentation
            'degrees': 0.0,          # NO rotation (HUD is always horizontal)
            'translate': 0.1,        # Translation augmentation
            'scale': 0.5,            # Scale augmentation
            'shear': 0.0,            # NO shear (HUD elements are rectangular)
            'perspective': 0.0,      # NO perspective (HUD is 2D overlay)
            'flipud': 0.0,           # NO vertical flip (HUD has fixed orientation)
            'fliplr': 0.0,           # NO horizontal flip (possession triangles have meaning)
            'mosaic': 0.0,           # NO mosaic (can create false HUD combinations)
            'mixup': 0.0,            # NO mixup (HUD elements shouldn't blend)
            'copy_paste': 0.0,       # NO copy-paste (HUD elements are contextual)
            
            # Regularization
            'weight_decay': 0.0005,  # L2 regularization
            'dropout': 0.1,          # ADDED: Dropout to reduce overfitting
            'label_smoothing': 0.1,  # ADDED: Label smoothing to reduce overconfidence
            
            # Learning rate schedule
            'lr0': 0.005,            # REDUCED: Lower initial learning rate
            'lrf': 0.01,             # Lower final learning rate
            'momentum': 0.937,
            'warmup_epochs': 5,      # INCREASED: More warmup epochs
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Training strategy
            'epochs': 100,           # INCREASED: More epochs for better convergence
            'patience': 20,          # INCREASED: More patience for early stopping
            'batch': 16,             # REDUCED: Smaller batch for better gradients
            'imgsz': 640,
            'cache': True,
            'device': 0,
            'workers': 8,
            'project': 'hud_region_training/hud_region_training_8class/runs',
            'name': 'hud_8class_fp_reduced',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',    # CHANGED: AdamW often better for precision
            'verbose': True,
            'seed': 42,              # ADDED: Reproducible results
            'deterministic': True,   # ADDED: Deterministic training
            'single_cls': False,
            'rect': False,
            'cos_lr': True,          # ADDED: Cosine learning rate schedule
            'close_mosaic': 15,      # Close mosaic augmentation early
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'save_period': 10,
            'val': True,
            'plots': True,
            'save': True
        }
        
        print("🎓 Enhanced Training Configuration:")
        print(f"  📈 Increased classification loss weight: {improved_config['cls']}")
        print(f"  🎯 Added dropout: {improved_config['dropout']}")
        print(f"  🔄 Added label smoothing: {improved_config['label_smoothing']}")
        print(f"  📚 Disabled problematic augmentations (mosaic, mixup, etc.)")
        print(f"  ⚡ Switched to AdamW optimizer")
        print(f"  🎪 Added cosine learning rate schedule")
        
        return improved_config
    
    def enhance_data_quality(self) -> Dict:
        """
        Strategy 4: Data quality improvements to reduce false positives.
        """
        
        data_strategies = {
            "hard_negative_mining": {
                "description": "Add images with NO HUD elements to teach what NOT to detect",
                "implementation": "Include blank gameplay frames, menu screens, loading screens"
            },
            "negative_examples": {
                "description": "Add similar-looking but incorrect regions",
                "implementation": "Include scoreboard elements, UI buttons, text overlays that aren't HUD"
            },
            "class_balancing": {
                "description": "Ensure balanced representation of all classes",
                "implementation": "Check annotation distribution and add more examples for underrepresented classes"
            },
            "annotation_quality": {
                "description": "Review and improve annotation precision",
                "implementation": "Tighter bounding boxes, consistent labeling, remove ambiguous examples"
            },
            "context_diversity": {
                "description": "Include diverse game contexts",
                "implementation": "Different quarters, scores, field positions, lighting conditions"
            }
        }
        
        print("📊 Data Quality Enhancement Strategies:")
        for strategy, details in data_strategies.items():
            print(f"  🎯 {strategy}: {details['description']}")
        
        return data_strategies
    
    def add_post_processing_filters(self) -> Dict:
        """
        Strategy 5: Post-processing filters to remove false positives.
        """
        
        filters = {
            "size_constraints": {
                "hud": {"min_width": 200, "max_width": 1200, "min_height": 30, "max_height": 150},
                "possession_triangle_area": {"min_width": 50, "max_width": 300, "min_height": 20, "max_height": 100},
                "territory_triangle_area": {"min_width": 30, "max_width": 150, "min_height": 15, "max_height": 80},
                "preplay_indicator": {"min_width": 40, "max_width": 200, "min_height": 15, "max_height": 60},
                "play_call_screen": {"min_width": 100, "max_width": 800, "min_height": 50, "max_height": 400},
                "down_distance_area": {"min_width": 40, "max_width": 200, "min_height": 15, "max_height": 50},
                "game_clock_area": {"min_width": 50, "max_width": 200, "min_height": 15, "max_height": 50},
                "play_clock_area": {"min_width": 30, "max_width": 100, "min_height": 15, "max_height": 50}
            },
            "position_constraints": {
                "hud": {"region": "top_half", "y_max": 0.3},
                "possession_triangle_area": {"region": "top_left", "x_max": 0.6, "y_max": 0.3},
                "territory_triangle_area": {"region": "top_right", "x_min": 0.4, "y_max": 0.3},
                "preplay_indicator": {"region": "bottom_left", "x_max": 0.4, "y_min": 0.7},
                "play_call_screen": {"region": "center", "x_min": 0.2, "x_max": 0.8, "y_min": 0.2, "y_max": 0.8},
                "down_distance_area": {"region": "top_center", "x_min": 0.3, "x_max": 0.7, "y_max": 0.3},
                "game_clock_area": {"region": "top_center", "x_min": 0.3, "x_max": 0.7, "y_max": 0.3},
                "play_clock_area": {"region": "top_right", "x_min": 0.6, "y_max": 0.3}
            },
            "aspect_ratio_constraints": {
                "hud": {"min_ratio": 3.0, "max_ratio": 20.0},
                "possession_triangle_area": {"min_ratio": 1.0, "max_ratio": 8.0},
                "territory_triangle_area": {"min_ratio": 0.5, "max_ratio": 4.0},
                "preplay_indicator": {"min_ratio": 1.0, "max_ratio": 6.0},
                "play_call_screen": {"min_ratio": 0.5, "max_ratio": 4.0},
                "down_distance_area": {"min_ratio": 1.5, "max_ratio": 8.0},
                "game_clock_area": {"min_ratio": 1.5, "max_ratio": 8.0},
                "play_clock_area": {"min_ratio": 0.8, "max_ratio": 4.0}
            }
        }
        
        print("🔍 Post-Processing Filters:")
        print("  📏 Size constraints for each class")
        print("  📍 Position constraints based on HUD layout")
        print("  📐 Aspect ratio constraints for realistic shapes")
        
        return filters
    
    def add_geometric_constraints(self) -> Dict:
        """
        Strategy 6: Geometric validation for HUD elements.
        """
        
        geometric_rules = {
            "triangle_validation": {
                "possession_triangle_area": {
                    "check_triangle_shape": True,
                    "min_triangle_area": 50,
                    "max_triangle_area": 2000,
                    "aspect_ratio_tolerance": 0.3
                },
                "territory_triangle_area": {
                    "check_triangle_shape": True,
                    "min_triangle_area": 30,
                    "max_triangle_area": 1000,
                    "aspect_ratio_tolerance": 0.3
                }
            },
            "text_validation": {
                "down_distance_area": {
                    "check_text_pattern": r"^\d+(st|nd|rd|th)\s*&\s*\d+$",
                    "min_text_confidence": 0.7
                },
                "game_clock_area": {
                    "check_text_pattern": r"^\d+(st|nd|rd|th)\s+\d{1,2}:\d{2}$",
                    "min_text_confidence": 0.6
                },
                "play_clock_area": {
                    "check_text_pattern": r"^\d{1,2}$",
                    "min_text_confidence": 0.6
                }
            },
            "spatial_relationships": {
                "hud_contains_elements": {
                    "description": "HUD should contain other elements",
                    "required_overlap": 0.8
                },
                "triangles_not_overlapping": {
                    "description": "Possession and territory triangles shouldn't overlap",
                    "max_overlap": 0.1
                }
            }
        }
        
        print("🔬 Geometric Validation Rules:")
        print("  🔺 Triangle shape validation for possession/territory indicators")
        print("  📝 Text pattern validation for down/distance and clock areas")
        print("  🗺️ Spatial relationship validation between elements")
        
        return geometric_rules
    
    def generate_comprehensive_strategy(self):
        """Generate a comprehensive false positive reduction strategy."""
        
        print("🎯 COMPREHENSIVE FALSE POSITIVE REDUCTION STRATEGY")
        print("=" * 60)
        
        # Run all strategies
        conf_thresholds = self.tune_confidence_thresholds()
        nms_settings = self.optimize_nms_settings()
        training_config = self.improve_training_config()
        data_strategies = self.enhance_data_quality()
        post_filters = self.add_post_processing_filters()
        
        print("\n📋 IMPLEMENTATION PRIORITY:")
        print("1. 🎓 Retrain with enhanced configuration (biggest impact)")
        print("2. 🔍 Apply post-processing filters (immediate improvement)")
        print("3. 📊 Add hard negative examples to training data")
        print("4. 🎯 Fine-tune confidence thresholds per class")
        
        return {
            'confidence_thresholds': conf_thresholds,
            'nms_settings': nms_settings,
            'training_config': training_config,
            'data_strategies': data_strategies,
            'post_filters': post_filters
        }

if __name__ == "__main__":
    reducer = FalsePositiveReducer()
    strategy = reducer.generate_comprehensive_strategy() 