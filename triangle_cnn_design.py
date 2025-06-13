"""
CNN-Based Triangle Detection for Madden 25 HUD
==============================================

This module outlines the design for replacing geometric validation
with a CNN-based approach for triangle detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class TriangleDirection(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"

@dataclass
class TrianglePrediction:
    """Result from CNN triangle detection."""
    is_triangle: bool
    direction: TriangleDirection
    confidence: float
    triangle_type: str  # 'possession' or 'territory'

class TriangleCNN(nn.Module):
    """
    Lightweight CNN for triangle detection in Madden HUD regions.
    
    Input: 64x64 RGB image patches from HUD regions
    Output: 
        - is_triangle: Binary classification (triangle vs not-triangle)
        - direction: 4-class classification (up/down/left/right)
        - confidence: Softmax probability
    """
    
    def __init__(self, num_classes_direction=4):
        super(TriangleCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 64x64 -> 8x8 after 3 pooling layers
        self.fc2 = nn.Linear(256, 128)
        
        # Output heads
        self.triangle_classifier = nn.Linear(128, 2)  # triangle vs not-triangle
        self.direction_classifier = nn.Linear(128, num_classes_direction)  # up/down/left/right
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output heads
        triangle_logits = self.triangle_classifier(x)
        direction_logits = self.direction_classifier(x)
        
        return triangle_logits, direction_logits

class TriangleCNNDetector:
    """
    CNN-based triangle detector for Madden HUD regions.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TriangleCNN()
        self.model.to(self.device)
        
        if model_path and model_path.exists():
            self.load_model(model_path)
        
        # Direction mapping
        self.direction_map = {
            0: TriangleDirection.UP,
            1: TriangleDirection.DOWN,
            2: TriangleDirection.LEFT,
            3: TriangleDirection.RIGHT
        }
    
    def load_model(self, model_path: Path):
        """Load trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def preprocess_roi(self, roi_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess ROI image for CNN input.
        
        Args:
            roi_img: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor ready for CNN
        """
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 64x64
        resized = cv2.resize(rgb_img, (64, 64))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def detect_triangle(self, roi_img: np.ndarray, triangle_type: str) -> TrianglePrediction:
        """
        Detect triangle in ROI using CNN.
        
        Args:
            roi_img: ROI image from YOLO detection
            triangle_type: 'possession' or 'territory'
            
        Returns:
            TrianglePrediction with results
        """
        # Preprocess image
        input_tensor = self.preprocess_roi(roi_img)
        
        # Run inference
        with torch.no_grad():
            triangle_logits, direction_logits = self.model(input_tensor)
            
            # Get probabilities
            triangle_probs = F.softmax(triangle_logits, dim=1)
            direction_probs = F.softmax(direction_logits, dim=1)
            
            # Get predictions
            is_triangle = triangle_probs[0, 1].item() > 0.5  # Index 1 = triangle class
            triangle_confidence = triangle_probs[0, 1].item()
            
            direction_idx = torch.argmax(direction_probs, dim=1).item()
            direction = self.direction_map[direction_idx]
            direction_confidence = direction_probs[0, direction_idx].item()
            
            # Overall confidence (geometric mean)
            overall_confidence = (triangle_confidence * direction_confidence) ** 0.5
        
        return TrianglePrediction(
            is_triangle=is_triangle,
            direction=direction,
            confidence=overall_confidence,
            triangle_type=triangle_type
        )

# Training Data Generation Strategy
class TriangleDatasetGenerator:
    """
    Generate training dataset for triangle CNN.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "triangles" / "up").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "triangles" / "down").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "triangles" / "left").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "triangles" / "right").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "non_triangles").mkdir(parents=True, exist_ok=True)
    
    def extract_training_patches(self, video_paths: List[Path]):
        """
        Extract training patches from Madden gameplay videos.
        
        Strategy:
        1. Use YOLO to detect triangle regions
        2. Manual annotation of actual triangles vs false positives
        3. Data augmentation (rotation, brightness, contrast)
        4. Generate synthetic negative examples (digits, other shapes)
        """
        pass
    
    def generate_synthetic_data(self):
        """
        Generate synthetic triangle and non-triangle examples.
        
        - Create perfect triangles in various orientations
        - Generate digit shapes (0-9) as negative examples
        - Add noise and variations to simulate real HUD conditions
        """
        pass

# Integration with existing system
def integrate_cnn_with_existing_system():
    """
    Integration plan for CNN-based triangle detection:
    
    1. Replace TriangleOrientationDetector.analyze_*_triangle methods
    2. Keep YOLO for region detection (it works well)
    3. Use CNN for final triangle validation instead of geometric rules
    4. Maintain same API for backward compatibility
    
    Benefits:
    - Drop-in replacement for existing geometric validation
    - Much more robust to real-world variations
    - Can be trained on actual Madden footage
    - Handles edge cases better than rule-based approach
    """
    pass

if __name__ == "__main__":
    print("CNN Triangle Detection Design")
    print("=" * 40)
    print()
    print("🎯 IMPLEMENTATION PLAN:")
    print()
    print("Phase 1: Data Collection")
    print("- Extract triangle regions from existing footage")
    print("- Manual annotation of true triangles vs false positives")
    print("- Generate synthetic training data")
    print()
    print("Phase 2: Model Training")
    print("- Train lightweight CNN on triangle dataset")
    print("- Validate on held-out Madden footage")
    print("- Optimize for inference speed")
    print()
    print("Phase 3: Integration")
    print("- Replace geometric validation with CNN")
    print("- Maintain existing YOLO + validation pipeline")
    print("- Test on real gameplay footage")
    print()
    print("🚀 EXPECTED BENEFITS:")
    print("- Much higher accuracy on real triangles")
    print("- Better rejection of digit false positives")
    print("- Robust to HUD variations and lighting")
    print("- Trainable on actual game footage") 