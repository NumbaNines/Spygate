"""
Configuration for triangle detection with validation.
"""
from typing import Dict, Tuple
from pydantic import BaseModel, Field

class TriangleDetectionConfig(BaseModel):
    """Configuration for triangle detection parameters."""
    
    # Contour approximation parameters
    EPSILON_FACTOR: float = Field(default=0.04, 
                                description="Factor for contour approximation precision")
    
    # Triangle orientation detection
    BASE_POINT_TOLERANCE: int = Field(default=5,
                                    description="Pixel tolerance for finding base points")
    ORIENTATION_X_TOLERANCE: int = Field(default=10,
                                       description="X-coordinate tolerance for orientation detection")
    
    # Image processing parameters
    GAUSSIAN_KERNEL: Tuple[int, int] = Field(default=(5, 5),
                                           description="Kernel size for Gaussian blur")
    GAUSSIAN_SIGMA: int = Field(default=0,
                              description="Sigma for Gaussian blur")
    CANNY_THRESHOLDS: Tuple[int, int] = Field(default=(50, 150),
                                            description="Thresholds for Canny edge detection")
    
    # Visualization colors (BGR format)
    COLORS: Dict[str, Tuple[int, int, int]] = Field(
        default={
            "possession": (0, 165, 255),  # Orange
            "territory": (128, 0, 128),   # Purple
            "text": (255, 255, 255),      # White
            "unknown": (128, 128, 128)    # Gray
        },
        description="Color mappings for visualization"
    )
    
    # Font settings
    FONT_SCALE: float = Field(default=0.6,
                            description="Font scale for OpenCV text")
    FONT_THICKNESS: int = Field(default=2,
                              description="Font thickness for OpenCV text")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        frozen = True  # Make config immutable 