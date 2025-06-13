from enum import Enum, auto
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Final
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TriangleType(Enum):
    """Types of triangles in Madden 25's HUD."""
    POSSESSION = auto()  # → arrow showing which team has the ball
    TERRITORY = auto()   # ▲/▼ showing field position

class Direction(Enum):
    """Possible triangle orientations."""
    LEFT = "left"       # ← possession arrow
    RIGHT = "right"     # → possession arrow
    UP = "up"          # ▲ territory (opponent's side)
    DOWN = "down"      # ▼ territory (own side)
    UNKNOWN = "unknown" # Invalid/undetectable orientation

@dataclass(frozen=True)
class TriangleOrientation:
    """Immutable representation of a triangle's orientation analysis."""
    is_valid: bool
    direction: Direction
    confidence: float
    center: Tuple[int, int]
    points: np.ndarray
    triangle_type: TriangleType
    validation_reason: str = ""  # Reason for validation result

    def __post_init__(self):
        """Validate the orientation data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not isinstance(self.direction, Direction):
            object.__setattr__(self, 'direction', Direction(self.direction))
        if not isinstance(self.triangle_type, TriangleType):
            raise ValueError(f"Invalid triangle type: {self.triangle_type}")

class TriangleOrientationDetector:
    """Detects and validates triangle orientations in Madden 25's HUD."""
    
    # Class constants
    MIN_AREA: Final[float] = 50.0   # Reduced from 100.0 for smaller triangles
    MAX_AREA: Final[float] = 1000.0
    MIN_ASPECT: Final[float] = 0.4  # Reduced from 0.5 for more flexibility
    MAX_ASPECT: Final[float] = 2.5  # Increased from 2.0 for more flexibility
    ANGLE_TOLERANCE: Final[float] = 20.0  # Increased from 15.0 degrees
    MIN_CONFIDENCE: Final[float] = 0.2  # Reduced from 0.3 for more lenient acceptance
    
    # Aspect ratio thresholds (RELAXED)
    POSSESSION_MIN_ASPECT: Final[float] = 1.0  # Reduced from 1.2 for more arrow shapes
    TERRITORY_MIN_ASPECT: Final[float] = 0.6   # Reduced from 0.8 for more triangle shapes
    TERRITORY_MAX_ASPECT: Final[float] = 1.5   # Increased from 1.2 for more triangle shapes
    
    # Enhanced validation thresholds (RELAXED for better triangle detection)
    MIN_CONVEXITY: Final[float] = 0.70  # Reduced from 0.85 - allows more curved shapes
    MAX_VERTICES: Final[int] = 8        # Increased from 6 - allows more complex shapes
    MIN_ANGLE: Final[float] = 25.0      # Reduced from 30.0 - allows sharper angles
    MAX_ANGLE: Final[float] = 130.0     # Increased from 120.0 - allows wider angles
    SYMMETRY_TOLERANCE: Final[float] = 0.5  # Increased from 0.3 - more lenient symmetry
    
    def __init__(self, debug_output_dir: Optional[Path] = None):
        """
        Initialize the triangle detector.
        
        Args:
            debug_output_dir: Optional directory to save debug visualizations
        """
        self.debug_output_dir = debug_output_dir
        if debug_output_dir and not debug_output_dir.exists():
            debug_output_dir.mkdir(parents=True)
    
    def analyze_possession_triangle(self, contour: np.ndarray, frame: Optional[np.ndarray] = None) -> TriangleOrientation:
        """
        Analyze possession triangle (→) orientation.
        Returns left/right direction based on arrow shape.
        """
        try:
            # Step 1: Enhanced shape validation
            validation_result = self._validate_enhanced_triangle(contour, TriangleType.POSSESSION)
            if not validation_result[0]:
                return self._create_invalid_result(contour, TriangleType.POSSESSION, reason=validation_result[1])
            
            # Step 2: Get the box that contains our triangle
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # Step 3: Check if it's wide enough to be an arrow (arrows are wider than tall)
            aspect = w / h if h != 0 else 0
            if aspect < self.POSSESSION_MIN_ASPECT:  # Must be at least 1.2 times wider than tall
                return self._create_invalid_result(contour, TriangleType.POSSESSION, center, "Aspect ratio too low for arrow")
            
            # Step 4: Find the leftmost and rightmost points
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])   # Point with smallest x
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])  # Point with largest x
            
            # Step 5: Find the arrow's point (the point furthest from the left-right line)
            left_right_line = np.array([leftmost, rightmost])
            distances = []
            for point in contour:
                point = point[0]
                # Calculate perpendicular distance from point to left-right line
                dist = np.abs(np.cross(left_right_line[1] - left_right_line[0], point - left_right_line[0])) / np.linalg.norm(left_right_line[1] - left_right_line[0])
                distances.append((dist, point))
            
            # The point furthest from the line is our arrow's point
            arrow_point = max(distances, key=lambda x: x[0])[1]
            
            # Step 6: Determine direction by checking if arrow point is closer to right or left
            right_dist = np.linalg.norm(arrow_point - rightmost)  # Distance to right side
            left_dist = np.linalg.norm(arrow_point - leftmost)    # Distance to left side
            
            # If closer to right side, arrow points right, otherwise left
            direction = Direction.RIGHT if right_dist < left_dist else Direction.LEFT
            confidence = self._calculate_enhanced_confidence(contour, aspect, TriangleType.POSSESSION)
            
            # Save debug visualization if enabled
            if self.debug_output_dir and frame is not None:
                self._save_debug_visualization(frame, contour, direction, confidence, TriangleType.POSSESSION)
            
            return TriangleOrientation(
                is_valid=True,
                direction=direction,
                confidence=confidence,
                center=center,
                points=contour,
                triangle_type=TriangleType.POSSESSION,
                validation_reason="Passed enhanced validation"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing possession triangle: {e}")
            return self._create_invalid_result(contour, TriangleType.POSSESSION, reason=f"Exception: {e}")
    
    def analyze_territory_triangle(self, contour: np.ndarray, frame: Optional[np.ndarray] = None) -> TriangleOrientation:
        """
        Analyze territory triangle (▲/▼) orientation.
        Returns up/down direction based on triangle orientation.
        """
        try:
            # Step 1: Enhanced shape validation
            validation_result = self._validate_enhanced_triangle(contour, TriangleType.TERRITORY)
            if not validation_result[0]:
                return self._create_invalid_result(contour, TriangleType.TERRITORY, reason=validation_result[1])
            
            # Step 2: Get the box that contains our triangle
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # Step 3: Check if it's roughly equilateral (height ≈ width)
            aspect = w / h if h != 0 else 0
            if not (self.TERRITORY_MIN_ASPECT <= aspect <= self.TERRITORY_MAX_ASPECT):  # Should be between 0.8 and 1.2
                return self._create_invalid_result(contour, TriangleType.TERRITORY, center, "Aspect ratio not suitable for territory triangle")
            
            # Step 4: Find the top and bottom points
            topmost = tuple(contour[contour[:,:,1].argmin()][0])    # Point with smallest y
            bottommost = tuple(contour[contour[:,:,1].argmax()][0]) # Point with largest y
            
            # Step 5: Find the base point (point furthest from top-bottom line)
            top_bottom_line = np.array([topmost, bottommost])
            distances = []
            for point in contour:
                point = point[0]
                # Calculate perpendicular distance from point to top-bottom line
                dist = np.abs(np.cross(top_bottom_line[1] - top_bottom_line[0], point - top_bottom_line[0])) / np.linalg.norm(top_bottom_line[1] - top_bottom_line[0])
                distances.append((dist, point))
            
            # The point furthest from the line is our base point
            base_point = max(distances, key=lambda x: x[0])[1]
            
            # Step 6: Determine direction by checking if base is below center
            # If base is below center, triangle points up, otherwise down
            direction = Direction.UP if base_point[1] > center[1] else Direction.DOWN
            confidence = self._calculate_enhanced_confidence(contour, aspect, TriangleType.TERRITORY)
            
            # Save debug visualization if enabled
            if self.debug_output_dir and frame is not None:
                self._save_debug_visualization(frame, contour, direction, confidence, TriangleType.TERRITORY)
            
            return TriangleOrientation(
                is_valid=True,
                direction=direction,
                confidence=confidence,
                center=center,
                points=contour,
                triangle_type=TriangleType.TERRITORY,
                validation_reason="Passed enhanced validation"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing territory triangle: {e}")
            return self._create_invalid_result(contour, TriangleType.TERRITORY, reason=f"Exception: {e}")
    
    def _validate_enhanced_triangle(self, contour: np.ndarray, triangle_type: TriangleType) -> Tuple[bool, str]:
        """
        Enhanced triangle validation with multiple checks to reject digit shapes.
        
        Args:
            contour: The contour points to validate
            triangle_type: Type of triangle being validated
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check 1: Basic validation
            if not self._validate_basic_triangle(contour):
                return False, "Failed basic triangle validation"
            
            # Check 2: Convexity test - real triangles should be mostly convex
            convexity = self._calculate_convexity(contour)
            if convexity < self.MIN_CONVEXITY:
                return False, f"Low convexity: {convexity:.3f} < {self.MIN_CONVEXITY}"
            
            # Check 3: Vertex count - should have approximately 3 main vertices
            vertices = self._find_main_vertices(contour)
            if len(vertices) > self.MAX_VERTICES:
                return False, f"Too many vertices: {len(vertices)} > {self.MAX_VERTICES}"
            
            # Check 4: Angle analysis - should have triangle-like angles
            if len(vertices) >= 3:
                angles = self._calculate_vertex_angles(vertices)
                valid_angles = [a for a in angles if self.MIN_ANGLE <= a <= self.MAX_ANGLE]
                if len(valid_angles) < 2:  # Need at least 2 valid triangle angles
                    return False, f"Invalid angles: {angles}, valid: {len(valid_angles)}"
            
            # Check 5: Symmetry test for territory triangles
            if triangle_type == TriangleType.TERRITORY:
                symmetry_score = self._calculate_symmetry(contour)
                if symmetry_score > self.SYMMETRY_TOLERANCE:
                    return False, f"Poor symmetry: {symmetry_score:.3f} > {self.SYMMETRY_TOLERANCE}"
            
            return True, "Passed all enhanced validation checks"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _calculate_convexity(self, contour: np.ndarray) -> float:
        """Calculate convexity ratio (contour area / convex hull area)."""
        try:
            contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area == 0:
                return 0.0
                
            return contour_area / hull_area
        except:
            return 0.0
    
    def _find_main_vertices(self, contour: np.ndarray) -> List[Tuple[int, int]]:
        """Find main vertices using Douglas-Peucker approximation."""
        try:
            # Use adaptive epsilon based on contour perimeter
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return [tuple(point[0]) for point in approx]
        except:
            return []
    
    def _calculate_vertex_angles(self, vertices: List[Tuple[int, int]]) -> List[float]:
        """Calculate angles at each vertex."""
        if len(vertices) < 3:
            return []
        
        angles = []
        for i in range(len(vertices)):
            p1 = np.array(vertices[i-1])
            p2 = np.array(vertices[i])
            p3 = np.array(vertices[(i+1) % len(vertices)])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            try:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
            except:
                continue
                
        return angles
    
    def _calculate_symmetry(self, contour: np.ndarray) -> float:
        """Calculate symmetry score for territory triangles (lower is more symmetric)."""
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            
            # Calculate left and right side areas
            left_points = [p for p in contour if p[0][0] < center_x]
            right_points = [p for p in contour if p[0][0] >= center_x]
            
            if len(left_points) == 0 or len(right_points) == 0:
                return 1.0  # Maximum asymmetry
            
            # Mirror right side to left and calculate difference
            mirrored_right = []
            for p in right_points:
                mirrored_x = 2 * center_x - p[0][0]
                mirrored_right.append([[mirrored_x, p[0][1]]])
            
            # Simple symmetry metric: ratio of point count difference
            count_diff = abs(len(left_points) - len(mirrored_right))
            total_points = len(left_points) + len(mirrored_right)
            
            return count_diff / total_points if total_points > 0 else 1.0
            
        except:
            return 1.0  # Maximum asymmetry on error
    
    def _validate_basic_triangle(self, contour: np.ndarray) -> bool:
        """
        Basic triangle validation checks.
        
        Args:
            contour: The contour points to validate
            
        Returns:
            bool: True if the contour passes basic validation
        """
        try:
            # Check if we have enough points
            if len(contour) < 3:
                logger.debug("Contour has fewer than 3 points")
                return False
            
            # Check area constraints
            area = cv2.contourArea(contour)
            if not (self.MIN_AREA <= area <= self.MAX_AREA):
                logger.debug(f"Area {area} outside valid range [{self.MIN_AREA}, {self.MAX_AREA}]")
                return False
            
            # Check aspect ratio constraints
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                logger.debug("Height is zero")
                return False
                
            aspect = w / h
            if not (self.MIN_ASPECT <= aspect <= self.MAX_ASPECT):
                logger.debug(f"Aspect ratio {aspect} outside valid range [{self.MIN_ASPECT}, {self.MAX_ASPECT}]")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in basic triangle validation: {e}")
            return False

    def _calculate_enhanced_confidence(self, contour: np.ndarray, aspect: float, triangle_type: TriangleType) -> float:
        """
        Calculate enhanced confidence score incorporating multiple validation metrics.
        
        Args:
            contour: The triangle contour
            aspect: Aspect ratio of the bounding rectangle
            triangle_type: Type of triangle being analyzed
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            confidence_factors = []
            
            # Factor 1: Area-based confidence
            area = cv2.contourArea(contour)
            optimal_area = (self.MIN_AREA + self.MAX_AREA) / 2
            area_confidence = 1.0 - abs(area - optimal_area) / optimal_area
            confidence_factors.append(max(0.0, area_confidence))
            
            # Factor 2: Aspect ratio confidence
            if triangle_type == TriangleType.POSSESSION:
                optimal_aspect = 1.5  # Slightly wider than tall for arrows
                aspect_range = self.POSSESSION_MIN_ASPECT
            else:  # TERRITORY
                optimal_aspect = 1.0  # Square-ish for territory triangles
                aspect_range = (self.TERRITORY_MAX_ASPECT - self.TERRITORY_MIN_ASPECT) / 2
            
            aspect_confidence = 1.0 - abs(aspect - optimal_aspect) / aspect_range
            confidence_factors.append(max(0.0, aspect_confidence))
            
            # Factor 3: Convexity confidence
            convexity = self._calculate_convexity(contour)
            convexity_confidence = min(1.0, convexity / self.MIN_CONVEXITY)
            confidence_factors.append(convexity_confidence)
            
            # Factor 4: Vertex count confidence
            vertices = self._find_main_vertices(contour)
            vertex_confidence = max(0.0, 1.0 - (len(vertices) - 3) / 3)  # Penalty for extra vertices
            confidence_factors.append(vertex_confidence)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.25, 0.15]  # Area and aspect most important
            final_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
            
            return max(self.MIN_CONFIDENCE, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
            return self.MIN_CONFIDENCE

    def _create_invalid_result(
        self, 
        contour: np.ndarray, 
        triangle_type: TriangleType,
        center: Tuple[int, int] = (0, 0),
        reason: str = "Failed validation"
    ) -> TriangleOrientation:
        """Create an invalid triangle orientation result."""
        return TriangleOrientation(
            is_valid=False,
            direction=Direction.UNKNOWN,
            confidence=0.0,
            center=center,
            points=contour,
            triangle_type=triangle_type,
            validation_reason=reason
        )

    def _save_debug_visualization(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        direction: Direction,
        confidence: float,
        triangle_type: TriangleType
    ) -> None:
        """Save debug visualization of triangle detection."""
        if not self.debug_output_dir:
            return
            
        try:
            # Create visualization
            vis = frame.copy()
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            
            # Add text annotation
            x, y, w, h = cv2.boundingRect(contour)
            text = f"{triangle_type.name}: {direction.value} ({confidence:.2f})"
            cv2.putText(vis, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save
            filename = f"{triangle_type.name.lower()}_{direction.value}_{confidence:.2f}.png"
            cv2.imwrite(str(self.debug_output_dir / filename), vis)
            
        except Exception as e:
            logger.error(f"Error saving debug visualization: {e}") 