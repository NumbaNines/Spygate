"""
Template-based triangle detection that integrates with YOLO detections.
Uses YOLO-detected bounding boxes for precise template matching.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Removed circular import - EnhancedGameAnalyzer imports this module


class TriangleType(Enum):
    POSSESSION = "possession"
    TERRITORY = "territory"


class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@dataclass
class TemplateMatch:
    """A template match result with scale information."""

    position: tuple[int, int]  # Top-left corner
    confidence: float
    scale_factor: float
    template_name: str
    triangle_type: TriangleType
    direction: Direction
    bounding_box: tuple[int, int, int, int]  # x, y, w, h


class TemplateTriangleDetector:
    """
    Standalone triangle detector using template matching.
    Can be used independently without YOLO integration.
    """

    def __init__(self, debug_output_dir: Optional[Path] = None):
        """
        Initialize standalone template triangle detector.

        Args:
            debug_output_dir: Directory for debug output
        """
        self.templates_dir = Path("templates/triangles")
        self.debug_output_dir = debug_output_dir

        if self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        # Template matching parameters - Expert-calibrated based on empirical testing
        # Real-world data: Average triangle confidence ~0.69 (much higher than down detection)
        self.SCALE_FACTORS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
        self.MIN_MATCH_CONFIDENCE = 0.45  # Production-ready: Well below 0.69 average, allows good detections
        self.NMS_OVERLAP_THRESHOLD = 0.5

        # Load templates
        self.templates = {}
        self.template_metadata = {}
        self.load_templates()

        # Create default templates if none exist
        if not self.templates:
            self.create_default_templates()
            self.load_templates()

    def detect_triangles_in_roi(
        self, roi_img: np.ndarray, triangle_type_str: str
    ) -> list[TemplateMatch]:
        """
        Detect triangles in a given ROI using template matching.

        Args:
            roi_img: Region of interest image
            triangle_type_str: "possession" or "territory"

        Returns:
            List of template matches
        """
        if roi_img is None or roi_img.size == 0:
            return []

        triangle_type = (
            TriangleType.POSSESSION if triangle_type_str == "possession" else TriangleType.TERRITORY
        )

        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        matches = []

        # Get relevant templates for this triangle type
        relevant_templates = {
            name: template
            for name, template in self.templates.items()
            if self.template_metadata[name]["triangle_type"] == triangle_type.value
        }

        # Try each template at multiple scales
        for template_name, template_data in relevant_templates.items():
            template = template_data["image"]
            metadata = self.template_metadata[template_name]

            for scale in self.SCALE_FACTORS:
                # Scale the template
                scaled_w = int(template.shape[1] * scale)
                scaled_h = int(template.shape[0] * scale)

                # Skip if scaled template is larger than ROI
                if scaled_w > gray_roi.shape[1] or scaled_h > gray_roi.shape[0]:
                    continue

                # Skip if scaled template is too small
                if scaled_w < 5 or scaled_h < 5:
                    continue

                scaled_template = cv2.resize(template, (scaled_w, scaled_h))

                # Template matching
                result = cv2.matchTemplate(gray_roi, scaled_template, cv2.TM_CCOEFF_NORMED)

                # Find matches above threshold
                locations = np.where(result >= self.MIN_MATCH_CONFIDENCE)

                for pt in zip(*locations[::-1]):  # Switch x and y
                    confidence = result[pt[1], pt[0]]

                    match = TemplateMatch(
                        position=(pt[0], pt[1]),
                        confidence=confidence,
                        scale_factor=scale,
                        template_name=template_name,
                        triangle_type=triangle_type,
                        direction=Direction(metadata["direction"]),
                        bounding_box=(pt[0], pt[1], scaled_w, scaled_h),
                    )
                    matches.append(match)

        return matches

    def select_best_single_triangles(
        self, matches: list[TemplateMatch], triangle_type_str: str
    ) -> Optional[dict]:
        """
        Select the best single triangle from matches.

        Args:
            matches: List of template matches
            triangle_type_str: "possession" or "territory"

        Returns:
            Dictionary with best triangle info or None
        """
        if not matches:
            return None

        # Apply NMS first
        nms_matches = self.apply_nms(matches)

        if not nms_matches:
            return None

        # Select the best one using our scoring system
        best_match = self._select_best_triangle(nms_matches, triangle_type_str)

        if best_match:
            return {
                "direction": best_match.direction.value,
                "confidence": best_match.confidence,
                "position": best_match.position,
                "bounding_box": best_match.bounding_box,
                "template_name": best_match.template_name,
            }

        return None

    # Add shared methods from YOLOIntegratedTriangleDetector
    def apply_nms(self, matches: list[TemplateMatch]) -> list[TemplateMatch]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not matches:
            return []

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        final_matches = []

        for match in matches:
            # Check if this match overlaps significantly with any accepted match
            is_duplicate = False

            for accepted in final_matches:
                overlap = self.calculate_overlap(match.bounding_box, accepted.bounding_box)
                if overlap > self.NMS_OVERLAP_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_matches.append(match)

        return final_matches

    def _select_best_triangle(
        self, matches: list[TemplateMatch], triangle_type: str
    ) -> Optional[TemplateMatch]:
        """Select the best triangle from a list of matches using advanced scoring."""
        if not matches:
            return None

        # Calculate average area for size scoring
        areas = [m.bounding_box[2] * m.bounding_box[3] for m in matches]
        avg_area = sum(areas) / len(areas) if areas else 0

        best_match = None
        best_score = -1

        for match in matches:
            # Calculate composite score with weighted factors
            confidence_score = match.confidence * 0.35  # 35% weight
            template_quality_score = (
                self._calculate_template_quality_score(match.template_name) * 0.25
            )  # 25% weight
            size_score = self._calculate_smart_size_score(match, avg_area) * 0.15  # 15% weight
            position_score = (
                self._calculate_position_score(match, triangle_type) * 0.10
            )  # 10% weight
            scale_score = self._calculate_scale_score(match.scale_factor) * 0.10  # 10% weight
            aspect_score = self._calculate_aspect_ratio_score(match) * 0.05  # 5% weight

            total_score = (
                confidence_score
                + template_quality_score
                + size_score
                + position_score
                + scale_score
                + aspect_score
            )

            if total_score > best_score:
                best_score = total_score
                best_match = match

        return best_match

    def _calculate_template_quality_score(self, template_name: str) -> float:
        """Calculate quality score based on template name (prefers Madden-specific templates)."""
        if "madden" in template_name.lower():
            return 1.0  # Highest quality for game-specific templates
        elif "arrow" in template_name.lower():
            return 0.8  # Good quality for arrow templates
        elif "triangle" in template_name.lower():
            return 0.6  # Medium quality for generic triangles
        else:
            return 0.4  # Lower quality for unknown templates

    def _calculate_smart_size_score(self, match: TemplateMatch, avg_area: float) -> float:
        """Calculate size score with optimal ranges, not just 'bigger is better'."""
        w, h = match.bounding_box[2], match.bounding_box[3]
        area = w * h

        # Define optimal size ranges based on triangle type
        if match.triangle_type == TriangleType.POSSESSION:
            optimal_min, optimal_max = 120, 800  # Possession triangles tend to be medium-sized
        else:  # TERRITORY
            optimal_min, optimal_max = 100, 625  # Territory triangles tend to be smaller

        if optimal_min <= area <= optimal_max:
            return 1.0  # Perfect size
        elif area < optimal_min:
            # Penalize too small (might be noise)
            return max(0.1, area / optimal_min)
        else:
            # Penalize too large (might be false positive)
            return max(0.1, optimal_max / area)

    def _calculate_scale_score(self, scale_factor: float) -> float:
        """Calculate score based on scale factor (prefer reasonable scales)."""
        if 0.8 <= scale_factor <= 2.0:
            return 1.0  # Optimal scale range
        elif 0.5 <= scale_factor <= 4.0:
            return 0.7  # Acceptable scale range
        else:
            return 0.3  # Poor scale range

    def _calculate_aspect_ratio_score(self, match: TemplateMatch) -> float:
        """Calculate score based on aspect ratio (triangles should have reasonable proportions)."""
        w, h = match.bounding_box[2], match.bounding_box[3]
        if h == 0:
            return 0.0

        aspect_ratio = w / h

        # Triangles should have reasonable aspect ratios
        if 0.5 <= aspect_ratio <= 2.0:
            return 1.0  # Good aspect ratio
        elif 0.3 <= aspect_ratio <= 3.0:
            return 0.7  # Acceptable aspect ratio
        else:
            return 0.2  # Poor aspect ratio

    def _calculate_position_score(self, match: TemplateMatch, triangle_type: str) -> float:
        """Calculate score based on position (possession left, territory right)."""
        x = match.position[0]

        if triangle_type == "possession":
            # Possession triangles should be on the left side
            return 1.0 if x < 640 else 0.5  # Assume 1280px width
        else:  # territory
            # Territory triangles should be on the right side
            return 1.0 if x > 640 else 0.5

    def calculate_overlap(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def create_default_templates(self):
        """Create default triangle templates if none exist."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Create basic arrow templates
        self.create_arrow_template("arrow_left", Direction.LEFT, TriangleType.POSSESSION)
        self.create_arrow_template("arrow_right", Direction.RIGHT, TriangleType.POSSESSION)
        self.create_triangle_template("triangle_up", Direction.UP, TriangleType.TERRITORY)
        self.create_triangle_template("triangle_down", Direction.DOWN, TriangleType.TERRITORY)

    def create_arrow_template(self, name: str, direction: Direction, triangle_type: TriangleType):
        """Create a simple arrow template."""
        template = np.zeros((20, 20), dtype=np.uint8)

        if direction == Direction.LEFT:
            # Create left-pointing arrow
            cv2.fillPoly(template, [np.array([[15, 10], [5, 5], [5, 15]])], 255)
        else:  # RIGHT
            # Create right-pointing arrow
            cv2.fillPoly(template, [np.array([[5, 10], [15, 5], [15, 15]])], 255)

        self.save_template(name, template, direction, triangle_type)

    def create_triangle_template(
        self, name: str, direction: Direction, triangle_type: TriangleType
    ):
        """Create a simple triangle template."""
        template = np.zeros((20, 20), dtype=np.uint8)

        if direction == Direction.UP:
            # Create upward-pointing triangle
            cv2.fillPoly(template, [np.array([[10, 5], [5, 15], [15, 15]])], 255)
        else:  # DOWN
            # Create downward-pointing triangle
            cv2.fillPoly(template, [np.array([[10, 15], [5, 5], [15, 5]])], 255)

        self.save_template(name, template, direction, triangle_type)

    def save_template(
        self, name: str, template: np.ndarray, direction: Direction, triangle_type: TriangleType
    ):
        """Save template with metadata."""
        template_path = self.templates_dir / f"{name}.png"
        metadata_path = self.templates_dir / f"{name}.json"

        # Save template image
        cv2.imwrite(str(template_path), template)

        # Save metadata
        metadata = {
            "direction": direction.value,
            "triangle_type": triangle_type.value,
            "created_date": str(Path().cwd()),
            "template_size": template.shape,
        }

        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_templates(self):
        """Load all templates from the templates directory."""
        if not self.templates_dir.exists():
            return

        import json

        for template_file in self.templates_dir.glob("*.png"):
            metadata_file = template_file.with_suffix(".json")

            if metadata_file.exists():
                # Load template image
                template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)

                if template_img is not None:
                    # Load metadata
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    self.templates[template_file.stem] = {
                        "image": template_img,
                        "path": template_file,
                    }
                    self.template_metadata[template_file.stem] = metadata


class YOLOIntegratedTriangleDetector:
    """
    Triangle detector that uses YOLO-detected bounding boxes for precise template matching.
    Integrates with the existing game analyzer.
    """

    def __init__(
        self,
        game_analyzer,
        templates_dir: Optional[Path] = None,
        debug_output_dir: Optional[Path] = None,
    ):
        """
        Initialize detector with existing YOLO game analyzer.

        Args:
            game_analyzer: Existing game analyzer instance with trained YOLO model
            templates_dir: Directory containing triangle templates
            debug_output_dir: Directory for debug output
        """
        self.game_analyzer = game_analyzer
        self.templates_dir = templates_dir or Path("templates/triangles")
        self.debug_output_dir = debug_output_dir

        if self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        # Template matching parameters - Expert-calibrated based on empirical testing
        # Real-world data: Average triangle confidence ~0.69 (much higher than down detection)
        self.SCALE_FACTORS = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
        self.MIN_MATCH_CONFIDENCE = 0.45  # Production-ready: Well below 0.69 average, allows good detections
        self.NMS_OVERLAP_THRESHOLD = 0.5

        # Load templates
        self.templates = {}
        self.template_metadata = {}
        self.load_templates()

        # Create default templates if none exist
        if not self.templates:
            self.create_default_templates()
            self.load_templates()

    def detect_triangles_in_yolo_regions(self, frame: np.ndarray) -> list[TemplateMatch]:
        """
        Use YOLO to detect triangle regions, then apply template matching within those regions.
        This is much more precise than searching broad ROI areas.
        """
        # Get YOLO detections from the game analyzer
        detections = self.game_analyzer.model.detect(frame)

        all_matches = []

        for detection in detections:
            # Check if this is a triangle area detection
            class_name = detection["class"]  # Already a string class name

            if class_name == "possession_triangle_area":
                # Extract the precise YOLO-detected bounding box
                bbox = detection["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)

                # Extract ROI from YOLO detection
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    # Apply template matching within this precise region
                    matches = self.match_templates_in_roi(
                        roi, TriangleType.POSSESSION, offset=(x1, y1)
                    )
                    all_matches.extend(matches)

            elif class_name == "territory_triangle_area":
                # Extract the precise YOLO-detected bounding box
                bbox = detection["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)

                # Extract ROI from YOLO detection
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    # Apply template matching within this precise region
                    matches = self.match_templates_in_roi(
                        roi, TriangleType.TERRITORY, offset=(x1, y1)
                    )
                    all_matches.extend(matches)

        # Apply non-maximum suppression to remove overlapping detections
        nms_matches = self.apply_nms(all_matches)

        # Select exactly 1 possession and 1 territory triangle (our main strategy!)
        final_matches = self.select_best_single_triangles(nms_matches)

        print(
            f"🎯 Triangle Detection Summary: {len(all_matches)} raw → {len(nms_matches)} after NMS → {len(final_matches)} final"
        )

        # Debug visualization
        if self.debug_output_dir:
            self.create_template_debug_visualization(frame, final_matches)

        return final_matches

    def match_templates_in_roi(
        self, roi_img: np.ndarray, triangle_type: TriangleType, offset: tuple[int, int]
    ) -> list[TemplateMatch]:
        """
        Match templates in a specific YOLO-detected ROI with multi-scale search.
        """
        if roi_img is None or roi_img.size == 0:
            return []

        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        matches = []

        # Get relevant templates for this triangle type
        relevant_templates = {
            name: template
            for name, template in self.templates.items()
            if self.template_metadata[name]["triangle_type"] == triangle_type.value
        }

        # Try each template at multiple scales
        for template_name, template_data in relevant_templates.items():
            template = template_data["image"]
            metadata = self.template_metadata[template_name]

            for scale in self.SCALE_FACTORS:
                # Scale the template
                scaled_w = int(template.shape[1] * scale)
                scaled_h = int(template.shape[0] * scale)

                # Skip if scaled template is larger than ROI
                if scaled_w > gray_roi.shape[1] or scaled_h > gray_roi.shape[0]:
                    continue

                # Skip if scaled template is too small
                if scaled_w < 5 or scaled_h < 5:
                    continue

                scaled_template = cv2.resize(template, (scaled_w, scaled_h))

                # Template matching
                result = cv2.matchTemplate(gray_roi, scaled_template, cv2.TM_CCOEFF_NORMED)

                # Find matches above threshold
                locations = np.where(result >= self.MIN_MATCH_CONFIDENCE)

                for pt in zip(*locations[::-1]):  # Switch x and y
                    confidence = result[pt[1], pt[0]]

                    match = TemplateMatch(
                        position=(pt[0] + offset[0], pt[1] + offset[1]),
                        confidence=confidence,
                        scale_factor=scale,
                        template_name=template_name,
                        triangle_type=triangle_type,
                        direction=Direction(metadata["direction"]),
                        bounding_box=(pt[0] + offset[0], pt[1] + offset[1], scaled_w, scaled_h),
                    )
                    matches.append(match)

        return matches

    def apply_nms(self, matches: list[TemplateMatch]) -> list[TemplateMatch]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not matches:
            return []

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        final_matches = []

        for match in matches:
            # Check if this match overlaps significantly with any accepted match
            is_duplicate = False

            for accepted in final_matches:
                overlap = self.calculate_overlap(match.bounding_box, accepted.bounding_box)
                if overlap > self.NMS_OVERLAP_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_matches.append(match)

        return final_matches

    def select_best_single_triangles(self, matches: list[TemplateMatch]) -> list[TemplateMatch]:
        """
        Select exactly 1 possession triangle and 1 territory triangle from all matches.
        Uses a scoring system based on confidence, size, and position to pick the best ones.
        """
        if not matches:
            return []

        # Separate by triangle type
        possession_matches = [m for m in matches if m.triangle_type == TriangleType.POSSESSION]
        territory_matches = [m for m in matches if m.triangle_type == TriangleType.TERRITORY]

        best_triangles = []

        # Select best possession triangle
        if possession_matches:
            best_possession = self._select_best_triangle(possession_matches, "possession")
            if best_possession:
                best_triangles.append(best_possession)

        # Select best territory triangle
        if territory_matches:
            best_territory = self._select_best_triangle(territory_matches, "territory")
            if best_territory:
                best_triangles.append(best_territory)

        return best_triangles

    def _select_best_triangle(
        self, matches: list[TemplateMatch], triangle_type: str
    ) -> Optional[TemplateMatch]:
        """
        Select the single best triangle from a list of matches using an advanced scoring system.
        Prioritizes: 1) Confidence, 2) Template quality, 3) Optimal size range, 4) Position, 5) Scale factor
        """
        if not matches:
            return None

        # Calculate composite scores for each match
        scored_matches = []

        # Get size statistics for better size scoring
        areas = [match.bounding_box[2] * match.bounding_box[3] for match in matches]
        avg_area = sum(areas) / len(areas) if areas else 100

        for match in matches:
            # 1. Confidence score (0.0 - 1.0) - Most important
            confidence_score = match.confidence

            # 2. Template quality score - prefer specific Madden templates
            template_quality_score = self._calculate_template_quality_score(match.template_name)

            # 3. Smart size score - optimal range, not just "bigger is better"
            size_score = self._calculate_smart_size_score(match, avg_area)

            # 4. Position score - triangles should be in expected locations
            position_score = self._calculate_position_score(match, triangle_type)

            # 5. Scale score - prefer reasonable scales (not too extreme)
            scale_score = self._calculate_scale_score(match.scale_factor)

            # 6. Aspect ratio score - triangles should have reasonable proportions
            aspect_ratio_score = self._calculate_aspect_ratio_score(match)

            # Composite score with refined weights
            composite_score = (
                confidence_score * 0.35
                + template_quality_score * 0.25  # 35% confidence (most important)
                + size_score * 0.15  # 25% template quality
                + position_score * 0.10  # 15% smart size
                + scale_score * 0.10  # 10% position
                + aspect_ratio_score * 0.05  # 10% scale factor  # 5% aspect ratio
            )

            scored_matches.append((composite_score, match))

        # Sort by composite score (highest first) and return the best
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        best_match = scored_matches[0][1]

        # Show detailed scoring for debugging
        print(f"🎯 Selected best {triangle_type} triangle: {best_match.template_name}")
        print(f"   📊 Confidence: {best_match.confidence:.3f}")
        print(f"   📏 Size: {best_match.bounding_box[2]}x{best_match.bounding_box[3]} pixels")
        print(f"   🔍 Scale: {best_match.scale_factor:.2f}x")
        print(f"   🏆 Final Score: {scored_matches[0][0]:.3f}")

        # Show runner-up for comparison
        if len(scored_matches) > 1:
            runner_up = scored_matches[1][1]
            print(
                f"   🥈 Runner-up: {runner_up.template_name} (score: {scored_matches[1][0]:.3f}, conf: {runner_up.confidence:.3f})"
            )

        return best_match

    def _calculate_template_quality_score(self, template_name: str) -> float:
        """
        Score templates based on their quality/specificity.
        Prefer Madden-specific templates over generic ones.
        """
        # Madden-specific templates are highest quality
        if "madden" in template_name.lower():
            return 1.0

        # Game-specific templates are good
        if any(game in template_name.lower() for game in ["cfb", "ncaa", "fifa"]):
            return 0.8

        # Generic templates are acceptable
        if any(
            term in template_name.lower()
            for term in ["possession", "territory", "triangle", "arrow"]
        ):
            return 0.6

        # Unknown templates get lower score
        return 0.4

    def _calculate_smart_size_score(self, match: TemplateMatch, avg_area: float) -> float:
        """
        Smart size scoring that prefers optimal size ranges, not just bigger.
        Real triangles are usually in a specific size range.
        """
        x, y, w, h = match.bounding_box
        area = w * h

        # Define optimal size ranges for triangles (based on typical HUD sizes)
        if match.triangle_type == TriangleType.POSSESSION:
            # Possession arrows are typically 15-40 pixels wide, 8-20 pixels tall
            optimal_min_area = 120  # 15x8
            optimal_max_area = 800  # 40x20
        else:  # TERRITORY
            # Territory triangles are typically 10-25 pixels wide/tall
            optimal_min_area = 100  # 10x10
            optimal_max_area = 625  # 25x25

        # Score based on how close to optimal range
        if optimal_min_area <= area <= optimal_max_area:
            # Perfect size range
            return 1.0
        elif area < optimal_min_area:
            # Too small - penalize heavily
            ratio = area / optimal_min_area
            return max(0.1, ratio * 0.7)  # Max 70% score for undersized
        else:
            # Too large - penalize but not as heavily as too small
            ratio = optimal_max_area / area
            return max(0.2, ratio * 0.8)  # Max 80% score for oversized

    def _calculate_scale_score(self, scale_factor: float) -> float:
        """
        Score scale factors, preferring reasonable scales.
        Extreme scales (too small/large) are often false positives.
        """
        # Optimal scale range is 0.8 to 2.0
        if 0.8 <= scale_factor <= 2.0:
            return 1.0
        elif 0.5 <= scale_factor <= 3.0:
            # Acceptable range
            return 0.7
        else:
            # Extreme scales are suspicious
            return 0.3

    def _calculate_aspect_ratio_score(self, match: TemplateMatch) -> float:
        """
        Score aspect ratios. Real triangles have reasonable width/height ratios.
        """
        x, y, w, h = match.bounding_box

        if h == 0:  # Avoid division by zero
            return 0.1

        aspect_ratio = w / h

        if match.triangle_type == TriangleType.POSSESSION:
            # Possession arrows are typically wider than tall (1.5:1 to 3:1)
            if 1.2 <= aspect_ratio <= 4.0:
                return 1.0
            elif 0.8 <= aspect_ratio <= 5.0:
                return 0.7
            else:
                return 0.3
        else:  # TERRITORY
            # Territory triangles are typically square-ish (0.7:1 to 1.5:1)
            if 0.7 <= aspect_ratio <= 1.5:
                return 1.0
            elif 0.5 <= aspect_ratio <= 2.0:
                return 0.7
            else:
                return 0.3

    def _calculate_position_score(self, match: TemplateMatch, triangle_type: str) -> float:
        """
        Calculate position-based score. Possession triangles should be on left side,
        territory triangles should be on right side of HUD.
        """
        x, y, w, h = match.bounding_box

        # Assume HUD width is around 800-1200 pixels (typical game resolution)
        # This is a rough estimate - in practice we'd get this from YOLO HUD detection
        estimated_hud_width = 1000

        if triangle_type == "possession":
            # Possession triangles should be on the left side (first 40% of HUD)
            if x < estimated_hud_width * 0.4:
                return 1.0  # Perfect position
            elif x < estimated_hud_width * 0.6:
                return 0.5  # Acceptable position
            else:
                return 0.1  # Wrong side

        elif triangle_type == "territory":
            # Territory triangles should be on the right side (last 40% of HUD)
            if x > estimated_hud_width * 0.6:
                return 1.0  # Perfect position
            elif x > estimated_hud_width * 0.4:
                return 0.5  # Acceptable position
            else:
                return 0.1  # Wrong side

        return 0.5  # Default score

    def calculate_overlap(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0

    def create_default_templates(self):
        """Create default triangle templates programmatically."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Create possession arrow templates (pointing left and right)
        self.create_arrow_template("possession_left", Direction.LEFT, TriangleType.POSSESSION)
        self.create_arrow_template("possession_right", Direction.RIGHT, TriangleType.POSSESSION)

        # Create territory triangle templates (pointing up and down)
        self.create_triangle_template("territory_up", Direction.UP, TriangleType.TERRITORY)
        self.create_triangle_template("territory_down", Direction.DOWN, TriangleType.TERRITORY)

        print(f"✅ Created default templates in {self.templates_dir}")

    def create_arrow_template(self, name: str, direction: Direction, triangle_type: TriangleType):
        """Create a possession arrow template."""
        # Create a 20x12 arrow template
        template = np.zeros((12, 20), dtype=np.uint8)

        if direction == Direction.LEFT:
            # Left-pointing arrow: ◄
            points = np.array([[15, 2], [5, 6], [15, 10]], dtype=np.int32)
        else:  # RIGHT
            # Right-pointing arrow: ►
            points = np.array([[5, 2], [15, 6], [5, 10]], dtype=np.int32)

        cv2.fillPoly(template, [points], 255)

        # Save template and metadata
        self.save_template(name, template, direction, triangle_type)

    def create_triangle_template(
        self, name: str, direction: Direction, triangle_type: TriangleType
    ):
        """Create a territory triangle template."""
        # Create a 12x12 triangle template
        template = np.zeros((12, 12), dtype=np.uint8)

        if direction == Direction.UP:
            # Up-pointing triangle: ▲
            points = np.array([[6, 2], [2, 10], [10, 10]], dtype=np.int32)
        else:  # DOWN
            # Down-pointing triangle: ▼
            points = np.array([[6, 10], [2, 2], [10, 2]], dtype=np.int32)

        cv2.fillPoly(template, [points], 255)

        # Save template and metadata
        self.save_template(name, template, direction, triangle_type)

    def save_template(
        self, name: str, template: np.ndarray, direction: Direction, triangle_type: TriangleType
    ):
        """Save a template with its metadata."""
        # Save image
        template_path = self.templates_dir / f"{name}.png"
        cv2.imwrite(str(template_path), template)

        # Save metadata
        metadata = {
            "name": name,
            "direction": direction.value,
            "triangle_type": triangle_type.value,
            "size": template.shape,
            "created_programmatically": True,
        }

        metadata_path = self.templates_dir / f"{name}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Store in memory
        self.templates[name] = {"image": template}
        self.template_metadata[name] = metadata

    def load_templates(self):
        """Load all templates from the templates directory."""
        if not self.templates_dir.exists():
            return

        for template_file in self.templates_dir.glob("*.png"):
            name = template_file.stem
            metadata_file = self.templates_dir / f"{name}.json"

            # Load template image
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            # Load metadata
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Convert integer triangle_type to string format for compatibility
                if "triangle_type" in metadata:
                    if metadata["triangle_type"] == 1:
                        metadata["triangle_type"] = "possession"
                    elif metadata["triangle_type"] == 2:
                        metadata["triangle_type"] = "territory"

            self.templates[name] = {"image": template}
            self.template_metadata[name] = metadata

    def create_template_debug_visualization(self, frame: np.ndarray, matches: list[TemplateMatch]):
        """Create debug visualization showing template matches."""
        debug_img = frame.copy()

        # Draw matches
        for match in matches:
            x, y, w, h = match.bounding_box

            # Color based on triangle type
            if match.triangle_type == TriangleType.POSSESSION:
                color = (0, 255, 0)  # Green for possession
            else:
                color = (0, 0, 255)  # Red for territory

            # Draw bounding box
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)

            # Draw confidence and scale info
            label = f"{match.template_name} {match.confidence:.2f} (x{match.scale_factor:.1f})"
            cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save debug image
        debug_path = self.debug_output_dir / "yolo_template_matches_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)

        print(f"📁 YOLO-integrated template matching debug saved: {debug_path}")

    def add_custom_template_from_roi(
        self,
        hud_img: np.ndarray,
        roi: tuple[int, int, int, int],
        name: str,
        direction: Direction,
        triangle_type: TriangleType,
    ):
        """
        Extract a custom template from a specific ROI in the HUD.
        Use this to create templates from actual game footage.
        """
        x, y, w, h = roi
        template = cv2.cvtColor(hud_img[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

        # Save the custom template
        self.save_template(name, template, direction, triangle_type)

        print(f"✅ Added custom template '{name}' from ROI {roi}")
