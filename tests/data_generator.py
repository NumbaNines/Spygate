"""Test data generation utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spygate.utils.tracking_hardware import TrackingMode
from spygate.video.object_tracker import MultiObjectTracker, ObjectTracker
from tests.utils import BoundingBox, Color, Frame, Position


class TestDataGenerator:
    """Generate test data for tracking scenarios."""

    def __init__(self, output_dir: Path):
        """Initialize test data generator.

        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_linear_motion_dataset(
        self,
        num_sequences: int = 10,
        frames_per_sequence: int = 30,
        frame_size: tuple[int, int] = (640, 480),
        object_size: tuple[int, int] = (50, 50),
        velocity_range: tuple[float, float] = (-10.0, 10.0),
        noise_level: float = 0.05,
    ) -> dict[str, list[dict]]:
        """Generate dataset with objects moving in linear paths.

        Args:
            num_sequences: Number of sequences to generate
            frames_per_sequence: Frames per sequence
            frame_size: Size of frames (width, height)
            object_size: Size of objects (width, height)
            velocity_range: Range for random velocities
            noise_level: Amount of noise to add

        Returns:
            Dictionary with sequences and ground truth
        """
        dataset = {"sequences": [], "ground_truth": []}

        for seq_idx in range(num_sequences):
            # Random starting position
            start_x = np.random.randint(0, frame_size[0] - object_size[0])
            start_y = np.random.randint(0, frame_size[1] - object_size[1])

            # Random velocity
            vx = np.random.uniform(*velocity_range)
            vy = np.random.uniform(*velocity_range)

            sequence = []
            ground_truth = []

            for frame_idx in range(frames_per_sequence):
                # Calculate position
                x = int(start_x + vx * frame_idx)
                y = int(start_y + vy * frame_idx)

                # Add noise
                x += int(np.random.normal(0, noise_level * frame_size[0]))
                y += int(np.random.normal(0, noise_level * frame_size[1]))

                # Create frame
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                cv2.rectangle(
                    frame, (x, y), (x + object_size[0], y + object_size[1]), (255, 255, 255), -1
                )

                # Save frame
                frame_path = self.output_dir / f"seq_{seq_idx:03d}_frame_{frame_idx:03d}.png"
                cv2.imwrite(str(frame_path), frame)

                sequence.append(str(frame_path))
                ground_truth.append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": [x, y, object_size[0], object_size[1]],
                        "velocity": [float(vx), float(vy)],
                    }
                )

            dataset["sequences"].append(sequence)
            dataset["ground_truth"].append(ground_truth)

        # Save metadata
        metadata = {
            "num_sequences": num_sequences,
            "frames_per_sequence": frames_per_sequence,
            "frame_size": frame_size,
            "object_size": object_size,
            "velocity_range": velocity_range,
            "noise_level": noise_level,
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return dataset

    def generate_occlusion_dataset(
        self,
        num_sequences: int = 10,
        frames_per_sequence: int = 60,
        frame_size: tuple[int, int] = (640, 480),
        object_size: tuple[int, int] = (50, 50),
    ) -> dict[str, list[dict]]:
        """Generate dataset with occlusion scenarios.

        Args:
            num_sequences: Number of sequences to generate
            frames_per_sequence: Frames per sequence
            frame_size: Size of frames (width, height)
            object_size: Size of objects (width, height)

        Returns:
            Dictionary with sequences and ground truth
        """
        dataset = {"sequences": [], "ground_truth": []}

        for seq_idx in range(num_sequences):
            # Random starting positions
            x1 = np.random.randint(0, frame_size[0] // 3)
            y1 = np.random.randint(0, frame_size[1] - object_size[1])
            x2 = np.random.randint(2 * frame_size[0] // 3, frame_size[0] - object_size[0])
            y2 = np.random.randint(0, frame_size[1] - object_size[1])

            # Collision frame
            collision_frame = frames_per_sequence // 2

            sequence = []
            ground_truth_obj1 = []
            ground_truth_obj2 = []

            for frame_idx in range(frames_per_sequence):
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

                # Calculate positions
                if frame_idx < collision_frame:
                    progress = frame_idx / collision_frame
                    x1_curr = int(x1 + (frame_size[0] // 2 - x1) * progress)
                    x2_curr = int(x2 + (frame_size[0] // 2 - x2) * progress)
                    y1_curr = y1
                    y2_curr = y2
                else:
                    progress = (frame_idx - collision_frame) / (
                        frames_per_sequence - collision_frame
                    )
                    x1_curr = int(frame_size[0] // 2 + (frame_size[0] - x1) * progress)
                    x2_curr = int(frame_size[0] // 2 + (0 - x2) * progress)
                    y1_curr = y1
                    y2_curr = y2

                # Draw objects
                cv2.rectangle(
                    frame,
                    (x1_curr, y1_curr),
                    (x1_curr + object_size[0], y1_curr + object_size[1]),
                    (255, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (x2_curr, y2_curr),
                    (x2_curr + object_size[0], y2_curr + object_size[1]),
                    (0, 255, 0),
                    -1,
                )

                # Save frame
                frame_path = (
                    self.output_dir / f"occlusion_seq_{seq_idx:03d}_frame_{frame_idx:03d}.png"
                )
                cv2.imwrite(str(frame_path), frame)

                sequence.append(str(frame_path))
                ground_truth_obj1.append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": [x1_curr, y1_curr, object_size[0], object_size[1]],
                        "occluded": frame_idx == collision_frame,
                    }
                )
                ground_truth_obj2.append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": [x2_curr, y2_curr, object_size[0], object_size[1]],
                        "occluded": frame_idx == collision_frame,
                    }
                )

            dataset["sequences"].append(sequence)
            dataset["ground_truth"].append(
                {"object1": ground_truth_obj1, "object2": ground_truth_obj2}
            )

        # Save metadata
        metadata = {
            "num_sequences": num_sequences,
            "frames_per_sequence": frames_per_sequence,
            "frame_size": frame_size,
            "object_size": object_size,
            "scenario": "occlusion",
        }

        with open(self.output_dir / "occlusion_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return dataset

    def generate_formation_dataset(
        self,
        num_sequences: int = 5,
        frames_per_sequence: int = 30,
        field_size: tuple[int, int] = (1024, 768),
        formations: list[str] = ["4-4-2", "4-3-3"],
    ) -> dict[str, list[dict]]:
        """Generate dataset with team formations.

        Args:
            num_sequences: Number of sequences to generate
            frames_per_sequence: Frames per sequence
            field_size: Size of the field (width, height)
            formations: List of formations to use

        Returns:
            Dictionary with sequences and ground truth
        """
        dataset = {"sequences": [], "ground_truth": []}

        for seq_idx in range(num_sequences):
            formation = np.random.choice(formations)
            sequence = []
            ground_truth = []

            # Generate base formation
            base_frame = np.zeros((field_size[1], field_size[0], 3), dtype=np.uint8)
            player_positions = []

            # Add players based on formation
            if formation == "4-4-2":
                rows = [4, 4, 2]  # Players in each line
                base_y = field_size[1] // 2
                spacing = field_size[1] // 6

                for line_idx, num_players in enumerate(rows):
                    x = (line_idx + 1) * field_size[0] // 5
                    for player_idx in range(num_players):
                        y = base_y + (player_idx - (num_players - 1) / 2) * spacing
                        player_positions.append((int(x), int(y)))

            elif formation == "4-3-3":
                rows = [4, 3, 3]
                base_y = field_size[1] // 2
                spacing = field_size[1] // 6

                for line_idx, num_players in enumerate(rows):
                    x = (line_idx + 1) * field_size[0] // 5
                    for player_idx in range(num_players):
                        y = base_y + (player_idx - (num_players - 1) / 2) * spacing
                        player_positions.append((int(x), int(y)))

            # Generate frames with slight movement
            for frame_idx in range(frames_per_sequence):
                frame = base_frame.copy()
                frame_positions = []

                for pos in player_positions:
                    # Add small random movement
                    noise_x = np.random.normal(0, 5)
                    noise_y = np.random.normal(0, 5)
                    new_x = int(pos[0] + noise_x)
                    new_y = int(pos[1] + noise_y)

                    # Draw player
                    cv2.circle(frame, (new_x, new_y), 10, (255, 255, 255), -1)
                    frame_positions.append((new_x, new_y))

                # Save frame
                frame_path = (
                    self.output_dir / f"formation_seq_{seq_idx:03d}_frame_{frame_idx:03d}.png"
                )
                cv2.imwrite(str(frame_path), frame)

                sequence.append(str(frame_path))
                ground_truth.append(
                    {
                        "frame_idx": frame_idx,
                        "formation": formation,
                        "player_positions": frame_positions,
                    }
                )

            dataset["sequences"].append(sequence)
            dataset["ground_truth"].append(ground_truth)

        # Save metadata
        metadata = {
            "num_sequences": num_sequences,
            "frames_per_sequence": frames_per_sequence,
            "field_size": field_size,
            "formations": formations,
            "scenario": "formation",
        }

        with open(self.output_dir / "formation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return dataset
