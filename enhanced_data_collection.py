"""
Enhanced Madden OCR Data Collection
Extracts comprehensive training data from videos and existing debug regions.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path("src").absolute()))


class EnhancedDataCollector:
    """Collect comprehensive OCR training data from all available sources"""

    def __init__(self, output_dir: str = "madden_ocr_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "raw_extractions").mkdir(exist_ok=True)

        self.db_path = self.output_dir / "ocr_dataset.db"
        self.init_database()

        # Sample counter for unique naming
        self.sample_counter = 0

    def init_database(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                text_content TEXT,
                region_type TEXT,
                confidence REAL,
                source TEXT,
                frame_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def collect_from_video_analysis(self, video_path: str, max_samples: int = 200) -> int:
        """Extract training samples by analyzing video with existing system"""
        print(f"🎥 Analyzing video: {video_path}")

        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return 0

        try:
            from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer

            analyzer = EnhancedGameAnalyzer()
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = max(1, total_frames // max_samples)

            samples_created = 0
            frame_count = 0

            print(f"📊 Processing {total_frames} frames, sampling every {sample_interval} frames")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_interval == 0:
                    try:
                        # Analyze frame
                        game_state = analyzer.analyze_frame(frame, frame_number=frame_count)

                        # Extract regions and save as training samples
                        if hasattr(game_state, "visualization_layers"):
                            layers = game_state.visualization_layers

                            # Process each detected region
                            for region_name, region_image in layers.items():
                                if region_name in [
                                    "down_distance_area",
                                    "game_clock_area",
                                    "play_clock_area",
                                ]:
                                    region_type = region_name.replace("_area", "")

                                    # Save region image
                                    sample_path = self.save_region_sample(
                                        region_image,
                                        f"video_f{frame_count}_{region_type}",
                                        region_type,
                                    )

                                    if sample_path:
                                        # Get OCR result as ground truth
                                        text_content = self.extract_current_ocr(
                                            region_image, region_type
                                        )

                                        if text_content and text_content != "UNKNOWN":
                                            self.add_to_database(
                                                sample_path,
                                                text_content,
                                                region_type,
                                                source=f"video_analysis_{Path(video_path).name}",
                                                frame_number=frame_count,
                                            )
                                            samples_created += 1

                        if samples_created % 10 == 0 and samples_created > 0:
                            print(
                                f"  📈 Created {samples_created} samples from {frame_count} frames"
                            )

                    except Exception as e:
                        print(f"  ⚠️  Error processing frame {frame_count}: {e}")

                frame_count += 1

                if samples_created >= max_samples:
                    break

            cap.release()
            print(f"✅ Video analysis complete: {samples_created} samples created")
            return samples_created

        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
            return 0

    def extract_current_ocr(self, region_image: np.ndarray, region_type: str) -> str:
        """Extract text using current OCR system"""
        try:
            from spygate.ml.enhanced_ocr import EnhancedOCR

            ocr = EnhancedOCR()
            result = ocr.extract_text(region_image)

            if result and result.get("text"):
                text = result["text"].strip()

                # Apply basic validation
                if region_type == "down_distance":
                    if any(pattern in text.upper() for pattern in ["ST", "ND", "RD", "TH", "&"]):
                        return text
                elif region_type == "game_clock":
                    if ":" in text or text.replace(":", "").isdigit():
                        return text
                elif region_type == "play_clock":
                    if text.isdigit() and 1 <= int(text) <= 40:
                        return text

                return text if len(text) > 0 else "UNKNOWN"

        except Exception as e:
            print(f"  ⚠️  OCR extraction failed: {e}")

        return "UNKNOWN"

    def save_region_sample(self, image: np.ndarray, filename: str, region_type: str) -> str:
        """Save region image as training sample"""
        try:
            # Create region directory
            region_dir = self.output_dir / "images" / region_type
            region_dir.mkdir(exist_ok=True)

            # Generate unique filename
            self.sample_counter += 1
            save_path = region_dir / f"{self.sample_counter:06d}_{filename}.png"

            # Ensure image is valid
            if image is not None and image.size > 0:
                cv2.imwrite(str(save_path), image)
                return str(save_path)

        except Exception as e:
            print(f"  ⚠️  Failed to save {filename}: {e}")

        return None

    def add_to_database(
        self,
        image_path: str,
        text_content: str,
        region_type: str,
        source: str = "unknown",
        frame_number: int = None,
    ):
        """Add sample to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO ocr_samples
                (image_path, text_content, region_type, confidence, source, frame_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (image_path, text_content, region_type, 0.8, source, frame_number),
            )

            conn.commit()

        except sqlite3.IntegrityError:
            pass  # Skip duplicates

        conn.close()

    def create_synthetic_samples(self, base_samples: int = 50) -> int:
        """Create synthetic training samples with known text"""
        print(f"🎨 Creating {base_samples} synthetic training samples")

        synthetic_data = {
            "down_distance": [
                "1ST & 10",
                "2ND & 10",
                "3RD & 10",
                "4TH & 10",
                "1ST & 5",
                "2ND & 7",
                "3RD & 3",
                "4TH & 1",
                "2ND & 15",
                "3RD & 8",
                "1ST & 15",
                "4TH & 2",
            ],
            "game_clock": [
                "15:00",
                "12:34",
                "8:45",
                "3:21",
                "0:58",
                "14:59",
                "11:11",
                "7:30",
                "2:00",
                "0:15",
            ],
            "play_clock": ["40", "35", "30", "25", "20", "15", "10", "5", "3", "1"],
        }

        samples_created = 0

        for region_type, texts in synthetic_data.items():
            region_dir = self.output_dir / "images" / region_type
            region_dir.mkdir(exist_ok=True)

            for i, text in enumerate(texts):
                # Create simple synthetic image with text
                synthetic_image = self.create_text_image(text, region_type)

                if synthetic_image is not None:
                    self.sample_counter += 1
                    save_path = (
                        region_dir / f"{self.sample_counter:06d}_synthetic_{region_type}_{i}.png"
                    )

                    cv2.imwrite(str(save_path), synthetic_image)

                    self.add_to_database(
                        str(save_path), text, region_type, source="synthetic_generation"
                    )

                    samples_created += 1

        print(f"✅ Created {samples_created} synthetic samples")
        return samples_created

    def create_text_image(self, text: str, region_type: str) -> np.ndarray:
        """Create synthetic image with text"""
        try:
            # Create base image (similar to Madden HUD colors)
            if region_type == "down_distance":
                width, height = 120, 30
                bg_color = (20, 25, 35)  # Dark blue-gray
                text_color = (255, 255, 255)  # White
            elif region_type == "game_clock":
                width, height = 80, 25
                bg_color = (15, 20, 30)
                text_color = (255, 255, 255)
            else:  # play_clock
                width, height = 40, 25
                bg_color = (25, 30, 40)
                text_color = (255, 255, 255)

            # Create image
            image = np.full((height, width, 3), bg_color, dtype=np.uint8)

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Center text
            x = (width - text_width) // 2
            y = (height + text_height) // 2

            cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

            return image

        except Exception as e:
            print(f"  ⚠️  Failed to create synthetic image for '{text}': {e}")
            return None

    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT region_type, COUNT(*), source
            FROM ocr_samples
            GROUP BY region_type, source
        """
        )

        stats = {}
        total = 0

        for region_type, count, source in cursor.fetchall():
            if region_type not in stats:
                stats[region_type] = {"total": 0, "sources": {}}

            stats[region_type]["sources"][source] = count
            stats[region_type]["total"] += count
            total += count

        stats["grand_total"] = total

        conn.close()
        return stats

    def run_comprehensive_collection(self, video_path: str = None) -> Dict:
        """Run complete data collection process"""
        print("🔥 STARTING COMPREHENSIVE MADDEN OCR DATA COLLECTION")

        total_samples = 0

        # Step 1: Collect existing debug data
        print("\n📁 Collecting existing debug data...")
        from create_madden_ocr_dataset import MaddenOCRDatasetCreator

        creator = MaddenOCRDatasetCreator(str(self.output_dir))
        debug_samples = creator.collect_debug_data()
        total_samples += debug_samples
        print(f"Collected {debug_samples} samples from debug data")

        # Step 2: Create synthetic samples
        print("\n🎨 Creating synthetic training samples...")
        synthetic_samples = self.create_synthetic_samples(100)
        total_samples += synthetic_samples

        # Step 3: Process video if provided
        if video_path and os.path.exists(video_path):
            print(f"\n🎥 Processing video: {video_path}")
            video_samples = self.collect_from_video_analysis(video_path, 150)
            total_samples += video_samples

        # Final statistics
        stats = self.get_dataset_stats()

        print(f"\n📊 COLLECTION COMPLETE!")
        print(f"Total samples: {total_samples}")
        print("\nBreakdown by region:")
        for region, info in stats.items():
            if region != "grand_total":
                print(f"  {region}: {info['total']} samples")
                for source, count in info["sources"].items():
                    print(f"    - {source}: {count}")

        return stats


def main():
    """Main execution"""
    collector = EnhancedDataCollector()

    # Look for video files to process
    video_files = [
        "C:/Users/Nines/Downloads/$1000 1v1me Madden 25 League FINALS Vs CleffTheGod.mp4"
    ]

    # Use first available video
    video_path = None
    for video in video_files:
        if os.path.exists(video):
            video_path = video
            break

    # Run collection
    stats = collector.run_comprehensive_collection(video_path)

    print(f"\n🎯 READY FOR TRAINING!")
    print(f"Dataset contains {stats['grand_total']} total samples")


if __name__ == "__main__":
    main()
