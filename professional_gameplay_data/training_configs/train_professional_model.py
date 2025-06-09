#!/usr/bin/env python3
"""
Professional Model Training Script for SpygateAI
=================================================

This script trains YOLOv8 models specifically on professional gameplay footage
with enhanced validation, strategic analysis, and coaching-grade precision.

Usage:
    python train_professional_model.py --config professional_dataset.yaml
    python train_professional_model.py --config professional_dataset.yaml --resume
    python train_professional_model.py --benchmark  # Run benchmarking mode
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from spygate.core.hardware import HardwareDetector
    from spygate.ml.yolov8_model import EnhancedYOLOv8
except ImportError:
    print("Warning: SpygateAI modules not found. Using basic YOLO training.")
    HardwareDetector = None
    EnhancedYOLOv8 = None

# Configure logging for professional training
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f'professional_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ProfessionalModelTrainer:
    """Enhanced trainer for professional-grade models with coaching validation."""

    def __init__(self, config_path: str):
        """Initialize the professional trainer."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.hardware = HardwareDetector() if HardwareDetector else None
        self.start_time = time.time()

        # Professional training standards
        self.professional_standards = {
            "min_map50": 0.99,  # Minimum 99% mAP50
            "min_map95": 0.85,  # Minimum 85% mAP95
            "max_inference_time": 2.0,  # Maximum 2ms inference
            "min_training_samples": 1000,  # Minimum training samples
            "quality_threshold": 0.95,  # Quality gate threshold
        }

    def _load_config(self) -> dict:
        """Load and validate professional training configuration."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            # Validate professional requirements
            required_fields = ["path", "nc", "names", "quality_threshold", "benchmarks"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field in config: {field}")

            logger.info(f"✅ Professional config loaded: {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise

    def _setup_professional_environment(self) -> dict:
        """Setup optimized environment for professional training."""
        setup_info = {
            "timestamp": datetime.now().isoformat(),
            "config_path": str(self.config_path),
            "professional_mode": True,
        }

        if self.hardware:
            # Hardware-specific optimizations
            if self.hardware.has_cuda:
                setup_info["device"] = "cuda"
                setup_info["gpu_name"] = self.hardware.gpu_info["name"]
                setup_info["gpu_memory"] = f"{self.hardware.gpu_info['memory_gb']:.1f}GB"

                # Professional training optimizations
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for performance

            else:
                setup_info["device"] = "cpu"
                setup_info["warning"] = "Training on CPU - performance will be limited"

        # Set professional training environment variables
        os.environ["YOLO_VERBOSE"] = "True"  # Detailed logging
        os.environ["ULTRALYTICS_DATASETS_DIR"] = str(Path(self.config["path"]).parent)

        logger.info("🚀 Professional training environment configured")
        logger.info(f"📊 Setup: {setup_info}")

        return setup_info

    def _validate_dataset_quality(self) -> bool:
        """Validate dataset meets professional standards."""
        dataset_path = Path(self.config["path"])

        # Check dataset structure
        required_dirs = ["images", "labels"]
        for dir_name in required_dirs:
            if not (dataset_path / dir_name).exists():
                logger.error(f"❌ Missing required directory: {dir_name}")
                return False

        # Count samples
        image_files = list((dataset_path / "images").glob("*.png")) + list(
            (dataset_path / "images").glob("*.jpg")
        )
        label_files = list((dataset_path / "labels").glob("*.txt"))

        logger.info(f"📸 Found {len(image_files)} images")
        logger.info(f"🏷️ Found {len(label_files)} labels")

        # Professional quality gates
        if len(image_files) < self.professional_standards["min_training_samples"]:
            logger.error(
                f"❌ Insufficient training samples: {len(image_files)} < {self.professional_standards['min_training_samples']}"
            )
            return False

        if len(image_files) != len(label_files):
            logger.warning(
                f"⚠️ Image/label count mismatch: {len(image_files)} vs {len(label_files)}"
            )

        # Validate class distribution
        class_counts = {i: 0 for i in range(self.config["nc"])}
        for label_file in label_files:
            try:
                with open(label_file) as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1
            except Exception as e:
                logger.warning(f"⚠️ Error reading label file {label_file}: {e}")

        logger.info(f"📈 Class distribution: {class_counts}")

        # Check minimum samples per class for professional training
        min_per_class = self.config.get("validation", {}).get("min_samples_per_class", 50)
        for class_id, count in class_counts.items():
            if count < min_per_class:
                logger.error(
                    f"❌ Insufficient samples for class {class_id}: {count} < {min_per_class}"
                )
                return False

        logger.info("✅ Dataset quality validation passed")
        return True

    def train(self, resume: bool = False) -> dict:
        """Train the professional model with enhanced monitoring."""
        logger.info("🎯 Starting Professional Model Training")
        logger.info("=" * 60)

        # Setup environment
        setup_info = self._setup_professional_environment()

        # Validate dataset
        if not self._validate_dataset_quality():
            raise ValueError("Dataset does not meet professional standards")

        # Initialize model
        if EnhancedYOLOv8:
            # Use SpygateAI enhanced model if available
            model = EnhancedYOLOv8(
                model_size="s",  # Start with YOLOv8s for professional precision
                device="auto",
                precision="professional",
            )
            logger.info("🔧 Using Enhanced SpygateAI model")
        else:
            # Fallback to standard YOLO
            model = YOLO("yolov8s.pt")
            logger.info("🔧 Using standard YOLOv8 model")

        # Professional training parameters
        training_params = self.config.get("training_params", {})
        professional_params = {
            "data": str(self.config_path),
            "epochs": training_params.get("epochs", 50),
            "batch": training_params.get("batch_size", 16),
            "lr0": training_params.get("learning_rate", 0.001),
            "weight_decay": training_params.get("weight_decay", 0.0005),
            "warmup_epochs": training_params.get("warmup_epochs", 5),
            "patience": 10,  # Early stopping patience
            "save_period": 5,  # Save checkpoint every 5 epochs
            "project": "professional_runs",
            "name": f'professional_v{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",  # Professional-grade optimizer
            "verbose": True,
            "seed": 42,  # Reproducible training
            "deterministic": True,  # Deterministic training for professional consistency
            "workers": 8,  # Multi-worker data loading
            "resume": resume,
        }

        # Add hardware-specific optimizations
        if self.hardware and self.hardware.has_cuda:
            professional_params.update(
                {
                    "device": "cuda",
                    "amp": True,  # Automatic mixed precision
                    "half": False,  # Keep full precision for professional training
                }
            )
        else:
            professional_params.update(
                {
                    "device": "cpu",
                    "amp": False,
                    "workers": 4,  # Reduce workers for CPU
                }
            )

        logger.info(f"🏋️ Training parameters: {professional_params}")

        # Start training with professional monitoring
        try:
            logger.info("🚀 Beginning professional model training...")
            results = model.train(**professional_params)

            # Professional validation
            validation_results = self._validate_professional_model(model, results)

            return {
                "setup_info": setup_info,
                "training_params": professional_params,
                "results": results,
                "validation": validation_results,
                "training_time": time.time() - self.start_time,
                "professional_certified": validation_results["meets_professional_standards"],
            }

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def _validate_professional_model(self, model, training_results) -> dict:
        """Validate model meets professional standards."""
        logger.info("🔍 Running Professional Validation")

        # Extract metrics from training results
        try:
            # Get validation metrics from the last epoch
            metrics = (
                training_results.results_dict if hasattr(training_results, "results_dict") else {}
            )

            # Professional benchmarks
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map95 = metrics.get("metrics/mAP50-95(B)", 0.0)

            # Performance validation
            validation_results = {
                "map50": float(map50),
                "map95": float(map95),
                "meets_map50_standard": map50 >= self.professional_standards["min_map50"],
                "meets_map95_standard": map95 >= self.professional_standards["min_map95"],
                "professional_grade": False,
                "recommendations": [],
            }

            # Speed test
            speed_test = self._run_speed_benchmark(model)
            validation_results.update(speed_test)

            # Overall professional certification
            professional_checks = [
                validation_results["meets_map50_standard"],
                validation_results["meets_map95_standard"],
                validation_results["meets_speed_standard"],
            ]

            validation_results["meets_professional_standards"] = all(professional_checks)
            validation_results["professional_grade"] = validation_results[
                "meets_professional_standards"
            ]

            # Generate recommendations
            if not validation_results["meets_map50_standard"]:
                validation_results["recommendations"].append(
                    f"Increase training data or epochs to reach {self.professional_standards['min_map50']:.1%} mAP50"
                )

            if not validation_results["meets_map95_standard"]:
                validation_results["recommendations"].append(
                    f"Improve annotation quality to reach {self.professional_standards['min_map95']:.1%} mAP95"
                )

            if not validation_results["meets_speed_standard"]:
                validation_results["recommendations"].append(
                    f"Optimize model for <{self.professional_standards['max_inference_time']}ms inference"
                )

            # Log results
            if validation_results["professional_grade"]:
                logger.info("🏆 MODEL CERTIFIED: Professional Grade")
            else:
                logger.warning("⚠️ MODEL NEEDS IMPROVEMENT: Below Professional Standards")

            logger.info(
                f"📊 mAP50: {map50:.1%} (target: {self.professional_standards['min_map50']:.1%})"
            )
            logger.info(
                f"📊 mAP95: {map95:.1%} (target: {self.professional_standards['min_map95']:.1%})"
            )
            logger.info(
                f"⚡ Inference: {validation_results['avg_inference_time']:.1f}ms (target: <{self.professional_standards['max_inference_time']}ms)"
            )

            return validation_results

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                "error": str(e),
                "meets_professional_standards": False,
                "professional_grade": False,
                "recommendations": ["Manual validation required due to error"],
            }

    def _run_speed_benchmark(self, model) -> dict:
        """Run inference speed benchmark for professional validation."""
        logger.info("⚡ Running speed benchmark...")

        try:
            # Create dummy input for speed test
            dummy_input = torch.randn(1, 3, 640, 640)
            if self.hardware and self.hardware.has_cuda:
                dummy_input = dummy_input.cuda()

            # Warmup runs
            for _ in range(10):
                _ = model.predict(dummy_input, verbose=False)

            # Benchmark runs
            times = []
            for _ in range(100):
                start_time = time.time()
                _ = model.predict(dummy_input, verbose=False)
                times.append((time.time() - start_time) * 1000)  # Convert to ms

            avg_time = sum(times) / len(times)
            meets_standard = avg_time <= self.professional_standards["max_inference_time"]

            return {
                "avg_inference_time": avg_time,
                "min_inference_time": min(times),
                "max_inference_time": max(times),
                "meets_speed_standard": meets_standard,
            }

        except Exception as e:
            logger.warning(f"⚠️ Speed benchmark failed: {e}")
            return {"avg_inference_time": 999.0, "meets_speed_standard": False, "error": str(e)}

    def benchmark_against_casual_model(self, casual_model_path: str) -> dict:
        """Benchmark professional model against casual model."""
        logger.info("🥊 Benchmarking Professional vs Casual Model")
        # This would implement comparison logic
        # For now, return placeholder
        return {
            "professional_advantage": "TBD",
            "improvement_areas": ["TBD"],
            "coaching_insights": ["TBD"],
        }


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train Professional SpygateAI Model")
    parser.add_argument(
        "--config", default="professional_dataset.yaml", help="Path to professional dataset config"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmarking mode against casual model"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = ProfessionalModelTrainer(args.config)

    if args.benchmark:
        # Benchmarking mode
        results = trainer.benchmark_against_casual_model("../models/yolov8s.pt")
        logger.info(f"🏆 Benchmark Results: {results}")
    else:
        # Training mode
        results = trainer.train(resume=args.resume)

        # Summary
        logger.info("🎯 PROFESSIONAL TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️ Training Time: {results['training_time']:.1f}s")
        logger.info(f"🏆 Professional Grade: {results['validation']['professional_grade']}")

        if results["validation"]["recommendations"]:
            logger.info("📋 Recommendations:")
            for rec in results["validation"]["recommendations"]:
                logger.info(f"  • {rec}")


if __name__ == "__main__":
    main()
