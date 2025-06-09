"""
YOLOv8 Performance Benchmarking Suite
=====================================

This script provides comprehensive benchmarking for YOLOv8 performance,
focusing on CPU optimization and measuring against target benchmarks:
- 1.170s for random images
- 0.038s for 1080x1920 demo images

Author: Expert Developer
Task: 19.3 - YOLOv8 Performance Benchmarking
"""

import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv8PerformanceBenchmark:
    """Comprehensive performance benchmarking for YOLOv8."""

    def __init__(self):
        """Initialize the benchmark suite."""
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "benchmarks": {},
            "target_benchmarks": {
                "random_image_target": 1.170,  # seconds
                "demo_image_target": 0.038,  # seconds
            },
        }
        self.models = {}

    def collect_system_info(self) -> dict:
        """Collect detailed system information."""
        logger.info("Collecting system information...")

        info = {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "opencv_version": cv2.__version__,
        }

        if torch.cuda.is_available():
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]

        self.results["system_info"] = info

        # Log system info
        logger.info(
            f"CPU Cores: {info['cpu_count']} logical, {info['cpu_count_physical']} physical"
        )
        logger.info(
            f"RAM: {info['memory_total_gb']}GB total, {info['memory_available_gb']}GB available"
        )
        logger.info(f"PyTorch: {info['pytorch_version']}")
        logger.info(f"CUDA Available: {info['cuda_available']}")
        logger.info(f"OpenCV: {info['opencv_version']}")

        return info

    def load_models(self) -> bool:
        """Load YOLOv8 models for testing."""
        logger.info("Loading YOLOv8 models...")

        model_files = [("yolov8n.pt", "YOLOv8 Nano"), ("yolov8m.pt", "YOLOv8 Medium")]

        for model_file, model_name in model_files:
            model_path = Path(model_file)
            if model_path.exists():
                try:
                    logger.info(f"Loading {model_name} from {model_file}...")
                    model = YOLO(str(model_path))
                    self.models[model_file] = {
                        "model": model,
                        "name": model_name,
                        "size_mb": round(model_path.stat().st_size / (1024**2), 2),
                    }
                    logger.info(
                        f"✅ {model_name} loaded successfully ({self.models[model_file]['size_mb']}MB)"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name}: {e}")
            else:
                logger.warning(f"⚠️ {model_file} not found, skipping")

        if not self.models:
            logger.error("❌ No models loaded successfully")
            return False

        return True

    def create_test_images(self) -> dict[str, np.ndarray]:
        """Create various test images for benchmarking."""
        logger.info("Creating test images...")

        images = {}

        # Random test image (640x640) - standard YOLO input
        images["random_640x640"] = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # High resolution test image (1080x1920) - mobile/vertical format
        images["random_1080x1920"] = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

        # Standard HD test image (1920x1080)
        images["random_1920x1080"] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Small test image (320x320) - fast inference
        images["random_320x320"] = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        # Load demo frame if available
        demo_path = Path("demo_frame.jpg")
        if demo_path.exists():
            demo_image = cv2.imread(str(demo_path))
            if demo_image is not None:
                images["demo_frame"] = demo_image
                logger.info(f"✅ Demo frame loaded: {demo_image.shape}")

        logger.info(f"Created {len(images)} test images")
        return images

    def run_single_benchmark(
        self, model_info: dict, image_name: str, image: np.ndarray, num_runs: int = 10
    ) -> dict:
        """Run benchmark for a single model-image combination."""
        logger.info(f"Benchmarking {model_info['name']} on {image_name} ({num_runs} runs)...")

        model = model_info["model"]
        times = []

        # Warmup run
        _ = model(image, verbose=False)

        # Benchmark runs
        for i in range(num_runs):
            start_time = time.perf_counter()
            results = model(image, verbose=False)
            end_time = time.perf_counter()

            inference_time = end_time - start_time
            times.append(inference_time)

            if (i + 1) % 5 == 0:
                logger.info(f"  Completed {i + 1}/{num_runs} runs")

        # Calculate statistics
        benchmark_results = {
            "times": times,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "num_runs": num_runs,
            "image_shape": image.shape,
            "model_name": model_info["name"],
            "model_size_mb": model_info["size_mb"],
        }

        logger.info(
            f"✅ {model_info['name']} on {image_name}: {benchmark_results['mean']:.3f}s ± {benchmark_results['std']:.3f}s"
        )

        return benchmark_results

    def analyze_cpu_optimization(self) -> dict:
        """Analyze CPU optimization opportunities."""
        logger.info("Analyzing CPU optimization...")

        optimization_info = {
            "cpu_cores_available": psutil.cpu_count(logical=True),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_usage_before": psutil.cpu_percent(interval=1),
            "memory_usage_before": psutil.virtual_memory().percent,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }

        # Test different thread configurations
        thread_configs = [1, 2, 4, optimization_info["cpu_cores_physical"]]
        if optimization_info["cpu_cores_available"] not in thread_configs:
            thread_configs.append(optimization_info["cpu_cores_available"])

        thread_results = {}

        if self.models:
            model_key = list(self.models.keys())[0]
            model = self.models[model_key]["model"]
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            for num_threads in thread_configs:
                logger.info(f"Testing with {num_threads} threads...")

                original_threads = torch.get_num_threads()
                torch.set_num_threads(num_threads)

                times = []
                for _ in range(5):
                    start_time = time.perf_counter()
                    _ = model(test_image, verbose=False)
                    times.append(time.perf_counter() - start_time)

                thread_results[num_threads] = {
                    "mean_time": statistics.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }

                # Restore original thread count
                torch.set_num_threads(original_threads)

                logger.info(
                    f"  {num_threads} threads: {thread_results[num_threads]['mean_time']:.3f}s"
                )

        optimization_info["thread_performance"] = thread_results

        # Find optimal thread count
        if thread_results:
            optimal_threads = min(
                thread_results.keys(), key=lambda x: thread_results[x]["mean_time"]
            )
            optimization_info["optimal_threads"] = optimal_threads
            logger.info(f"✅ Optimal thread count: {optimal_threads}")

        return optimization_info

    def compare_against_targets(self) -> dict:
        """Compare results against target benchmarks."""
        logger.info("Comparing against target benchmarks...")

        comparisons = {}

        # Find relevant benchmark results
        for model_key, model_info in self.models.items():
            model_results = self.results["benchmarks"].get(model_key, {})

            # Check random image performance
            for image_name, result in model_results.items():
                if "random" in image_name and "640" in image_name:
                    target = self.results["target_benchmarks"]["random_image_target"]
                    actual = result["mean"]
                    ratio = actual / target

                    comparisons[f"{model_info['name']}_random_vs_target"] = {
                        "target": target,
                        "actual": actual,
                        "ratio": ratio,
                        "performance": "BETTER" if ratio < 1.0 else "WORSE",
                        "difference_percent": (ratio - 1.0) * 100,
                    }

                # Check demo image performance (if available)
                elif image_name == "demo_frame":
                    target = self.results["target_benchmarks"]["demo_image_target"]
                    actual = result["mean"]
                    ratio = actual / target

                    comparisons[f"{model_info['name']}_demo_vs_target"] = {
                        "target": target,
                        "actual": actual,
                        "ratio": ratio,
                        "performance": "BETTER" if ratio < 1.0 else "WORSE",
                        "difference_percent": (ratio - 1.0) * 100,
                    }

        return comparisons

    def run_comprehensive_benchmark(self) -> dict:
        """Run the complete benchmark suite."""
        logger.info("=" * 60)
        logger.info("YOLOV8 PERFORMANCE BENCHMARKING SUITE")
        logger.info("=" * 60)

        # Collect system information
        self.collect_system_info()

        # Load models
        if not self.load_models():
            return self.results

        # Create test images
        test_images = self.create_test_images()

        # Run benchmarks for each model-image combination
        for model_key, model_info in self.models.items():
            logger.info(f"\n--- Benchmarking {model_info['name']} ---")

            model_results = {}
            for image_name, image in test_images.items():
                try:
                    # Adjust number of runs based on expected performance
                    num_runs = 20 if "demo_frame" in image_name else 10

                    benchmark_result = self.run_single_benchmark(
                        model_info, image_name, image, num_runs
                    )
                    model_results[image_name] = benchmark_result

                except Exception as e:
                    logger.error(
                        f"❌ Benchmark failed for {model_info['name']} on {image_name}: {e}"
                    )

            self.results["benchmarks"][model_key] = model_results

        # Analyze CPU optimization
        self.results["cpu_optimization"] = self.analyze_cpu_optimization()

        # Compare against targets
        self.results["target_comparison"] = self.compare_against_targets()

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self):
        """Generate benchmark summary."""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)

        # System summary
        sys_info = self.results["system_info"]
        logger.info(f"System: {sys_info['cpu_count']} cores, {sys_info['memory_total_gb']}GB RAM")
        logger.info(
            f"CUDA: {'Available' if sys_info['cuda_available'] else 'Not Available (CPU-only)'}"
        )

        # Performance summary
        logger.info("\nPerformance Results:")
        for model_key, model_results in self.results["benchmarks"].items():
            model_name = self.models[model_key]["name"]
            logger.info(f"\n{model_name}:")

            for image_name, result in model_results.items():
                logger.info(f"  {image_name}: {result['mean']:.3f}s ± {result['std']:.3f}s")

        # Target comparison
        if "target_comparison" in self.results:
            logger.info("\nTarget Comparison:")
            for comparison_name, comparison in self.results["target_comparison"].items():
                logger.info(
                    f"  {comparison_name}: {comparison['actual']:.3f}s vs {comparison['target']:.3f}s "
                    f"({comparison['performance']}, {comparison['difference_percent']:+.1f}%)"
                )

        # CPU optimization
        if (
            "cpu_optimization" in self.results
            and "optimal_threads" in self.results["cpu_optimization"]
        ):
            optimal = self.results["cpu_optimization"]["optimal_threads"]
            logger.info(f"\nOptimal CPU threads: {optimal}")

    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yolov8_benchmark_{timestamp}.json"

        results_path = Path(".benchmarks") / filename
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"✅ Benchmark results saved to {results_path}")
        return results_path


def main():
    """Main execution function."""
    try:
        # Create and run benchmark
        benchmark = YOLOv8PerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark()

        # Save results
        results_path = benchmark.save_results()

        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARKING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {results_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Benchmarking failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
