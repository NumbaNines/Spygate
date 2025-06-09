"""Documentation generation utilities for test results."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader


class TestDocumentationGenerator:
    """Generate documentation for test results."""

    def __init__(self, output_dir: Path):
        """Initialize documentation generator.

        Args:
            output_dir: Directory to save documentation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up Jinja2 environment
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)), autoescape=True)

        # Create necessary directories
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)

    def generate_tracking_docs(
        self, report: dict[str, Any], format: str = "html", title: str = "Tracking Test Results"
    ) -> None:
        """Generate tracking test documentation.

        Args:
            report: Test report data
            format: Output format ('html' or 'markdown')
            title: Document title
        """
        # Load appropriate template
        template_name = f"tracking_report.{'html' if format == 'html' else 'md'}"
        template = self.env.get_template(template_name)

        # Prepare context
        context = {
            "title": title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report": report,
            "metrics_summary": self._generate_metrics_summary(report),
            "performance_summary": self._generate_performance_summary(report),
            "visualization_paths": self._get_visualization_paths(report),
        }

        # Render template
        output = template.render(**context)

        # Save output
        output_file = self.output_dir / format / f"tracking_report.{format}"
        with open(output_file, "w") as f:
            f.write(output)

    def generate_benchmark_docs(
        self, report: dict[str, Any], format: str = "html", title: str = "Benchmark Results"
    ) -> None:
        """Generate benchmark documentation.

        Args:
            report: Benchmark report data
            format: Output format ('html' or 'markdown')
            title: Document title
        """
        # Load appropriate template
        template_name = f"benchmark_report.{'html' if format == 'html' else 'md'}"
        template = self.env.get_template(template_name)

        # Prepare context
        context = {
            "title": title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report": report,
            "benchmark_summary": self._generate_benchmark_summary(report),
            "visualization_paths": self._get_visualization_paths(report),
        }

        # Render template
        output = template.render(**context)

        # Save output
        output_file = self.output_dir / format / f"benchmark_report.{format}"
        with open(output_file, "w") as f:
            f.write(output)

    def _generate_metrics_summary(self, report: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of tracking metrics.

        Args:
            report: Test report data

        Returns:
            Dictionary containing metrics summary
        """
        summary = {}

        if "tracking_metrics" in report:
            metrics = report["tracking_metrics"]

            # Accuracy metrics
            if "accuracy" in metrics:
                summary["accuracy"] = {
                    "mean": sum(metrics["accuracy"].values()) / len(metrics["accuracy"]),
                    "min": min(metrics["accuracy"].values()),
                    "max": max(metrics["accuracy"].values()),
                }

            # Position metrics
            if "position" in metrics:
                summary["position"] = {
                    "mean_error": metrics["position"].get("mean_error", 0),
                    "rmse": metrics["position"].get("rmse", 0),
                }

            # Velocity metrics
            if "velocity" in metrics:
                summary["velocity"] = {
                    "mean_error": metrics["velocity"].get("mean_error", 0),
                    "rmse": metrics["velocity"].get("rmse", 0),
                }

            # Occlusion metrics
            if "occlusion" in metrics:
                summary["occlusion"] = {
                    "recovery_rate": metrics["occlusion"].get("recovery_rate", 0),
                    "tracking_persistence": metrics["occlusion"].get("tracking_persistence", 0),
                }

            # Formation metrics
            if "formation" in metrics:
                summary["formation"] = {
                    "formation_accuracy": metrics["formation"].get("formation_accuracy", 0),
                    "stability_score": metrics["formation"].get("stability_score", 0),
                }

        return summary

    def _generate_performance_summary(self, report: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of performance metrics.

        Args:
            report: Test report data

        Returns:
            Dictionary containing performance summary
        """
        summary = {}

        if "performance_metrics" in report:
            metrics = report["performance_metrics"]

            # Timing metrics
            if "timing" in metrics:
                summary["timing"] = {
                    "total_time": metrics["timing"].get("total_time", 0),
                    "tests_per_second": metrics["timing"].get("tests_per_second", 0),
                }

            # System metrics
            if "system" in metrics:
                summary["system"] = {
                    "cpu_usage": metrics["system"].get("cpu_user", 0)
                    + metrics["system"].get("cpu_system", 0),
                    "memory_usage": metrics["system"].get("memory_rss", 0) / (1024 * 1024),  # MB
                }

            # Coverage metrics
            if "coverage" in metrics and metrics["coverage"]:
                summary["coverage"] = {
                    "line_rate": metrics["coverage"].get("line_rate", 0),
                    "branch_rate": metrics["coverage"].get("branch_rate", 0),
                }

        return summary

    def _generate_benchmark_summary(self, report: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of benchmark results.

        Args:
            report: Benchmark report data

        Returns:
            Dictionary containing benchmark summary
        """
        summary = {}

        if "benchmark_results" in report:
            results = report["benchmark_results"]

            # Calculate overall statistics
            all_times = []
            for benchmark in results.values():
                if isinstance(benchmark, dict) and "stats" in benchmark:
                    all_times.extend(benchmark["stats"].get("times", []))

            if all_times:
                summary["overall"] = {
                    "mean": sum(all_times) / len(all_times),
                    "min": min(all_times),
                    "max": max(all_times),
                    "total_benchmarks": len(results),
                }

            # Individual benchmark summaries
            summary["benchmarks"] = {}
            for name, data in results.items():
                if isinstance(data, dict) and "stats" in data:
                    stats = data["stats"]
                    summary["benchmarks"][name] = {
                        "mean": stats.get("mean", 0),
                        "stddev": stats.get("stddev", 0),
                        "iterations": stats.get("iterations", 0),
                    }

        return summary

    def _get_visualization_paths(self, report: dict[str, Any]) -> dict[str, str]:
        """Get paths to visualization files.

        Args:
            report: Report data

        Returns:
            Dictionary mapping visualization names to file paths
        """
        paths = {}

        # Tracking visualizations
        if "tracking_metrics" in report:
            paths.update(
                {
                    "tracking_accuracy": "images/tracking_accuracy.png",
                    "position_errors": "images/position_errors.png",
                    "velocity_errors": "images/velocity_errors.png",
                    "occlusion_metrics": "images/occlusion_metrics.png",
                    "formation_metrics": "images/formation_metrics.png",
                }
            )

        # Performance visualizations
        if "performance_metrics" in report:
            paths["performance_metrics"] = "images/performance_metrics.png"

        # Benchmark visualizations
        if "benchmark_results" in report:
            paths["benchmark_results"] = "images/benchmark_results.png"

        return paths
