"""Visualization utilities for test results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json

from tests.metrics import TrackingMetrics

class TestVisualizer:
    """Visualize test results and metrics."""
    
    def __init__(self, output_dir: Path):
        """Initialize test visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_tracking_accuracy(
        self,
        accuracy_metrics: Dict[str, float],
        title: str = "Tracking Accuracy Metrics",
        filename: str = "tracking_accuracy.png"
    ) -> None:
        """Plot tracking accuracy metrics.
        
        Args:
            accuracy_metrics: Dictionary of accuracy metrics
            title: Plot title
            filename: Output filename
        """
        metrics = ['accuracy', 'mean_iou', 'min_iou', 'max_iou']
        values = [accuracy_metrics[m] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title(title)
        plt.ylim(0, 1.1)  # Metrics are typically between 0 and 1
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_position_errors(
        self,
        position_metrics: Dict[str, float],
        title: str = "Position Error Metrics",
        filename: str = "position_errors.png"
    ) -> None:
        """Plot position error metrics.
        
        Args:
            position_metrics: Dictionary of position error metrics
            title: Plot title
            filename: Output filename
        """
        metrics = ['mean_error', 'std_error', 'min_error', 'max_error', 'rmse']
        values = [position_metrics[m] for m in metrics]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title(title)
        plt.ylabel("Error (pixels)")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_velocity_errors(
        self,
        velocity_metrics: Dict[str, float],
        title: str = "Velocity Error Metrics",
        filename: str = "velocity_errors.png"
    ) -> None:
        """Plot velocity error metrics.
        
        Args:
            velocity_metrics: Dictionary of velocity error metrics
            title: Plot title
            filename: Output filename
        """
        # Separate magnitude and direction errors
        magnitude_metrics = {k: v for k, v in velocity_metrics.items() 
                           if 'magnitude' in k}
        direction_metrics = {k: v for k, v in velocity_metrics.items() 
                           if 'direction' in k}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot magnitude errors
        mag_bars = ax1.bar(magnitude_metrics.keys(), magnitude_metrics.values())
        ax1.set_title("Magnitude Errors")
        ax1.set_ylabel("Error (pixels/frame)")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot direction errors
        dir_bars = ax2.bar(direction_metrics.keys(), direction_metrics.values())
        ax2.set_title("Direction Errors")
        ax2.set_ylabel("Error (degrees/radians)")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for ax, bars in [(ax1, mag_bars), (ax2, dir_bars)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_occlusion_metrics(
        self,
        occlusion_metrics: Dict[str, float],
        title: str = "Occlusion Handling Metrics",
        filename: str = "occlusion_metrics.png"
    ) -> None:
        """Plot occlusion handling metrics.
        
        Args:
            occlusion_metrics: Dictionary of occlusion metrics
            title: Plot title
            filename: Output filename
        """
        metrics = list(occlusion_metrics.keys())
        values = list(occlusion_metrics.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.title(title)
        plt.ylim(0, 1.1)  # Metrics are typically between 0 and 1
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_formation_metrics(
        self,
        formation_metrics: Dict[str, float],
        title: str = "Formation Tracking Metrics",
        filename: str = "formation_metrics.png"
    ) -> None:
        """Plot formation tracking metrics.
        
        Args:
            formation_metrics: Dictionary of formation metrics
            title: Plot title
            filename: Output filename
        """
        # Separate error and stability metrics
        error_metrics = {k: v for k, v in formation_metrics.items() 
                        if 'error' in k}
        stability_metrics = {k: v for k, v in formation_metrics.items() 
                           if 'stability' in k}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot error metrics
        error_bars = ax1.bar(error_metrics.keys(), error_metrics.values())
        ax1.set_title("Error Metrics")
        ax1.set_ylabel("Error (pixels/meters)")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot stability metrics
        stability_bars = ax2.bar(stability_metrics.keys(), stability_metrics.values())
        ax2.set_title("Stability Metrics")
        ax2.set_ylabel("Stability Score")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for ax, bars in [(ax1, error_bars), (ax2, stability_bars)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_performance_metrics(
        self,
        performance_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        title: str = "Performance Metrics",
        filename: str = "performance_metrics.png"
    ) -> None:
        """Plot performance metrics with optional baseline comparison.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            baseline_metrics: Optional baseline metrics for comparison
            title: Plot title
            filename: Output filename
        """
        # Group metrics by type
        timing_metrics = {k: v for k, v in performance_metrics.items() 
                        if any(x in k for x in ['time', 'fps'])}
        memory_metrics = {k: v for k, v in performance_metrics.items() 
                         if 'memory' in k}
        gpu_metrics = {k: v for k, v in performance_metrics.items() 
                      if 'gpu' in k}
        
        num_plots = 2 + (len(gpu_metrics) > 0)
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 6))
        
        # Plot timing metrics
        self._plot_metric_group(axes[0], timing_metrics, "Timing Metrics",
                              baseline_metrics)
        
        # Plot memory metrics
        self._plot_metric_group(axes[1], memory_metrics, "Memory Metrics",
                              baseline_metrics)
        
        # Plot GPU metrics if available
        if gpu_metrics:
            self._plot_metric_group(axes[2], gpu_metrics, "GPU Metrics",
                                  baseline_metrics)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def _plot_metric_group(
        self,
        ax: plt.Axes,
        metrics: Dict[str, float],
        title: str,
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Helper method to plot a group of metrics.
        
        Args:
            ax: Matplotlib axes
            metrics: Dictionary of metrics to plot
            title: Plot title
            baseline_metrics: Optional baseline metrics
        """
        x = np.arange(len(metrics))
        width = 0.35
        
        # Plot current metrics
        bars1 = ax.bar(x - width/2 if baseline_metrics else x,
                      list(metrics.values()),
                      width, label='Current')
        
        # Plot baseline metrics if provided
        if baseline_metrics:
            baseline_values = [baseline_metrics.get(k, 0) for k in metrics.keys()]
            bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline')
        
        # Add value labels
        for bars in [bars1] + ([bars2] if baseline_metrics else []):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys(), rotation=45)
        ax.grid(True, alpha=0.3)
        if baseline_metrics:
            ax.legend()
    
    def plot_summary_report(
        self,
        report: Dict[str, Dict[str, float]],
        title: str = "Overall Performance Summary",
        filename: str = "summary_report.png"
    ) -> None:
        """Plot comprehensive summary report.
        
        Args:
            report: Dictionary containing all metrics
            title: Plot title
            filename: Output filename
        """
        # Extract overall scores
        scores = report['overall_scores']
        
        plt.figure(figsize=(10, 6))
        
        # Create spider plot
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Compute angles for spider plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # complete the polygon
        angles = np.concatenate((angles, [angles[0]]))  # complete the polygon
        
        # Create spider plot
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add value labels
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            ha = 'left' if angle < np.pi else 'right'
            ax.text(angle, value, f'{value:.2f}',
                   ha=ha, va='center')
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def save_metrics_report(
        self,
        report: Dict[str, Dict[str, float]],
        filename: str = "metrics_report.json"
    ) -> None:
        """Save metrics report to JSON file.
        
        Args:
            report: Dictionary containing all metrics
            filename: Output filename
        """
        with open(self.output_dir / filename, 'w') as f:
            json.dump(report, f, indent=2) 