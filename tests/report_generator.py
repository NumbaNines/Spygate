"""Test report generation utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np

from tests.metrics import TrackingMetrics
from tests.visualization import TestVisualizer

class TestReportGenerator:
    """Generate comprehensive test reports."""
    
    def __init__(self, output_dir: Path):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save report outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = TestVisualizer(self.output_dir / "visualizations")
        
    def generate_tracking_report(
        self,
        tracking_metrics: Dict[str, Dict[str, float]],
        performance_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        test_metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Generate comprehensive tracking test report.
        
        Args:
            tracking_metrics: Dictionary of tracking-related metrics
            performance_metrics: Dictionary of performance metrics
            baseline_metrics: Optional baseline metrics for comparison
            test_metadata: Optional metadata about the test run
        """
        # Create report structure
        report = {
            'metadata': self._generate_metadata(test_metadata),
            'tracking_metrics': tracking_metrics,
            'performance_metrics': performance_metrics,
            'baseline_comparison': self._compare_with_baseline(
                tracking_metrics, performance_metrics, baseline_metrics
            ) if baseline_metrics else None,
            'overall_scores': self._calculate_overall_scores(
                tracking_metrics, performance_metrics
            )
        }
        
        # Generate visualizations
        self._generate_visualizations(report)
        
        # Save report
        self.save_report(report)
        
    def generate_benchmark_report(
        self,
        benchmark_results: Dict[str, Dict[str, float]],
        baseline_results: Optional[Dict[str, Dict[str, float]]] = None,
        test_metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Generate benchmark test report.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            baseline_results: Optional baseline results for comparison
            test_metadata: Optional metadata about the test run
        """
        # Create report structure
        report = {
            'metadata': self._generate_metadata(test_metadata),
            'benchmark_results': benchmark_results,
            'baseline_comparison': self._compare_benchmarks(
                benchmark_results, baseline_results
            ) if baseline_results else None,
            'performance_analysis': self._analyze_performance(benchmark_results)
        }
        
        # Generate visualizations
        self._generate_benchmark_visualizations(report)
        
        # Save report
        self.save_benchmark_report(report)
    
    def _generate_metadata(
        self,
        custom_metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Generate report metadata.
        
        Args:
            custom_metadata: Optional custom metadata to include
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'python_version': self._get_python_version(),
            'platform': self._get_platform_info(),
            'gpu_info': self._get_gpu_info(),
            'test_framework': 'pytest'
        }
        
        if custom_metadata:
            metadata.update(custom_metadata)
            
        return metadata
    
    def _compare_with_baseline(
        self,
        tracking_metrics: Dict[str, Dict[str, float]],
        performance_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare current metrics with baseline.
        
        Args:
            tracking_metrics: Current tracking metrics
            performance_metrics: Current performance metrics
            baseline_metrics: Baseline metrics
            
        Returns:
            Dictionary of comparison results
        """
        comparison = {}
        
        # Compare tracking metrics
        for category, metrics in tracking_metrics.items():
            if category in baseline_metrics:
                comparison[f'{category}_comparison'] = {
                    metric: {
                        'current': value,
                        'baseline': baseline_metrics[category].get(metric, 0.0),
                        'difference': value - baseline_metrics[category].get(metric, 0.0),
                        'percent_change': (
                            (value - baseline_metrics[category].get(metric, 0.0)) /
                            baseline_metrics[category].get(metric, 1.0) * 100
                        )
                    }
                    for metric, value in metrics.items()
                }
        
        # Compare performance metrics
        for category, metrics in performance_metrics.items():
            if category in baseline_metrics:
                comparison[f'{category}_comparison'] = {
                    metric: {
                        'current': value,
                        'baseline': baseline_metrics[category].get(metric, 0.0),
                        'difference': value - baseline_metrics[category].get(metric, 0.0),
                        'percent_change': (
                            (value - baseline_metrics[category].get(metric, 0.0)) /
                            baseline_metrics[category].get(metric, 1.0) * 100
                        )
                    }
                    for metric, value in metrics.items()
                }
        
        return comparison
    
    def _calculate_overall_scores(
        self,
        tracking_metrics: Dict[str, Dict[str, float]],
        performance_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate overall performance scores.
        
        Args:
            tracking_metrics: Tracking metrics
            performance_metrics: Performance metrics
            
        Returns:
            Dictionary of overall scores
        """
        scores = {}
        
        # Calculate tracking score
        tracking_scores = []
        for metrics in tracking_metrics.values():
            tracking_scores.extend(metrics.values())
        scores['tracking_score'] = np.mean(tracking_scores) if tracking_scores else 0.0
        
        # Calculate performance score
        perf_scores = []
        for metrics in performance_metrics.values():
            perf_scores.extend(metrics.values())
        scores['performance_score'] = np.mean(perf_scores) if perf_scores else 0.0
        
        # Calculate overall score
        scores['overall_score'] = np.mean([
            scores['tracking_score'],
            scores['performance_score']
        ])
        
        return scores
    
    def _compare_benchmarks(
        self,
        current_results: Dict[str, Dict[str, float]],
        baseline_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compare benchmark results with baseline.
        
        Args:
            current_results: Current benchmark results
            baseline_results: Baseline benchmark results
            
        Returns:
            Dictionary of comparison results
        """
        comparison = {}
        
        for category, results in current_results.items():
            if category in baseline_results:
                comparison[category] = {
                    metric: {
                        'current': value,
                        'baseline': baseline_results[category].get(metric, 0.0),
                        'difference': value - baseline_results[category].get(metric, 0.0),
                        'percent_change': (
                            (value - baseline_results[category].get(metric, 0.0)) /
                            baseline_results[category].get(metric, 1.0) * 100
                        )
                    }
                    for metric, value in results.items()
                }
        
        return comparison
    
    def _analyze_performance(
        self,
        benchmark_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze benchmark performance results.
        
        Args:
            benchmark_results: Benchmark results
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        for category, results in benchmark_results.items():
            metrics = list(results.values())
            analysis[category] = {
                'mean': np.mean(metrics),
                'std': np.std(metrics),
                'min': np.min(metrics),
                'max': np.max(metrics),
                'median': np.median(metrics)
            }
        
        return analysis
    
    def _generate_visualizations(self, report: Dict) -> None:
        """Generate visualizations for the report.
        
        Args:
            report: Report data
        """
        # Plot tracking metrics
        for category, metrics in report['tracking_metrics'].items():
            if 'accuracy' in category.lower():
                self.visualizer.plot_tracking_accuracy(
                    metrics,
                    title=f"{category} Metrics"
                )
            elif 'position' in category.lower():
                self.visualizer.plot_position_errors(
                    metrics,
                    title=f"{category} Metrics"
                )
            elif 'velocity' in category.lower():
                self.visualizer.plot_velocity_errors(
                    metrics,
                    title=f"{category} Metrics"
                )
            elif 'occlusion' in category.lower():
                self.visualizer.plot_occlusion_metrics(
                    metrics,
                    title=f"{category} Metrics"
                )
            elif 'formation' in category.lower():
                self.visualizer.plot_formation_metrics(
                    metrics,
                    title=f"{category} Metrics"
                )
        
        # Plot performance metrics
        self.visualizer.plot_performance_metrics(
            report['performance_metrics'],
            baseline_metrics=report.get('baseline_comparison'),
            title="Performance Metrics"
        )
        
        # Plot summary
        self.visualizer.plot_summary_report(
            report,
            title="Overall Performance Summary"
        )
    
    def _generate_benchmark_visualizations(self, report: Dict) -> None:
        """Generate visualizations for benchmark report.
        
        Args:
            report: Benchmark report data
        """
        # Plot benchmark results
        self.visualizer.plot_performance_metrics(
            report['benchmark_results'],
            baseline_metrics=report.get('baseline_comparison'),
            title="Benchmark Results"
        )
        
        # Plot performance analysis
        self.visualizer.plot_performance_metrics(
            report['performance_analysis'],
            title="Performance Analysis"
        )
    
    def save_report(
        self,
        report: Dict,
        filename: str = "test_report.json"
    ) -> None:
        """Save report to file.
        
        Args:
            report: Report data
            filename: Output filename
        """
        with open(self.output_dir / filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def save_benchmark_report(
        self,
        report: Dict,
        filename: str = "benchmark_report.json"
    ) -> None:
        """Save benchmark report to file.
        
        Args:
            report: Benchmark report data
            filename: Output filename
        """
        with open(self.output_dir / filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _get_python_version(self) -> str:
        """Get Python version information."""
        import sys
        return sys.version
    
    def _get_platform_info(self) -> str:
        """Get platform information."""
        import platform
        return platform.platform()
    
    def _get_gpu_info(self) -> Optional[str]:
        """Get GPU information if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0)})"
        except ImportError:
            pass
        return None 