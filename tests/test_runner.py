"""Test runner and result collection utilities."""

import pytest
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import time
import psutil
import os
import sys
from contextlib import contextmanager

from tests.metrics import TrackingMetrics
from tests.report_generator import TestReportGenerator

class TestRunner:
    """Execute tests and collect results."""
    
    def __init__(self, output_dir: Path):
        """Initialize test runner.
        
        Args:
            output_dir: Directory to save test outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_generator = TestReportGenerator(self.output_dir / "reports")
        
    def run_tracking_tests(
        self,
        test_paths: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        collect_coverage: bool = True,
        collect_performance: bool = True,
        baseline_file: Optional[str] = None,
        parallel: bool = True,
        timeout: Optional[int] = None,
        test_metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run tracking system tests.
        
        Args:
            test_paths: List of test paths to run, runs all tests if None
            markers: List of pytest markers to select tests
            collect_coverage: Whether to collect coverage data
            collect_performance: Whether to collect performance metrics
            baseline_file: Path to baseline metrics file for comparison
            parallel: Whether to run tests in parallel
            timeout: Test timeout in seconds
            test_metadata: Optional metadata about the test run
            
        Returns:
            Dictionary containing test results and metrics
        """
        # Prepare pytest arguments
        pytest_args = []
        
        # Add test paths
        if test_paths:
            pytest_args.extend(test_paths)
        else:
            pytest_args.append("tests")
        
        # Add markers
        if markers:
            for marker in markers:
                pytest_args.extend(["-m", marker])
        
        # Add coverage options
        if collect_coverage:
            pytest_args.extend([
                "--cov=spygate",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_html",
                "--cov-report=json:coverage.json"
            ])
        
        # Add performance options
        if collect_performance:
            pytest_args.extend([
                "--benchmark-only",
                "--benchmark-group-by=name",
                "--benchmark-save=benchmark",
                "--benchmark-save-data"
            ])
        
        # Add parallel execution
        if parallel:
            pytest_args.extend(["-n", "auto"])
        
        # Add timeout
        if timeout:
            pytest_args.extend(["--timeout", str(timeout)])
        
        # Run tests and collect results
        with self._collect_system_metrics() as system_metrics:
            start_time = time.time()
            pytest_result = pytest.main(pytest_args)
            end_time = time.time()
        
        # Load coverage data if collected
        coverage_data = None
        if collect_coverage:
            try:
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
            except FileNotFoundError:
                pass
        
        # Load benchmark data if collected
        benchmark_data = None
        if collect_performance:
            try:
                with open(".benchmarks/benchmark.json") as f:
                    benchmark_data = json.load(f)
            except FileNotFoundError:
                pass
        
        # Load baseline data if provided
        baseline_data = None
        if baseline_file:
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
            except FileNotFoundError:
                pass
        
        # Collect all metrics
        metrics = {
            'tracking_metrics': self._collect_tracking_metrics(),
            'performance_metrics': {
                'timing': {
                    'total_time': end_time - start_time,
                    'tests_per_second': len(pytest_args) / (end_time - start_time)
                },
                'system': system_metrics,
                'coverage': coverage_data['totals'] if coverage_data else None,
                'benchmarks': benchmark_data if benchmark_data else None
            }
        }
        
        # Generate report
        self.report_generator.generate_tracking_report(
            tracking_metrics=metrics['tracking_metrics'],
            performance_metrics=metrics['performance_metrics'],
            baseline_metrics=baseline_data,
            test_metadata=test_metadata
        )
        
        return metrics
    
    def run_benchmark_tests(
        self,
        test_paths: Optional[List[str]] = None,
        baseline_file: Optional[str] = None,
        test_metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run benchmark tests.
        
        Args:
            test_paths: List of benchmark test paths to run
            baseline_file: Path to baseline benchmark file
            test_metadata: Optional metadata about the test run
            
        Returns:
            Dictionary containing benchmark results
        """
        # Prepare pytest arguments
        pytest_args = []
        
        # Add test paths
        if test_paths:
            pytest_args.extend(test_paths)
        else:
            pytest_args.append("tests/benchmarks")
        
        # Add benchmark options
        pytest_args.extend([
            "--benchmark-only",
            "--benchmark-group-by=name",
            "--benchmark-save=benchmark",
            "--benchmark-save-data",
            "--benchmark-histogram=benchmark/histogram"
        ])
        
        # Run benchmarks
        with self._collect_system_metrics() as system_metrics:
            start_time = time.time()
            pytest_result = pytest.main(pytest_args)
            end_time = time.time()
        
        # Load benchmark data
        try:
            with open(".benchmarks/benchmark.json") as f:
                benchmark_data = json.load(f)
        except FileNotFoundError:
            benchmark_data = None
        
        # Load baseline data if provided
        baseline_data = None
        if baseline_file:
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
            except FileNotFoundError:
                pass
        
        # Collect results
        results = {
            'benchmark_results': benchmark_data if benchmark_data else {},
            'system_metrics': system_metrics,
            'timing': {
                'total_time': end_time - start_time,
                'tests_per_second': len(pytest_args) / (end_time - start_time)
            }
        }
        
        # Generate report
        self.report_generator.generate_benchmark_report(
            benchmark_results=results['benchmark_results'],
            baseline_results=baseline_data,
            test_metadata=test_metadata
        )
        
        return results
    
    def _collect_tracking_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect tracking-specific metrics from test results.
        
        Returns:
            Dictionary of tracking metrics
        """
        metrics = {}
        
        # Load test results
        try:
            with open(self.output_dir / "tracking_metrics.json") as f:
                metrics = json.load(f)
        except FileNotFoundError:
            pass
        
        return metrics
    
    @contextmanager
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics during test execution.
        
        Returns:
            Dictionary of system metrics
        """
        process = psutil.Process()
        start_cpu_times = process.cpu_times()
        start_memory = process.memory_info()
        
        try:
            yield {}
        finally:
            end_cpu_times = process.cpu_times()
            end_memory = process.memory_info()
            
            metrics = {
                'cpu_user': end_cpu_times.user - start_cpu_times.user,
                'cpu_system': end_cpu_times.system - start_cpu_times.system,
                'memory_rss': end_memory.rss - start_memory.rss,
                'memory_vms': end_memory.vms - start_memory.vms
            }
            
            # Add GPU metrics if available
            try:
                import torch
                if torch.cuda.is_available():
                    metrics.update({
                        'gpu_memory_allocated': torch.cuda.memory_allocated(),
                        'gpu_memory_cached': torch.cuda.memory_cached()
                    })
            except ImportError:
                pass
            
            return metrics 