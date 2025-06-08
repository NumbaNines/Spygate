"""Command-line interface for running tests."""

import click
from pathlib import Path
import json
from typing import List, Optional, Dict, Any
import sys

from tests.test_runner import TestRunner

@click.group()
def cli():
    """Spygate test runner CLI."""
    pass

@cli.command()
@click.option(
    '--test-paths',
    '-t',
    multiple=True,
    help='Test paths to run (runs all tests if not specified)'
)
@click.option(
    '--markers',
    '-m',
    multiple=True,
    help='Pytest markers to select tests'
)
@click.option(
    '--no-coverage',
    is_flag=True,
    help='Disable coverage collection'
)
@click.option(
    '--no-performance',
    is_flag=True,
    help='Disable performance metrics collection'
)
@click.option(
    '--baseline',
    '-b',
    type=click.Path(exists=True),
    help='Path to baseline metrics file'
)
@click.option(
    '--no-parallel',
    is_flag=True,
    help='Disable parallel test execution'
)
@click.option(
    '--timeout',
    type=int,
    help='Test timeout in seconds'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='test_results',
    help='Output directory for test results'
)
@click.option(
    '--metadata',
    '-d',
    type=click.Path(exists=True),
    help='Path to test metadata JSON file'
)
def run_tracking(
    test_paths: List[str],
    markers: List[str],
    no_coverage: bool,
    no_performance: bool,
    baseline: Optional[str],
    no_parallel: bool,
    timeout: Optional[int],
    output_dir: str,
    metadata: Optional[str]
) -> None:
    """Run tracking system tests."""
    # Load metadata if provided
    test_metadata = None
    if metadata:
        with open(metadata) as f:
            test_metadata = json.load(f)
    
    # Initialize test runner
    runner = TestRunner(Path(output_dir))
    
    try:
        # Run tests
        results = runner.run_tracking_tests(
            test_paths=list(test_paths) if test_paths else None,
            markers=list(markers) if markers else None,
            collect_coverage=not no_coverage,
            collect_performance=not no_performance,
            baseline_file=baseline,
            parallel=not no_parallel,
            timeout=timeout,
            test_metadata=test_metadata
        )
        
        # Check for test failures
        if results.get('pytest_result', 0) != 0:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error running tests: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option(
    '--test-paths',
    '-t',
    multiple=True,
    help='Benchmark test paths to run'
)
@click.option(
    '--baseline',
    '-b',
    type=click.Path(exists=True),
    help='Path to baseline benchmark file'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='benchmark_results',
    help='Output directory for benchmark results'
)
@click.option(
    '--metadata',
    '-d',
    type=click.Path(exists=True),
    help='Path to test metadata JSON file'
)
def run_benchmarks(
    test_paths: List[str],
    baseline: Optional[str],
    output_dir: str,
    metadata: Optional[str]
) -> None:
    """Run benchmark tests."""
    # Load metadata if provided
    test_metadata = None
    if metadata:
        with open(metadata) as f:
            test_metadata = json.load(f)
    
    # Initialize test runner
    runner = TestRunner(Path(output_dir))
    
    try:
        # Run benchmarks
        results = runner.run_benchmark_tests(
            test_paths=list(test_paths) if test_paths else None,
            baseline_file=baseline,
            test_metadata=test_metadata
        )
        
        # Check for benchmark failures
        if results.get('pytest_result', 0) != 0:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error running benchmarks: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('report_path', type=click.Path(exists=True))
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output path for visualization'
)
def visualize(report_path: str, output: Optional[str]) -> None:
    """Visualize test results from a report file."""
    try:
        # Load report
        with open(report_path) as f:
            report = json.load(f)
        
        # Create output directory
        output_dir = Path(output) if output else Path('visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        from tests.visualization import TestVisualizer
        visualizer = TestVisualizer(output_dir)
        
        # Generate visualizations based on report type
        if 'tracking_metrics' in report:
            visualizer.plot_tracking_accuracy(
                report['tracking_metrics'].get('accuracy', {}),
                filename='tracking_accuracy.png'
            )
            visualizer.plot_position_errors(
                report['tracking_metrics'].get('position', {}),
                filename='position_errors.png'
            )
            visualizer.plot_velocity_errors(
                report['tracking_metrics'].get('velocity', {}),
                filename='velocity_errors.png'
            )
            visualizer.plot_occlusion_metrics(
                report['tracking_metrics'].get('occlusion', {}),
                filename='occlusion_metrics.png'
            )
            visualizer.plot_formation_metrics(
                report['tracking_metrics'].get('formation', {}),
                filename='formation_metrics.png'
            )
            
            if 'performance_metrics' in report:
                visualizer.plot_performance_metrics(
                    report['performance_metrics'],
                    filename='performance_metrics.png'
                )
                
            visualizer.plot_summary_report(
                report,
                filename='summary_report.png'
            )
            
        elif 'benchmark_results' in report:
            visualizer.plot_performance_metrics(
                report['benchmark_results'],
                filename='benchmark_results.png'
            )
            
        click.echo(f"Visualizations saved to {output_dir}")
            
    except Exception as e:
        click.echo(f"Error visualizing results: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli() 