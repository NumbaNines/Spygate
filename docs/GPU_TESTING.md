# GPU Memory Management Testing Suite

## Overview

The SpygateAI GPU Memory Management Testing Suite provides comprehensive validation of GPU acceleration features, memory allocation, cleanup, and performance optimization across the codebase. This suite is designed to ensure robust GPU memory handling in production environments.

## Features

### Core Test Categories

1. **GPU Initialization Testing**

   - Validates GPU detection and initialization
   - Tests hardware tier detection
   - Monitors memory allocation during setup
   - Tracks initialization performance

2. **Memory Usage Analysis**

   - Monitors GPU memory consumption during analysis
   - Detects memory leaks in processing pipelines
   - Validates memory cleanup after operations
   - Tests concurrent GPU usage scenarios

3. **Performance Benchmarking**

   - Compares GPU vs CPU performance
   - Measures speedup factors
   - Validates processing efficiency
   - Tests scalability under load

4. **Error Handling Validation**

   - Tests fallback mechanisms when GPU is unavailable
   - Validates graceful degradation
   - Ensures system stability during GPU failures
   - Tests recovery from GPU errors

5. **Concurrent Usage Testing**

   - Tests multiple simultaneous GPU operations
   - Validates thread safety
   - Tests resource sharing and contention
   - Monitors performance under concurrent load

6. **Memory Cleanup Verification**
   - Validates proper resource deallocation
   - Tests garbage collection efficiency
   - Monitors memory leak prevention
   - Ensures clean system state after operations

## Usage

### Quick Start

Run essential tests quickly:

```bash
python scripts/test_gpu_memory.py --quick
```

### Comprehensive Testing

Run all tests with detailed reporting:

```bash
python scripts/test_gpu_memory.py
```

### Save Results

Save detailed results to a JSON file:

```bash
python scripts/test_gpu_memory.py --save gpu_test_results.json
```

### Verbose Output

Enable detailed logging:

```bash
python scripts/test_gpu_memory.py --verbose
```

### Skip GPU Tests

For CI/CD environments without GPU:

```bash
python scripts/test_gpu_memory.py --no-gpu
```

## Command Line Options

| Option                    | Description                                 |
| ------------------------- | ------------------------------------------- |
| `--quick` / `-q`          | Run essential tests only (faster execution) |
| `--save FILE` / `-s FILE` | Save detailed results to JSON file          |
| `--verbose` / `-v`        | Enable detailed logging output              |
| `--no-gpu`                | Skip GPU-specific tests (useful for CI/CD)  |
| `--help` / `-h`           | Show help message and available options     |

## Test Results Interpretation

### Test Status Indicators

- ✅ **PASSED**: Test completed successfully
- ❌ **FAILED**: Test failed (check error details)
- ⚠️ **SKIPPED**: Test skipped (usually due to missing GPU)

### Key Metrics

#### GPU Initialization

- **Initialization Time**: Time to set up GPU context
- **GPU Enabled**: Whether GPU acceleration is active
- **Memory Usage**: Memory consumed during initialization
- **Hardware Tier**: Detected hardware performance tier

#### Memory Analysis

- **Analysis Count**: Number of operations performed
- **Memory Leak Detection**: Whether memory leaks were detected
- **Peak Memory**: Maximum memory usage observed
- **Memory Variance**: Consistency of memory usage

#### Performance Comparison

- **GPU Time**: Time for GPU-accelerated operations
- **CPU Time**: Time for CPU-only operations
- **Speedup Factor**: Performance improvement (GPU/CPU ratio)

#### Concurrent Testing

- **Concurrent Analyses**: Number of simultaneous operations
- **Thread Usage**: Number of concurrent threads tested
- **Resource Contention**: Evidence of GPU resource conflicts

### Sample Output

```
🧠 SPYGATE AI - GPU MEMORY MANAGEMENT TEST RESULTS
================================================================================

📊 SYSTEM INFORMATION:
   Hardware Tier: ULTRA
   CPU Cores: 16
   Total Memory: 32.0 GB
   GPU Available: ✅ Yes

📈 OVERALL SUMMARY:
   Tests Run: 6
   Passed: 6 ✅
   Failed: 0 ❌
   Success Rate: 100.0%

🔍 DETAILED TEST RESULTS:

1. ✅ FORMATION_ANALYZER_GPU_INIT
   ⏱️  Initialization Time: 0.156s
   🖥️  GPU Enabled: True
   💾 Memory Usage: 245.3 MB

2. ✅ GPU_MEMORY_DURING_ANALYSIS
   🔄 Analyses Performed: 10
   🔍 Memory Leaks: ✅ None
   💾 Peak Memory: 1024.5 MB

3. ✅ GPU_VS_CPU_PERFORMANCE
   🚀 GPU Speedup: 3.2x

4. ✅ GPU_ERROR_HANDLING
   ✅ Fallback Successful

5. ✅ MEMORY_CLEANUP
   💾 Memory Cleaned: ✅ Yes
   🔄 Cleanup Efficiency: 95.2%

6. ✅ CONCURRENT_GPU_USAGE
   ⚡ Concurrent Analyses: 6
   🧵 Threads Used: 3
================================================================================
```

## Integration with pytest

The test suite can also be run using pytest for integration with CI/CD pipelines:

```bash
# Run all GPU memory tests
pytest tests/test_gpu_memory_management.py -v

# Run specific test
pytest tests/test_gpu_memory_management.py::TestGPUMemoryManagement::test_formation_analyzer_gpu_initialization -v

# Skip GPU tests if hardware not available
pytest tests/test_gpu_memory_management.py -v -k "not gpu_memory_during_analysis and not concurrent_gpu_usage"
```

## Troubleshooting

### Common Issues

#### GPU Not Detected

```
⚠️  No GPU detected. Some tests will be skipped.
```

**Solution**:

- Verify NVIDIA GPU drivers are installed
- Check CUDA installation
- Ensure OpenCV was compiled with CUDA support

#### Memory Leaks Detected

```
🔍 Memory Leaks: ⚠️ Detected
```

**Solution**:

- Check for unreleased GPU memory allocations
- Verify proper cleanup in analysis pipelines
- Review garbage collection efficiency

#### Performance Issues

```
🚀 GPU Speedup: 0.8x (slower than CPU)
```

**Solution**:

- Check GPU memory bandwidth
- Verify optimal batch sizes
- Review data transfer overhead
- Consider GPU memory fragmentation

### System Requirements

#### Minimum Requirements

- **GPU**: CUDA-capable NVIDIA GPU
- **Memory**: 4GB system RAM
- **CUDA**: CUDA Toolkit 11.0 or higher
- **OpenCV**: OpenCV compiled with CUDA support

#### Recommended Requirements

- **GPU**: RTX 3060 or higher
- **Memory**: 16GB system RAM
- **CUDA**: CUDA Toolkit 12.0 or higher
- **GPU Memory**: 8GB VRAM or higher

## CI/CD Integration

### GitHub Actions Example

```yaml
name: GPU Memory Tests

on: [push, pull_request]

jobs:
  gpu-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest

      - name: Run GPU memory tests (CPU fallback)
        run: |
          python scripts/test_gpu_memory.py --no-gpu --save ci_results.json

      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: gpu-test-results
          path: ci_results.json
```

### Docker Testing

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy test suite
COPY . /app
WORKDIR /app

# Run tests
CMD ["python3", "scripts/test_gpu_memory.py", "--save", "docker_results.json"]
```

## Extending the Test Suite

### Adding Custom Tests

To add custom GPU memory tests:

1. **Create test method** in `GPUMemoryTestSuite`:

```python
def test_custom_gpu_feature(self) -> Dict:
    """Test custom GPU feature."""
    results = {
        'test_name': 'custom_gpu_feature',
        'passed': False,
        'errors': []
    }

    try:
        # Your test logic here
        results['passed'] = True
    except Exception as e:
        results['errors'].append(str(e))

    return results
```

2. **Add to test runner** in `run_all_tests()`:

```python
test_methods = [
    # ... existing tests
    self.test_custom_gpu_feature,
]
```

3. **Add pytest wrapper** in `TestGPUMemoryManagement`:

```python
def test_custom_gpu_feature(self):
    """Test custom GPU feature."""
    result = self.test_suite.test_custom_gpu_feature()
    assert result['passed'], f"Custom test failed: {result.get('errors', [])}"
```

### Custom Metrics

Add custom metrics to test results:

```python
results.update({
    'custom_metric': your_measurement,
    'performance_score': calculated_score,
    'resource_usage': resource_data
})
```

## Best Practices

### Test Design

- **Isolate tests**: Each test should be independent
- **Clean up resources**: Always reset state between tests
- **Handle failures gracefully**: Provide meaningful error messages
- **Validate thoroughly**: Check all expected outcomes

### Performance Testing

- **Use realistic data**: Test with representative workloads
- **Measure consistently**: Use multiple runs for stable metrics
- **Monitor resources**: Track both CPU and GPU utilization
- **Consider thermal throttling**: Account for sustained load effects

### Memory Testing

- **Monitor continuously**: Track memory throughout operations
- **Test edge cases**: Validate behavior under memory pressure
- **Verify cleanup**: Ensure complete resource deallocation
- **Check for fragmentation**: Monitor memory fragmentation patterns

## Support

For issues with the GPU testing suite:

1. **Check system requirements** and ensure CUDA/OpenCV setup
2. **Review test logs** for specific error messages
3. **Run individual tests** to isolate problems
4. **Use verbose mode** for detailed debugging information
5. **Check GPU drivers** and CUDA installation

## Contributing

When contributing to the GPU testing suite:

1. **Follow existing patterns** for test structure and naming
2. **Add comprehensive documentation** for new tests
3. **Include error handling** for all failure scenarios
4. **Test on multiple hardware configurations** when possible
5. **Update documentation** to reflect any changes
