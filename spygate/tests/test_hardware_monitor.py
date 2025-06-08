from unittest.mock import MagicMock, patch

import cv2
import psutil
import pytest

from spygate.utils.hardware_monitor import HardwareMonitor


@pytest.fixture
def mock_system_info():
    """Mock system information for testing."""
    return {
        "cpu_count": 4,
        "cpu_freq": MagicMock(max=3000.0),
        "memory_total": 8 * 1024**3,  # 8GB
        "cuda_enabled": True,
        "cuda_devices": 1,
    }


@patch("psutil.cpu_count")
@patch("psutil.cpu_freq")
@patch("psutil.virtual_memory")
@patch("cv2.cuda.getCudaEnabledDeviceCount")
def test_hardware_monitor_initialization(
    mock_cuda_count,
    mock_virtual_memory,
    mock_cpu_freq,
    mock_cpu_count,
    mock_system_info,
):
    """Test hardware monitor initialization."""
    # Mock system calls
    mock_cpu_count.return_value = mock_system_info["cpu_count"]
    mock_cpu_freq.return_value = mock_system_info["cpu_freq"]
    mock_virtual_memory.return_value = MagicMock(total=mock_system_info["memory_total"])
    mock_cuda_count.return_value = mock_system_info["cuda_devices"]

    # Initialize monitor
    monitor = HardwareMonitor()

    # Verify initialization
    assert monitor.cpu_count == mock_system_info["cpu_count"]
    assert monitor.cpu_freq.max == mock_system_info["cpu_freq"].max
    assert monitor.total_memory == mock_system_info["memory_total"]
    assert monitor.has_cuda == mock_system_info["cuda_enabled"]
    assert monitor.cuda_device_count == mock_system_info["cuda_devices"]

    # Verify performance tier calculation
    assert monitor.get_performance_tier() in ["low", "medium", "high"]


@patch("psutil.cpu_count")
@patch("psutil.cpu_freq")
@patch("psutil.virtual_memory")
@patch("cv2.cuda.getCudaEnabledDeviceCount")
def test_performance_tier_calculation(
    mock_cuda_count, mock_virtual_memory, mock_cpu_freq, mock_cpu_count
):
    """Test performance tier calculation logic."""
    # Test high-end system
    mock_cpu_count.return_value = 8
    mock_cpu_freq.return_value = MagicMock(max=4000.0)
    mock_virtual_memory.return_value = MagicMock(total=32 * 1024**3)  # 32GB
    mock_cuda_count.return_value = 2

    monitor = HardwareMonitor()
    assert monitor.get_performance_tier() == "high"

    # Test medium system
    mock_cpu_count.return_value = 4
    mock_cpu_freq.return_value = MagicMock(max=2500.0)
    mock_virtual_memory.return_value = MagicMock(total=8 * 1024**3)  # 8GB
    mock_cuda_count.return_value = 1

    monitor = HardwareMonitor()
    assert monitor.get_performance_tier() == "medium"

    # Test low-end system
    mock_cpu_count.return_value = 2
    mock_cpu_freq.return_value = MagicMock(max=2000.0)
    mock_virtual_memory.return_value = MagicMock(total=4 * 1024**3)  # 4GB
    mock_cuda_count.return_value = 0

    monitor = HardwareMonitor()
    assert monitor.get_performance_tier() == "low"


@patch("psutil.cpu_percent")
@patch("psutil.virtual_memory")
def test_resource_monitoring(mock_virtual_memory, mock_cpu_percent):
    """Test resource usage monitoring."""
    monitor = HardwareMonitor()

    # Test CPU utilization
    mock_cpu_percent.return_value = 75.5
    assert monitor.get_cpu_utilization() == 75.5

    # Test memory usage
    mock_virtual_memory.return_value = MagicMock(percent=80.0)
    assert monitor.get_memory_usage() == 80.0

    # Test GPU utilization (when not available)
    if not monitor.has_cuda:
        assert monitor.get_gpu_utilization() is None


def test_system_info():
    """Test system information retrieval."""
    monitor = HardwareMonitor()
    info = monitor.get_system_info()

    # Verify required fields
    assert "os" in info
    assert "os_version" in info
    assert "cpu_count" in info
    assert "cpu_freq_max" in info
    assert "total_memory_gb" in info
    assert "has_cuda" in info
    assert "cuda_devices" in info
    assert "performance_tier" in info

    # Verify data types
    assert isinstance(info["cpu_count"], int)
    assert isinstance(info["cpu_freq_max"], float)
    assert isinstance(info["total_memory_gb"], float)
    assert isinstance(info["has_cuda"], bool)
    assert isinstance(info["cuda_devices"], int)
    assert info["performance_tier"] in ["low", "medium", "high"]


@patch("cv2.cuda.getCudaEnabledDeviceCount")
def test_gpu_error_handling(mock_cuda_count):
    """Test graceful handling of GPU detection errors."""
    # Simulate CUDA detection error
    mock_cuda_count.side_effect = Exception("CUDA error")

    # Monitor should initialize without error
    monitor = HardwareMonitor()
    assert not monitor.has_cuda
    assert monitor.cuda_device_count == 0

    # GPU utilization should return None
    assert monitor.get_gpu_utilization() is None
