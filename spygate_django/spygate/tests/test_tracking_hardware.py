"""
Tests for the tracking hardware module.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ..utils.tracking_hardware import TrackingAlgorithm, TrackingHardwareManager, TrackingMode


@pytest.fixture
def mock_hardware_info_basic():
    """Mock hardware info for basic tier."""
    return {
        "cpu_count": 2,
        "total_memory": 4 * 1024**3,  # 4GB
        "gpu_available": False,
        "gpu_memory": 0,
    }


@pytest.fixture
def mock_hardware_info_standard():
    """Mock hardware info for standard tier."""
    return {
        "cpu_count": 4,
        "total_memory": 8 * 1024**3,  # 8GB
        "gpu_available": True,
        "gpu_memory": 4 * 1024**3,  # 4GB VRAM
    }


@pytest.fixture
def mock_hardware_info_advanced():
    """Mock hardware info for advanced tier."""
    return {
        "cpu_count": 6,
        "total_memory": 16 * 1024**3,  # 16GB
        "gpu_available": True,
        "gpu_memory": 8 * 1024**3,  # 8GB VRAM
    }


@pytest.fixture
def mock_hardware_info_professional():
    """Mock hardware info for professional tier."""
    return {
        "cpu_count": 8,
        "total_memory": 32 * 1024**3,  # 32GB
        "gpu_available": True,
        "gpu_memory": 12 * 1024**3,  # 12GB VRAM
    }


def test_tracking_mode_determination_basic(mock_hardware_info_basic):
    """Test tracking mode determination for basic hardware."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = mock_hardware_info_basic
            mock_monitor.return_value.has_gpu_support.return_value = False

            manager = TrackingHardwareManager()
            assert manager.tracking_mode == TrackingMode.BASIC


def test_tracking_mode_determination_standard(mock_hardware_info_standard):
    """Test tracking mode determination for standard hardware."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = mock_hardware_info_standard
            mock_monitor.return_value.has_gpu_support.return_value = True

            manager = TrackingHardwareManager()
            assert manager.tracking_mode == TrackingMode.STANDARD


def test_tracking_mode_determination_advanced(mock_hardware_info_advanced):
    """Test tracking mode determination for advanced hardware."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = mock_hardware_info_advanced
            mock_monitor.return_value.has_gpu_support.return_value = True

            manager = TrackingHardwareManager()
            assert manager.tracking_mode == TrackingMode.ADVANCED


def test_tracking_mode_determination_professional(mock_hardware_info_professional):
    """Test tracking mode determination for professional hardware."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = mock_hardware_info_professional
            mock_monitor.return_value.has_gpu_support.return_value = True

            manager = TrackingHardwareManager()
            assert manager.tracking_mode == TrackingMode.PROFESSIONAL


def test_available_algorithms_basic():
    """Test available algorithms for basic mode."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = mock_hardware_info_basic()
            mock_monitor.return_value.has_gpu_support.return_value = False

            manager = TrackingHardwareManager()
            algorithms = manager.available_algorithms

            # Should only have CPU-based algorithms
            assert TrackingAlgorithm.CSRT in algorithms
            assert TrackingAlgorithm.KCF in algorithms
            assert TrackingAlgorithm.MOSSE in algorithms
            assert TrackingAlgorithm.MEDIANFLOW in algorithms
            assert TrackingAlgorithm.GOTURN not in algorithms
            assert TrackingAlgorithm.DEEPSORT not in algorithms
            assert TrackingAlgorithm.SORT not in algorithms


def test_available_algorithms_professional():
    """Test available algorithms for professional mode."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = (
                mock_hardware_info_professional()
            )
            mock_monitor.return_value.has_gpu_support.return_value = True

            manager = TrackingHardwareManager()
            algorithms = manager.available_algorithms

            # Should have all algorithms
            for algo in TrackingAlgorithm:
                assert algo in algorithms


def test_recommended_algorithm_accuracy_priority():
    """Test algorithm recommendation with high accuracy priority."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = (
                mock_hardware_info_professional()
            )
            mock_monitor.return_value.has_gpu_support.return_value = True

            manager = TrackingHardwareManager()
            algo = manager.get_recommended_algorithm(
                priority_accuracy=0.8,
                priority_speed=0.1,
                priority_occlusion=0.05,
                priority_recovery=0.05,
            )

            # DEEPSORT has highest accuracy
            assert algo == TrackingAlgorithm.DEEPSORT


def test_recommended_algorithm_speed_priority():
    """Test algorithm recommendation with high speed priority."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = (
                mock_hardware_info_professional()
            )
            mock_monitor.return_value.has_gpu_support.return_value = True

            manager = TrackingHardwareManager()
            algo = manager.get_recommended_algorithm(
                priority_accuracy=0.1,
                priority_speed=0.8,
                priority_occlusion=0.05,
                priority_recovery=0.05,
            )

            # MOSSE has highest speed
            assert algo == TrackingAlgorithm.MOSSE


def test_tracking_config():
    """Test tracking configuration retrieval."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = (
                mock_hardware_info_professional()
            )
            mock_monitor.return_value.has_gpu_support.return_value = True
            mock_monitor.return_value.get_performance_tier.return_value = "high"

            manager = TrackingHardwareManager()
            config = manager.get_tracking_config()

            assert config["tracking_mode"] == TrackingMode.PROFESSIONAL
            assert len(config["available_algorithms"]) > 0
            assert config["hardware_tier"] == "high"
            assert config["gpu_available"] is True
            assert config["cuda_available"] is True


def test_can_run_algorithm():
    """Test algorithm availability checking."""
    with patch("cv2.cuda.getCudaEnabledDeviceCount", return_value=0):
        with patch("spygate.utils.hardware_monitor.HardwareMonitor") as mock_monitor:
            mock_monitor.return_value.get_system_info.return_value = mock_hardware_info_basic()
            mock_monitor.return_value.has_gpu_support.return_value = False

            manager = TrackingHardwareManager()

            # CPU-based algorithms should be available
            assert manager.can_run_algorithm(TrackingAlgorithm.CSRT) is True
            assert manager.can_run_algorithm(TrackingAlgorithm.KCF) is True

            # GPU-based algorithms should not be available
            assert manager.can_run_algorithm(TrackingAlgorithm.DEEPSORT) is False
            assert manager.can_run_algorithm(TrackingAlgorithm.GOTURN) is False


def test_get_algorithm_requirements():
    """Test algorithm requirements retrieval."""
    manager = TrackingHardwareManager()
    reqs = manager.get_algorithm_requirements(TrackingAlgorithm.DEEPSORT)

    assert reqs["min_mode"] == TrackingMode.ADVANCED
    assert reqs["gpu_accelerated"] is True
    assert 0 <= reqs["accuracy"] <= 1
    assert 0 <= reqs["speed"] <= 1
    assert 0 <= reqs["occlusion_handling"] <= 1
    assert 0 <= reqs["recovery"] <= 1


def test_get_mode_requirements():
    """Test mode requirements retrieval."""
    manager = TrackingHardwareManager()
    reqs = manager.get_mode_requirements(TrackingMode.PROFESSIONAL)

    assert reqs["cpu_cores"] == 8
    assert reqs["ram_gb"] == 32
    assert reqs["gpu_required"] is True
    assert reqs["min_vram_gb"] == 8
    assert reqs["cuda_required"] is True
