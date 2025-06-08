"""
Tests for the hardware detection module.
"""

import unittest
from unittest.mock import MagicMock, patch

import psutil
import torch

from spygate.core.hardware import GPUInfo, HardwareDetector, SystemInfo


class TestHardwareDetector(unittest.TestCase):
    """Test cases for HardwareDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock system info for testing
        self.mock_system_info = SystemInfo(
            cpu_count=4,
            cpu_threads=8,
            ram_total=16384,  # 16GB
            ram_free=8192,  # 8GB
            gpus=[
                GPUInfo(
                    name="NVIDIA GeForce RTX 3060",
                    vram_total=12288,  # 12GB
                    vram_free=10240,  # 10GB
                    cuda_capability=8.6,
                    is_integrated=False,
                )
            ],
            platform="Windows",
            cuda_available=True,
            opencv_gpu=True,
        )

        # Create patcher for psutil.cpu_count
        self.cpu_count_patcher = patch("psutil.cpu_count")
        self.mock_cpu_count = self.cpu_count_patcher.start()
        self.mock_cpu_count.side_effect = lambda logical: 8 if logical else 4

        # Create patcher for psutil.virtual_memory
        self.virtual_memory_patcher = patch("psutil.virtual_memory")
        self.mock_virtual_memory = self.virtual_memory_patcher.start()
        self.mock_virtual_memory.return_value = MagicMock(
            total=16384 * 1024 * 1024,  # 16GB in bytes
            available=8192 * 1024 * 1024,  # 8GB in bytes
        )

        # Create patcher for torch.cuda
        self.cuda_patcher = patch("torch.cuda")
        self.mock_cuda = self.cuda_patcher.start()
        self.mock_cuda.is_available.return_value = True
        self.mock_cuda.device_count.return_value = 1
        self.mock_cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA GeForce RTX 3060",
            total_memory=12288 * 1024 * 1024,  # 12GB in bytes
            major=8,
            minor=6,
        )
        self.mock_cuda.memory_allocated.return_value = 2048 * 1024 * 1024  # 2GB in bytes

        # Create patcher for cv2.cuda
        self.cv2_cuda_patcher = patch("cv2.cuda")
        self.mock_cv2_cuda = self.cv2_cuda_patcher.start()
        self.mock_cv2_cuda.getCudaEnabledDeviceCount.return_value = 1

    def tearDown(self):
        """Clean up test fixtures."""
        self.cpu_count_patcher.stop()
        self.virtual_memory_patcher.stop()
        self.cuda_patcher.stop()
        self.cv2_cuda_patcher.stop()

    def test_game_compatibility_madden_25(self):
        """Test game compatibility check for Madden 25."""
        detector = HardwareDetector()
        is_compatible, issues = detector.check_game_compatibility("madden_25")
        self.assertTrue(is_compatible)
        self.assertEqual(len(issues), 0)

    def test_game_compatibility_unsupported_game(self):
        """Test game compatibility check for unsupported game."""
        detector = HardwareDetector()
        is_compatible, issues = detector.check_game_compatibility("unsupported_game")
        self.assertFalse(is_compatible)
        self.assertEqual(len(issues), 1)
        self.assertIn("Unsupported game", issues[0])

    def test_game_profile_madden_25(self):
        """Test game profile for Madden 25."""
        detector = HardwareDetector()
        profile = detector.get_game_profile("madden_25")
        
        self.assertIsNotNone(profile)
        self.assertTrue(profile["compatible"])
        self.assertTrue(profile["meets_minimum"])
        self.assertTrue(profile["meets_recommended"])
        self.assertEqual(profile["performance_tier"], "professional")
        self.assertEqual(len(profile["issues"]), 0)

    def test_optimal_settings_madden_25(self):
        """Test optimal settings for Madden 25."""
        detector = HardwareDetector()
        settings = detector.get_optimal_settings("madden_25")
        
        self.assertEqual(settings["resolution"], "native")
        self.assertEqual(settings["frame_sampling_rate"], 60)
        self.assertEqual(settings["processing_quality"], "ultra")
        self.assertEqual(settings["batch_size"], 32)

    def test_minimum_spec_system(self):
        """Test compatibility with minimum spec system."""
        # Mock a minimum spec system
        self.mock_cpu_count.side_effect = lambda logical: 4 if logical else 2
        self.mock_virtual_memory.return_value = MagicMock(
            total=8192 * 1024 * 1024,  # 8GB
            available=4096 * 1024 * 1024,  # 4GB
        )
        self.mock_cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA GeForce GTX 1650",
            total_memory=2048 * 1024 * 1024,  # 2GB
            major=7,
            minor=5,
        )

        detector = HardwareDetector()
        is_compatible, issues = detector.check_game_compatibility("madden_25")
        
        self.assertTrue(is_compatible)  # Should meet minimum requirements
        profile = detector.get_game_profile("madden_25")
        self.assertTrue(profile["meets_minimum"])
        self.assertFalse(profile["meets_recommended"])
        self.assertEqual(profile["performance_tier"], "minimum")

    def test_below_minimum_spec_system(self):
        """Test compatibility with below minimum spec system."""
        # Mock a below minimum spec system
        self.mock_cpu_count.side_effect = lambda logical: 2 if logical else 1
        self.mock_virtual_memory.return_value = MagicMock(
            total=4096 * 1024 * 1024,  # 4GB
            available=2048 * 1024 * 1024,  # 2GB
        )
        self.mock_cuda.is_available.return_value = False
        self.mock_cv2_cuda.getCudaEnabledDeviceCount.return_value = 0

        detector = HardwareDetector()
        is_compatible, issues = detector.check_game_compatibility("madden_25")
        
        self.assertFalse(is_compatible)  # Should not meet minimum requirements
        self.assertGreater(len(issues), 0)  # Should have multiple issues
        profile = detector.get_game_profile("madden_25")
        self.assertFalse(profile["meets_minimum"])
        self.assertFalse(profile["meets_recommended"])
        self.assertEqual(profile["performance_tier"], "minimum")

    def test_performance_tier_calculation(self):
        """Test performance tier calculation."""
        detector = HardwareDetector()
        
        # Should be "professional" with our mocked high-end system
        self.assertEqual(detector.performance_tier, "professional")

        # Test premium tier
        self.mock_virtual_memory.return_value = MagicMock(
            total=16384 * 1024 * 1024,  # 16GB
            available=8192 * 1024 * 1024,  # 8GB
        )
        self.mock_cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA GeForce RTX 3050",
            total_memory=8192 * 1024 * 1024,  # 8GB
            major=8,
            minor=6,
        )
        detector.refresh()
        self.assertEqual(detector.performance_tier, "premium")

        # Test standard tier
        self.mock_virtual_memory.return_value = MagicMock(
            total=8192 * 1024 * 1024,  # 8GB
            available=4096 * 1024 * 1024,  # 4GB
        )
        self.mock_cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA GeForce GTX 1650",
            total_memory=4096 * 1024 * 1024,  # 4GB
            major=7,
            minor=5,
        )
        detector.refresh()
        self.assertEqual(detector.performance_tier, "standard")

    def test_optimal_thread_count(self):
        """Test optimal thread count calculation."""
        detector = HardwareDetector()
        # With 8 threads, should use 75% = 6 threads
        self.assertEqual(detector.get_optimal_thread_count(), 6)

    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        detector = HardwareDetector()
        # Should be 32 for professional tier
        self.assertEqual(detector.get_optimal_batch_size(), 32)

        # Test other tiers
        self.mock_virtual_memory.return_value = MagicMock(
            total=8192 * 1024 * 1024,  # 8GB
            available=4096 * 1024 * 1024,  # 4GB
        )
        self.mock_cuda.get_device_properties.return_value = MagicMock(
            name="NVIDIA GeForce GTX 1650",
            total_memory=4096 * 1024 * 1024,  # 4GB
            major=7,
            minor=5,
        )
        detector.refresh()
        self.assertEqual(detector.get_optimal_batch_size(), 8)  # Standard tier

    def test_vram_info(self):
        """Test VRAM information retrieval."""
        detector = HardwareDetector()
        vram_info = detector.get_vram_info()
        
        self.assertEqual(vram_info["total"], 12288)  # 12GB
        self.assertEqual(vram_info["free"], 10240)  # 10GB
        self.assertEqual(vram_info["used"], 2048)  # 2GB

    def test_ram_info(self):
        """Test RAM information retrieval."""
        detector = HardwareDetector()
        ram_info = detector.get_ram_info()
        
        self.assertEqual(ram_info["total"], 16384)  # 16GB
        self.assertEqual(ram_info["free"], 8192)  # 8GB
        self.assertEqual(ram_info["used"], 8192)  # 8GB 