import unittest
from unittest.mock import patch, MagicMock
import torch
import psutil
from spygate.core.hardware import HardwareDetector, HardwareTier, HardwareSpecs

class TestHardwareDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.detector = HardwareDetector()

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_minimum_tier_detection(self, mock_cuda, mock_ram, mock_cpu):
        """Test detection of minimum tier hardware."""
        # Mock hardware with minimum specs
        mock_cpu.return_value = 4  # 4 cores
        mock_ram.return_value = MagicMock(total=8 * (1024**3))  # 8GB RAM
        mock_cuda.return_value = False  # No GPU

        detector = HardwareDetector()
        self.assertEqual(detector.tier, HardwareTier.MINIMUM)

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_standard_tier_detection(self, mock_props, mock_gpu_name, mock_cuda, mock_ram, mock_cpu):
        """Test detection of standard tier hardware."""
        # Mock hardware with standard specs
        mock_cpu.return_value = 6  # 6 cores
        mock_ram.return_value = MagicMock(total=12 * (1024**3))  # 12GB RAM
        mock_cuda.return_value = True
        mock_gpu_name.return_value = "NVIDIA GeForce GTX 1650"
        mock_props.return_value = MagicMock(total_memory=4 * (1024**3))  # 4GB VRAM

        detector = HardwareDetector()
        self.assertEqual(detector.tier, HardwareTier.STANDARD)

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_premium_tier_detection(self, mock_props, mock_gpu_name, mock_cuda, mock_ram, mock_cpu):
        """Test detection of premium tier hardware."""
        # Mock hardware with premium specs
        mock_cpu.return_value = 8  # 8 cores
        mock_ram.return_value = MagicMock(total=16 * (1024**3))  # 16GB RAM
        mock_cuda.return_value = True
        mock_gpu_name.return_value = "NVIDIA GeForce RTX 3060"
        mock_props.return_value = MagicMock(total_memory=8 * (1024**3))  # 8GB VRAM

        detector = HardwareDetector()
        self.assertEqual(detector.tier, HardwareTier.PREMIUM)

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    def test_professional_tier_detection(self, mock_props, mock_gpu_name, mock_cuda, mock_ram, mock_cpu):
        """Test detection of professional tier hardware."""
        # Mock hardware with professional specs
        mock_cpu.return_value = 12  # 12 cores
        mock_ram.return_value = MagicMock(total=32 * (1024**3))  # 32GB RAM
        mock_cuda.return_value = True
        mock_gpu_name.return_value = "NVIDIA GeForce RTX 4080"
        mock_props.return_value = MagicMock(total_memory=16 * (1024**3))  # 16GB VRAM

        detector = HardwareDetector()
        self.assertEqual(detector.tier, HardwareTier.PROFESSIONAL)

    def test_tier_capabilities(self):
        """Test that tier capabilities are correctly reported."""
        capabilities = self.detector.get_tier_capabilities()
        self.assertIn('target_fps', capabilities)
        self.assertIn('max_fps', capabilities)
        self.assertIn('resolution_scale', capabilities)
        self.assertIn('frame_skip', capabilities)

    def test_tier_features(self):
        """Test that tier features are correctly reported."""
        features = self.detector.get_tier_features()
        self.assertIn('enhanced_cv', features)
        self.assertIn('yolo_detection', features)
        self.assertIn('real_time_analysis', features)
        self.assertIn('advanced_formations', features)
        self.assertIn('experimental', features)

    def test_error_handling(self):
        """Test graceful handling of hardware detection errors."""
        with patch('psutil.cpu_count', side_effect=Exception('CPU detection failed')):
            detector = HardwareDetector()
            # Should fallback to minimum tier
            self.assertEqual(detector.tier, HardwareTier.MINIMUM)

if __name__ == '__main__':
    unittest.main()