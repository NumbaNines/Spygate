"""Tests for the tracking pipeline module."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from spygate.core.game_detector import GameVersion
from spygate.utils.tracking_hardware import TrackingMode
from spygate.video.algorithm_selector import SceneComplexity
from spygate.video.formation_analyzer import FormationConfig
from spygate.video.frame_extractor import PreprocessingConfig
from spygate.video.object_tracker import TrackingConfig
from spygate.video.tracking_pipeline import PipelineConfig, TrackingPipeline

# Constants for test configuration
TEST_FRAME_SIZE = (640, 480)
TEST_OBJECT_SIZE = (50, 50)
TEST_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

class TestTrackingPipeline:
    """Test suite for the TrackingPipeline class."""

    def test_pipeline_initialization(self, base_pipeline_config):
        """Test pipeline initialization with configuration."""
        pipeline = TrackingPipeline(config=base_pipeline_config)
        assert pipeline.config == base_pipeline_config
        assert pipeline.frame_count == 0
        assert pipeline.is_initialized is False

    def test_pipeline_configuration(self):
        """Test pipeline configuration validation and defaults."""
        # Test with minimal configuration
        config = PipelineConfig(
            game_version=GameVersion.MODERN,
            tracking_mode=TrackingMode.STANDARD
        )
        pipeline = TrackingPipeline(config=config)
        assert pipeline.config.game_version == GameVersion.MODERN
        assert pipeline.config.tracking_mode == TrackingMode.STANDARD
        
        # Test with full configuration
        full_config = PipelineConfig(
            game_version=GameVersion.MODERN,
            tracking_mode=TrackingMode.HIGH_PERFORMANCE,
            preprocessing=PreprocessingConfig(
                enable_noise_reduction=True,
                enable_stabilization=True
            ),
            tracking=TrackingConfig(
                enable_gpu=True,
                enable_prefetch=True
            ),
            formation=FormationConfig(
                max_players=22,
                field_zones=4
            )
        )
        pipeline = TrackingPipeline(config=full_config)
        assert pipeline.config.preprocessing.enable_noise_reduction
        assert pipeline.config.tracking.enable_gpu
        assert pipeline.config.formation.max_players == 22

    def test_frame_processing(self, pipeline, game_sequence):
        """Test end-to-end frame processing through pipeline."""
        # Initialize pipeline
        success = pipeline.initialize(game_sequence[0])
        assert success, "Failed to initialize pipeline"
        
        # Process sequence
        results = []
        for frame in game_sequence[1:]:
            result = pipeline.process_frame(frame)
            results.append(result)
            assert "tracked_objects" in result
            assert "formation_data" in result
            assert "performance_stats" in result

    def test_algorithm_selection(self, pipeline, scene_complexity_sequence):
        """Test dynamic algorithm selection based on scene complexity."""
        pipeline.initialize(scene_complexity_sequence[0])
        
        # Process frames with varying complexity
        for frame, expected_complexity in scene_complexity_sequence[1:]:
            result = pipeline.process_frame(frame)
            assert result["scene_complexity"] == expected_complexity
            
            # Verify algorithm adaptation
            if expected_complexity == SceneComplexity.HIGH:
                assert pipeline.current_tracker_type in ["CSRT", "KCF"]
            elif expected_complexity == SceneComplexity.LOW:
                assert pipeline.current_tracker_type in ["MOSSE", "MIL"]

    @pytest.mark.benchmark
    def test_pipeline_performance(self, pipeline, game_sequence, benchmark):
        """Benchmark pipeline performance."""
        pipeline.initialize(game_sequence[0])
        
        def process_sequence():
            for frame in game_sequence[1:]:
                pipeline.process_frame(frame)
        
        # Run benchmark
        benchmark(process_sequence)

    def test_hardware_optimization(self, base_pipeline_config):
        """Test pipeline hardware optimization features."""
        with patch('spygate.core.hardware.HardwareDetector') as mock_detector:
            mock_detector.has_gpu.return_value = True
            mock_detector.get_available_memory.return_value = 8192.0
            
            config = base_pipeline_config
            config.tracking.enable_gpu = True
            config.tracking.max_memory_usage = 4096.0
            
            pipeline = TrackingPipeline(config=config)
            assert pipeline.is_gpu_enabled
            assert pipeline.max_memory_usage == 4096.0

    def test_error_handling(self, pipeline):
        """Test pipeline error handling capabilities."""
        # Test invalid frame
        with pytest.raises(ValueError):
            pipeline.process_frame(None)
        
        # Test processing without initialization
        with pytest.raises(RuntimeError):
            pipeline.process_frame(np.zeros(TEST_FRAME_SIZE + (3,), dtype=np.uint8))
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            PipelineConfig(game_version=None, tracking_mode=None)

    def test_pipeline_reset(self, pipeline, game_sequence):
        """Test pipeline reset functionality."""
        # Initialize and process some frames
        pipeline.initialize(game_sequence[0])
        pipeline.process_frame(game_sequence[1])
        
        # Reset pipeline
        pipeline.reset()
        assert pipeline.frame_count == 0
        assert pipeline.is_initialized is False
        
        # Verify can initialize again
        success = pipeline.initialize(game_sequence[0])
        assert success, "Failed to initialize pipeline after reset"

    def test_formation_analysis(self, pipeline, formation_sequence):
        """Test formation analysis integration."""
        pipeline.initialize(formation_sequence[0])
        
        # Process sequence with known formations
        for frame, expected_formation in formation_sequence[1:]:
            result = pipeline.process_frame(frame)
            formation_data = result["formation_data"]
            
            assert formation_data["num_players"] == expected_formation["num_players"]
            assert formation_data["formation_type"] == expected_formation["formation_type"]
            assert np.allclose(
                formation_data["player_positions"],
                expected_formation["player_positions"],
                atol=10.0
            )

    def test_performance_monitoring(self, pipeline, game_sequence):
        """Test performance monitoring and statistics."""
        pipeline.initialize(game_sequence[0])
        
        # Process frames and collect metrics
        metrics = []
        for frame in game_sequence[1:]:
            result = pipeline.process_frame(frame)
            metrics.append(result["performance_stats"])
        
        # Verify metrics
        for metric in metrics:
            assert "processing_time" in metric
            assert "memory_usage" in metric
            assert "gpu_usage" in metric
            assert metric["processing_time"] > 0
            assert metric["memory_usage"] >= 0
            assert 0 <= metric["gpu_usage"] <= 100 if metric["gpu_usage"] is not None else True

    def test_batch_processing(self, pipeline, game_sequence):
        """Test batch processing capabilities."""
        # Initialize pipeline
        pipeline.initialize(game_sequence[0])
        
        # Process in batches
        batch_size = 5
        for i in range(0, len(game_sequence[1:]), batch_size):
            batch = game_sequence[1:][i:i+batch_size]
            results = pipeline.process_batch(batch)
            
            assert len(results) == len(batch)
            for result in results:
                assert "tracked_objects" in result
                assert "formation_data" in result
                assert "performance_stats" in result

    @pytest.mark.parametrize("tracking_mode", [
        TrackingMode.STANDARD,
        TrackingMode.HIGH_PERFORMANCE,
        TrackingMode.QUALITY
    ])
    def test_tracking_modes(self, base_pipeline_config, game_sequence, tracking_mode):
        """Test different tracking modes."""
        config = base_pipeline_config
        config.tracking_mode = tracking_mode
        pipeline = TrackingPipeline(config=config)
        
        pipeline.initialize(game_sequence[0])
        result = pipeline.process_frame(game_sequence[1])
        
        if tracking_mode == TrackingMode.HIGH_PERFORMANCE:
            assert result["performance_stats"]["processing_time"] < 0.1
        elif tracking_mode == TrackingMode.QUALITY:
            assert result["tracked_objects"][0]["confidence"] > 0.8

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 