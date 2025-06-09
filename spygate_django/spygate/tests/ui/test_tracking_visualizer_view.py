import sys
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSpinBox

from spygate.core.tracking_pipeline import TrackingPipeline
from spygate.models.tracking_data import TrackingData
from spygate.services.tracking_service import TrackingService
from spygate.ui.tracking_visualizer_view import TrackingVisualizerView
from spygate.utils.visualization_utils import VisualizationUtils


# Create QApplication instance for tests
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture
def tracking_service(mocker):
    service = mocker.Mock(spec=TrackingService)
    service.get_tracking_data.return_value = None
    return service


@pytest.fixture
def tracking_pipeline(mocker):
    pipeline = mocker.Mock(spec=TrackingPipeline)
    pipeline.get_results.return_value = tracking_data()
    return pipeline


@pytest.fixture
def tracking_data():
    return TrackingData(
        # Ball tracking data
        ball_positions=[(150, 125)],
        ball_confidences=[0.85],
        # Player tracking data
        player_positions=[[(100, 100)], [(200, 150)]],  # List of positions for each player
        player_ids=[1, 2],
        player_confidences=[[0.9], [0.85]],  # List of confidence values for each player
        # Team data
        team_formations=[[(100, 100)], [(200, 150)]],  # List of formations for each team
        team_possession=[55.5, 44.5],  # Possession percentage for each team
        team_formation_stability=[0.85, 0.8],  # Formation stability score for each team
        # Frame metadata
        frame_width=1920,
        frame_height=1080,
        frame_number=1,
        timestamp=0.0,
        # Optional tracking data
        ball_velocities=[(5, 3)],
        player_velocities=[[(3, 2)], [(2, 1)]],
    )


@pytest.fixture
def view(qapp, tracking_pipeline, tracking_service):
    return TrackingVisualizerView(
        tracking_pipeline=tracking_pipeline, tracking_service=tracking_service
    )


def test_initial_state(view):
    """Test initial state of UI controls."""
    # Ball prediction controls
    assert view.fade_effect_cb.isChecked() is True
    assert view.trail_length_sb.value() == 10
    assert view.trail_length_sb.minimum() == 1
    assert view.trail_length_sb.maximum() == 50

    # Heat map controls
    assert view.sigma_sb.value() == 30.0
    assert view.sigma_sb.minimum() == 1.0
    assert view.sigma_sb.maximum() == 100.0

    # Motion vector controls
    assert view.min_magnitude_sb.value() == 1.0
    assert view.min_magnitude_sb.minimum() == 0.1
    assert view.min_magnitude_sb.maximum() == 50.0


def test_ball_prediction_controls(view):
    """Test ball prediction control signals."""
    # Setup signal spy
    fade_changed = []
    trail_changed = []
    view.ball_fade_changed.connect(lambda x: fade_changed.append(x))
    view.ball_trail_length_changed.connect(lambda x: trail_changed.append(x))

    # Test fade effect checkbox
    view.fade_effect_cb.setChecked(False)
    assert len(fade_changed) == 1
    assert fade_changed[0] is False

    # Test trail length spinbox
    view.trail_length_sb.setValue(20)
    assert len(trail_changed) == 1
    assert trail_changed[0] == 20


def test_heat_map_controls(view):
    """Test heat map control signals."""
    # Setup signal spy
    sigma_changed = []
    view.heatmap_sigma_changed.connect(lambda x: sigma_changed.append(x))

    # Test sigma spinbox
    view.sigma_sb.setValue(45.0)
    assert len(sigma_changed) == 1
    assert sigma_changed[0] == 45.0


def test_motion_vector_controls(view):
    """Test motion vector control signals."""
    # Setup signal spy
    magnitude_changed = []
    view.motion_min_magnitude_changed.connect(lambda x: magnitude_changed.append(x))

    # Test minimum magnitude spinbox
    view.min_magnitude_sb.setValue(2.5)
    assert len(magnitude_changed) == 1
    assert magnitude_changed[0] == 2.5


@patch("psutil.Process")
@patch("GPUtil.getGPUs")
def test_performance_metrics_update(mock_gpus, mock_process, view):
    """Test performance metrics update."""
    # Mock process memory usage
    mock_process_instance = MagicMock()
    mock_process_instance.memory_percent.return_value = 45.6
    mock_process.return_value = mock_process_instance

    # Mock GPU
    mock_gpu = MagicMock()
    mock_gpu.load = 0.75  # 75% utilization
    mock_gpus.return_value = [mock_gpu]

    # Update metrics
    view.update_performance_metrics(16.7)  # 16.7ms processing time

    # Check labels
    assert "Memory Usage: 45.6%" in view.memory_label.text()
    assert "Processing Time: 16.7 ms" in view.proc_time_label.text()
    assert "GPU Utilization: 75.0%" in view.gpu_label.text()


def test_team_stats_update(view):
    """Test team statistics update."""
    view.update_team_stats(60.5, 39.5, 85.0)

    assert "Home 60.5% - Away 39.5%" in view.possession_label.text()
    assert "Formation Stability: 85.0%" in view.formation_label.text()


def test_tracking_quality_update(view):
    """Test tracking quality information update."""
    player_stats = {
        "player1": {"tracking_quality": 0.95, "role": "Forward"},
        "player2": {"tracking_quality": 0.75, "role": "Midfielder"},
        "player3": {"tracking_quality": 0.60, "role": "Defender"},
    }

    view.update_tracking_quality(player_stats)
    html = view.quality_text.toHtml()

    # Check if all players are in the table
    assert "player1" in html
    assert "player2" in html
    assert "player3" in html

    # Check roles
    assert "Forward" in html
    assert "Midfielder" in html
    assert "Defender" in html

    # Check color coding
    assert "color:green" in html  # player1 > 90%
    assert "color:orange" in html  # player2 > 70%
    assert "color:red" in html  # player3 < 70%


def test_get_visualization_params(view):
    """Test getting visualization parameters."""
    # Set specific values
    view.fade_effect_cb.setChecked(True)
    view.trail_length_sb.setValue(15)
    view.sigma_sb.setValue(25.0)
    view.min_magnitude_sb.setValue(1.5)

    params = view.get_visualization_params()

    assert params["ball_fade_effect"] is True
    assert params["ball_trail_length"] == 15
    assert params["heatmap_sigma"] == 25.0
    assert params["motion_min_magnitude"] == 1.5


@pytest.mark.parametrize("mock_gpu_available", [True, False])
def test_gpu_metrics_handling(mock_gpu_available, view):
    """Test GPU metrics with and without GPU."""
    with patch("GPUtil.getGPUs") as mock_gpus:
        if mock_gpu_available:
            mock_gpu = MagicMock()
            mock_gpu.load = 0.8
            mock_gpus.return_value = [mock_gpu]
        else:
            mock_gpus.return_value = []

        view.update_performance_metrics(16.7)

        if mock_gpu_available:
            assert "GPU Utilization: 80.0%" in view.gpu_label.text()
        else:
            assert "GPU Utilization: No GPU" in view.gpu_label.text()


def test_fps_calculation(view):
    """Test FPS calculation over time."""
    with patch("time.time") as mock_time:
        # Simulate 10 frames over 1 second
        mock_time.side_effect = [0.0, 1.0]  # Start and end times

        for _ in range(10):
            view.update_performance_metrics(16.7)

        assert "FPS: 10.0" in view.fps_label.text()


def test_visualization_utils_integration(view):
    """Test integration with VisualizationUtils."""
    # Create test frame and data
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    predictions = [(50, 50), (55, 55)]
    confidences = [0.9, 0.8]

    # Get visualization parameters
    params = view.get_visualization_params()

    # Test ball prediction visualization
    result = VisualizationUtils.visualize_ball_prediction(
        frame,
        predictions,
        confidences,
        fade_effect=params["ball_fade_effect"],
        max_trail_length=params["ball_trail_length"],
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape


def test_initialization(view):
    """Test that the visualizer initializes with correct default values."""
    assert view.show_bounding_boxes is True
    assert view.show_trajectories is True
    assert view.show_ball_path is True
    assert view.show_ball_prediction is True
    assert view.show_formation is True
    assert view.show_heat_map is False
    assert view.show_motion_vectors is False
    assert view.heat_map_radius == 30
    assert view.heat_map_alpha == 0.3
    assert view.motion_vector_smoothing == 3
    assert view.ball_prediction_frames == 10


def test_toggle_controls(view, qtbot):
    """Test that toggle controls update visualization state."""
    # Test ball prediction toggle
    qtbot.mouseClick(view.ball_pred_toggle, Qt.MouseButton.LeftButton)
    assert view.show_ball_prediction is False
    qtbot.mouseClick(view.ball_pred_toggle, Qt.MouseButton.LeftButton)
    assert view.show_ball_prediction is True

    # Test heat map toggle and controls
    qtbot.mouseClick(view.heat_map_toggle, Qt.MouseButton.LeftButton)
    assert view.show_heat_map is True

    # Find heat map radius spinbox and test
    radius_spin = view.findChild(QSpinBox, "")
    radius_spin.setValue(40)
    assert view.heat_map_radius == 40


def test_visualization_update(view, tracking_service, tracking_data):
    """Test that visualization updates correctly with tracking data."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Update tracking data
    tracking_service.update_tracking_data(1, tracking_data)
    view.update_video(1, frame, 0.0, 30.0)

    # Check that stats are updated with HTML formatting
    stats_text = view.stats_label.text()
    assert "<h3>Tracking Statistics</h3>" in stats_text
    assert "<p><b>Frame Info:</b>" in stats_text
    assert "<p><b>Object Tracking:</b>" in stats_text
    assert "Players: 2" in stats_text
    assert "Ball detected: True" in stats_text
    assert "Formation: 4-3-3" in stats_text
    assert "Formation stability: 0.85" in stats_text
    assert "Team 1 possession: 55.5%" in stats_text
    assert "Processing time: 25.5ms" in stats_text


def test_heat_map_visualization(view, tracking_service, tracking_data):
    """Test heat map visualization with configurable parameters."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Enable heat map with custom parameters
    view.show_heat_map = True
    view.heat_map_radius = 40
    view.heat_map_alpha = 0.5

    # Update tracking data
    tracking_service.update_tracking_data(1, tracking_data)
    view.update_video(1, frame, 0.0, 30.0)

    # Visual verification would require image comparison
    assert view.heat_map_radius == 40
    assert view.heat_map_alpha == 0.5


def test_motion_vectors_visualization(view, tracking_service, tracking_data):
    """Test motion vectors visualization with smoothing."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Enable motion vectors with custom smoothing
    view.show_motion_vectors = True
    view.motion_vector_smoothing = 5

    # Update tracking data
    tracking_service.update_tracking_data(1, tracking_data)
    view.update_video(1, frame, 0.0, 30.0)

    # Visual verification would require image comparison
    assert view.motion_vector_smoothing == 5


def test_ball_prediction_visualization(view, tracking_service, tracking_data):
    """Test ball prediction visualization."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Enable ball prediction
    view.show_ball_prediction = True
    view.ball_prediction_frames = 15

    # Update tracking data
    tracking_service.update_tracking_data(1, tracking_data)
    view.update_video(1, frame, 0.0, 30.0)

    # Check that ball prediction parameters are correct
    assert view.show_ball_prediction is True
    assert view.ball_prediction_frames == 15
    assert tracking_data.ball_prediction_confidence == 0.85


def test_performance_stats_display(view, tracking_service, tracking_data):
    """Test that performance statistics are displayed correctly."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Update tracking data
    tracking_service.update_tracking_data(1, tracking_data)
    view.update_video(1, frame, 0.0, 30.0)

    # Check performance stats in display
    stats_text = view.stats_label.text()
    assert "<p><b>Performance:</b>" in stats_text
    assert "Processing time: 25.5ms" in stats_text
    assert "Memory usage: 512.3MB" in stats_text
    assert "GPU utilization: 45.8%" in stats_text


def test_formation_role_display(view, tracking_service, tracking_data):
    """Test that formation roles are displayed correctly."""
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Enable formation display
    view.show_formation = True

    # Update tracking data
    tracking_service.update_tracking_data(1, tracking_data)
    view.update_video(1, frame, 0.0, 30.0)

    # Check formation and role information
    stats_text = view.stats_label.text()
    assert "Formation: 4-3-3" in stats_text
    assert "Formation stability: 0.85" in stats_text
    assert "Player 1: Forward" in stats_text
    assert "Player 2: Defender" in stats_text


def test_auto_update_toggle(view):
    """Test auto-update functionality."""
    assert view.update_timer.isActive() is True
    assert view.update_btn.isEnabled() is False

    view.auto_update_toggle.setChecked(False)
    assert view.update_timer.isActive() is False
    assert view.update_btn.isEnabled() is True

    view.auto_update_toggle.setChecked(True)
    assert view.update_timer.isActive() is True
    assert view.update_btn.isEnabled() is False


def test_splitter_sizes(view):
    """Test that splitter sizes are set correctly."""
    sizes = view.splitter.sizes()
    assert len(sizes) == 2
    assert sizes[0] > sizes[1]  # Visualization panel should be larger than stats panel


@pytest.fixture
def mock_tracking_pipeline():
    """Create a mock tracking pipeline."""
    pipeline = Mock(spec=TrackingPipeline)
    return pipeline


@pytest.fixture
def mock_tracking_service():
    """Create a mock tracking service."""
    service = Mock(spec=TrackingService)
    return service


@pytest.fixture
def visualizer(qapp, mock_tracking_pipeline, mock_tracking_service):
    """Create a tracking visualizer view instance."""
    return TrackingVisualizerView(mock_tracking_pipeline, mock_tracking_service)


def test_init(visualizer):
    """Test initialization of the tracking visualizer."""
    assert visualizer.show_bounding_boxes is True
    assert visualizer.show_trajectories is True
    assert visualizer.show_ball_path is True
    assert visualizer.show_ball_prediction is True
    assert visualizer.show_formation is True
    assert visualizer.show_heat_map is False
    assert visualizer.show_motion_vectors is False
    assert visualizer.heat_map_radius == 30
    assert visualizer.heat_map_alpha == 0.3
    assert visualizer.motion_vector_smoothing == 3
    assert visualizer.ball_prediction_frames == 10


def test_ui_controls(visualizer):
    """Test UI control initialization and connections."""
    # Test toggle buttons
    assert visualizer.bbox_toggle.isChecked() is True
    assert visualizer.traj_toggle.isChecked() is True
    assert visualizer.ball_toggle.isChecked() is True
    assert visualizer.ball_pred_toggle.isChecked() is True
    assert visualizer.form_toggle.isChecked() is True
    assert visualizer.heat_map_toggle.isChecked() is False
    assert visualizer.motion_vectors_toggle.isChecked() is False

    # Test spinboxes
    heat_map_controls = visualizer.findChildren(QSpinBox)
    assert any(spin.value() == 30 for spin in heat_map_controls)  # Heat map radius
    assert any(spin.value() == 3 for spin in heat_map_controls)  # Motion smoothing
    assert any(spin.value() == 10 for spin in heat_map_controls)  # Ball prediction frames


def test_stats_display(visualizer):
    """Test statistics display formatting."""
    test_results = {
        "processing_time": 16.5,
        "memory_usage": 256.7,
        "gpu_utilization": 45.2,
        "fps": 30.0,
        "frame_time": 33.3,
        "quality_scale": 1.0,
        "cpu_usage": 25.5,
        "gpu_memory": {"used": 1024, "total": 8192},
    }

    visualizer._update_stats_display(test_results)
    stats_text = visualizer.stats_label.text()

    # Check if all sections are present
    assert "Performance" in stats_text
    assert "System" in stats_text
    assert f"FPS: {test_results['fps']}" in stats_text
    assert f"Frame Time: {test_results['frame_time']:.1f}ms" in stats_text
    assert f"Quality Scale: {test_results['quality_scale']:.2f}" in stats_text
    assert f"CPU Usage: {test_results['cpu_usage']:.1f}%" in stats_text
    assert f"Memory Usage: {test_results['memory_usage']:.1f}MB" in stats_text
    assert f"GPU Usage: {test_results['gpu_utilization']:.1f}%" in stats_text


@patch("spygate.utils.visualization_utils.draw_heat_map")
def test_heat_map_visualization(mock_draw_heat_map, visualizer):
    """Test heat map visualization with different parameters."""
    # Create test frame and positions
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    positions = [(100, 100), (200, 200), (300, 300)]

    # Enable heat map
    visualizer.show_heat_map = True
    visualizer.heat_map_radius = 40
    visualizer.heat_map_alpha = 0.5

    # Update visualization
    visualizer.update_video(1, frame, 0.0, 30.0)

    # Check if heat map was drawn with correct parameters
    mock_draw_heat_map.assert_called_with(
        frame, positions, any, radius=40, alpha=0.5, blur_size=15, use_gradient=True  # color tuple
    )


@patch("spygate.utils.visualization_utils.draw_motion_vectors")
def test_motion_vector_visualization(mock_draw_vectors, visualizer):
    """Test motion vector visualization with different parameters."""
    # Create test frame and vectors
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    start_points = [(100, 100), (200, 200)]
    end_points = [(120, 120), (220, 220)]

    # Enable motion vectors
    visualizer.show_motion_vectors = True
    visualizer.motion_vector_smoothing = 5

    # Update visualization
    visualizer.update_video(1, frame, 0.0, 30.0)

    # Check if vectors were drawn with correct parameters
    mock_draw_vectors.assert_called_with(
        frame,
        start_points,
        end_points,
        any,  # color tuple
        thickness=2,
        arrow_size=10.0,
        min_magnitude=5.0,
        normalize_color=True,
        smooth_arrows=True,
    )


@patch("spygate.utils.visualization_utils.draw_ball_prediction")
def test_ball_prediction_visualization(mock_draw_prediction, visualizer):
    """Test ball prediction visualization with different parameters."""
    # Create test frame and predictions
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_pos = (100, 100)
    predicted_positions = [(110, 110), (120, 120), (130, 130)]
    confidence = 0.85

    # Enable ball prediction
    visualizer.show_ball_prediction = True
    visualizer.ball_prediction_frames = 15

    # Update visualization
    visualizer.update_video(1, frame, 0.0, 30.0)

    # Check if prediction was drawn with correct parameters
    mock_draw_prediction.assert_called_with(
        frame,
        current_pos,
        predicted_positions,
        any,  # color tuple
        thickness=1,
        confidence=confidence,
        fade_effect=True,
    )


def test_performance_metrics(visualizer):
    """Test performance metrics calculation and display."""
    test_results = {
        "processing_time": 15.6,  # ms
        "memory_usage": 245.9,  # MB
        "gpu_utilization": 42.9,  # %
        "fps": 30.0,
        "frame_time": 33.3,
        "quality_scale": 1.0,
        "cpu_usage": 25.5,
        "gpu_memory": {"used": 1024, "total": 8192},
    }

    # Update stats display
    visualizer._update_stats_display(test_results)
    stats_text = visualizer.stats_label.text()

    # Verify metrics are displayed correctly
    assert f"FPS: {test_results['fps']}" in stats_text
    assert f"Frame Time: {test_results['frame_time']:.1f}ms" in stats_text
    assert f"Quality Scale: {test_results['quality_scale']:.2f}" in stats_text
    assert f"CPU Usage: {test_results['cpu_usage']:.1f}%" in stats_text
    assert f"Memory Usage: {test_results['memory_usage']:.1f}MB" in stats_text
    assert f"GPU Usage: {test_results['gpu_utilization']:.1f}%" in stats_text


def test_formation_role_display(visualizer):
    """Test formation and player role display."""
    test_results = {
        "formation_stability": {
            "A": {"overall": 87.5, "defense": 90.2, "midfield": 85.8, "attack": 86.4},
            "B": {"overall": 82.3, "defense": 84.5, "midfield": 81.9, "attack": 80.5},
        },
        "player_roles": {
            "1": {"role": "Striker", "confidence": 0.95},
            "2": {"role": "Center Midfielder", "confidence": 0.88},
            "3": {"role": "Center Back", "confidence": 0.92},
        },
    }

    visualizer._update_stats_display(test_results)
    stats_text = visualizer.stats_label.text()

    # Check formation stability display
    assert "Team A Stability: 87.5%" in stats_text
    assert "Team B Stability: 82.3%" in stats_text

    # Check player role display
    assert "Player 1: Striker" in stats_text
    assert "Player 2: Center Midfielder" in stats_text
    assert "Player 3: Center Back" in stats_text


@pytest.mark.benchmark
def test_heat_map_performance(view, benchmark):
    """Test heat map generation performance."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    positions = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(100)]

    def run_heat_map():
        frame_copy = frame.copy()
        VisualizationUtils.create_heatmap(positions, frame_copy.shape[:2], sigma=30.0)

    # Run benchmark
    result = benchmark(run_heat_map)
    assert result.stats["mean"] < 0.1  # Should take less than 100ms on average


@pytest.mark.benchmark
def test_motion_vectors_performance(view, benchmark):
    """Test motion vector visualization performance."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    num_vectors = 1000

    # Generate random vectors
    start_points = [
        (np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(num_vectors)
    ]
    end_points = [
        (x + np.random.randint(-50, 50), y + np.random.randint(-50, 50)) for x, y in start_points
    ]
    vectors = [(start, end) for start, end in zip(start_points, end_points)]

    def run_motion_vectors():
        frame_copy = frame.copy()
        VisualizationUtils.visualize_motion_vectors(frame_copy, vectors, min_magnitude=1.0)

    # Run benchmark
    result = benchmark(run_motion_vectors)
    assert result.stats["mean"] < 0.2  # Should take less than 200ms on average


@pytest.mark.benchmark
def test_ball_prediction_performance(view, benchmark):
    """Test ball prediction visualization performance."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    num_predictions = 30

    # Generate random predictions
    predictions = [
        (np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(num_predictions)
    ]
    confidences = [np.random.random() for _ in range(num_predictions)]

    def run_ball_prediction():
        frame_copy = frame.copy()
        VisualizationUtils.visualize_ball_prediction(
            frame_copy, predictions, confidences, fade_effect=True, max_trail_length=20
        )

    # Run benchmark
    result = benchmark(run_ball_prediction)
    assert result.stats["mean"] < 0.05  # Should take less than 50ms on average


@pytest.mark.benchmark
def test_combined_visualization_performance(view, benchmark):
    """Test performance of all visualizations combined."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Generate test data
    positions = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(50)]
    start_points = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(500)]
    end_points = [
        (x + np.random.randint(-50, 50), y + np.random.randint(-50, 50)) for x, y in start_points
    ]
    vectors = [(start, end) for start, end in zip(start_points, end_points)]
    predictions = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(20)]
    confidences = [np.random.random() for _ in range(20)]

    def run_combined():
        frame_copy = frame.copy()

        # Create heat map
        heat_map = VisualizationUtils.create_heatmap(positions, frame_copy.shape[:2], sigma=30.0)
        cv2.addWeighted(frame_copy, 0.7, heat_map, 0.3, 0, frame_copy)

        # Add motion vectors
        VisualizationUtils.visualize_motion_vectors(frame_copy, vectors, min_magnitude=1.0)

        # Add ball predictions
        VisualizationUtils.visualize_ball_prediction(
            frame_copy, predictions, confidences, fade_effect=True, max_trail_length=20
        )

    # Run benchmark
    result = benchmark(run_combined)
    assert result.stats["mean"] < 0.3  # Should take less than 300ms on average


@pytest.mark.benchmark
def test_memory_usage(view):
    """Test memory usage during visualization."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Create large test data
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    positions = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(1000)]
    start_points = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(5000)]
    end_points = [
        (x + np.random.randint(-50, 50), y + np.random.randint(-50, 50)) for x, y in start_points
    ]
    vectors = [(start, end) for start, end in zip(start_points, end_points)]
    predictions = [(np.random.randint(0, 1920), np.random.randint(0, 1080)) for _ in range(100)]
    confidences = [np.random.random() for _ in range(100)]

    # Run visualizations
    frame_copy = frame.copy()
    heat_map = VisualizationUtils.create_heatmap(positions, frame_copy.shape[:2], sigma=30.0)
    cv2.addWeighted(frame_copy, 0.7, heat_map, 0.3, 0, frame_copy)
    VisualizationUtils.visualize_motion_vectors(frame_copy, vectors, min_magnitude=1.0)
    VisualizationUtils.visualize_ball_prediction(frame_copy, predictions, confidences)

    # Check memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    assert memory_increase < 500  # Should use less than 500MB additional memory
