import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import GPUtil
import numpy as np
import psutil
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from spygate.core.tracking_pipeline import TrackingPipeline
from spygate.services.tracking_service import TrackingService
from spygate.utils.visualization_utils import draw_heat_map, draw_motion_vectors


class TrackingVisualizerView(QWidget):
    """
    A widget that provides real-time visualization of tracking data including
    player positions, ball tracking, and team formations.
    """

    # Signals
    visualization_updated = pyqtSignal()
    stats_updated = pyqtSignal(dict)
    ball_fade_changed = pyqtSignal(bool)
    ball_trail_length_changed = pyqtSignal(int)
    heatmap_sigma_changed = pyqtSignal(float)
    motion_min_magnitude_changed = pyqtSignal(float)

    def __init__(
        self,
        tracking_pipeline: TrackingPipeline,
        tracking_service: TrackingService,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the tracking visualizer view."""
        super().__init__(parent)
        self.tracking_pipeline = tracking_pipeline
        self.tracking_service = tracking_service

        # Performance monitoring
        self.frame_times = []
        self.max_frame_times = 30  # Number of frames to average
        self.target_fps = 30.0
        self.min_fps = 20.0
        self.current_quality = 1.0  # Quality scale factor
        self.quality_step = 0.1  # How much to adjust quality by
        self.min_quality = 0.5
        self.max_quality = 1.0

        # State
        self.current_video_id = None
        self.current_frame = None
        self.current_time = 0.0
        self.fps = 30.0
        self.update_interval = 33  # ~30 FPS
        self.auto_update = True
        self.show_trajectories = True
        self.show_bounding_boxes = True
        self.show_player_ids = True
        self.show_ball_path = True
        self.show_ball_prediction = True
        self.show_formation = True
        self.show_stats = True
        self.show_heat_map = False
        self.show_motion_vectors = False

        # Visualization parameters
        self.heat_map_radius = 30
        self.heat_map_alpha = 0.3
        self.motion_vector_smoothing = 3  # Number of frames to smooth over
        self.ball_prediction_frames = 10  # Number of frames to predict ahead

        # Initialize UI
        self._init_ui()

        # Setup update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(self.update_interval)

        # Connect signals
        self._connect_signals()

        self.last_update_time = time.time()
        self.frame_count = 0

    def _init_ui(self):
        """Initialize the UI layout and components."""
        # Main layout
        layout = QHBoxLayout(self)

        # Create splitter for visualization and stats panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Visualization panel
        self.viz_panel = QFrame()
        viz_layout = QVBoxLayout(self.viz_panel)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_layout.addWidget(self.video_label)

        # Controls panel
        controls_layout = QVBoxLayout()

        # Ball prediction controls
        ball_group = QGroupBox("Ball Prediction")
        ball_layout = QVBoxLayout()

        self.fade_effect_cb = QCheckBox("Fade Effect")
        self.fade_effect_cb.setChecked(True)
        self.fade_effect_cb.stateChanged.connect(
            lambda: self.ball_fade_changed.emit(self.fade_effect_cb.isChecked())
        )

        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail Length:"))
        self.trail_length_sb = QSpinBox()
        self.trail_length_sb.setRange(1, 50)
        self.trail_length_sb.setValue(10)
        self.trail_length_sb.valueChanged.connect(self.ball_trail_length_changed)
        trail_layout.addWidget(self.trail_length_sb)

        ball_layout.addWidget(self.fade_effect_cb)
        ball_layout.addLayout(trail_layout)
        ball_group.setLayout(ball_layout)
        controls_layout.addWidget(ball_group)

        # Heat map controls
        heatmap_group = QGroupBox("Heat Map")
        heatmap_layout = QVBoxLayout()

        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Blur Sigma:"))
        self.sigma_sb = QDoubleSpinBox()
        self.sigma_sb.setRange(1.0, 100.0)
        self.sigma_sb.setValue(30.0)
        self.sigma_sb.valueChanged.connect(self.heatmap_sigma_changed)
        sigma_layout.addWidget(self.sigma_sb)

        heatmap_layout.addLayout(sigma_layout)
        heatmap_group.setLayout(heatmap_layout)
        controls_layout.addWidget(heatmap_group)

        # Motion vector controls
        motion_group = QGroupBox("Motion Vectors")
        motion_layout = QVBoxLayout()

        magnitude_layout = QHBoxLayout()
        magnitude_layout.addWidget(QLabel("Min Magnitude:"))
        self.min_magnitude_sb = QDoubleSpinBox()
        self.min_magnitude_sb.setRange(0.1, 50.0)
        self.min_magnitude_sb.setValue(1.0)
        self.min_magnitude_sb.valueChanged.connect(self.motion_min_magnitude_changed)
        magnitude_layout.addWidget(self.min_magnitude_sb)

        motion_layout.addLayout(magnitude_layout)
        motion_group.setLayout(motion_layout)
        controls_layout.addWidget(motion_group)

        # Toggle switches
        self.bbox_toggle = QCheckBox("Show Bounding Boxes")
        self.bbox_toggle.setChecked(self.show_bounding_boxes)
        self.traj_toggle = QCheckBox("Show Trajectories")
        self.traj_toggle.setChecked(self.show_trajectories)
        self.ball_toggle = QCheckBox("Show Ball Path")
        self.ball_toggle.setChecked(self.show_ball_path)
        self.ball_pred_toggle = QCheckBox("Show Ball Prediction")
        self.ball_pred_toggle.setChecked(self.show_ball_prediction)
        self.form_toggle = QCheckBox("Show Formations")
        self.form_toggle.setChecked(self.show_formation)
        self.heat_map_toggle = QCheckBox("Show Heat Map")
        self.heat_map_toggle.setChecked(self.show_heat_map)
        self.motion_vectors_toggle = QCheckBox("Show Motion")
        self.motion_vectors_toggle.setChecked(self.show_motion_vectors)

        controls_layout.addWidget(self.bbox_toggle)
        controls_layout.addWidget(self.traj_toggle)
        controls_layout.addWidget(self.ball_toggle)
        controls_layout.addWidget(self.ball_pred_toggle)
        controls_layout.addWidget(self.form_toggle)
        controls_layout.addWidget(self.heat_map_toggle)
        controls_layout.addWidget(self.motion_vectors_toggle)

        # Update controls
        self.auto_update_toggle = QCheckBox("Auto Update")
        self.auto_update_toggle.setChecked(self.auto_update)
        self.update_btn = QPushButton("Update")
        self.update_btn.setEnabled(not self.auto_update)

        controls_layout.addWidget(self.auto_update_toggle)
        controls_layout.addWidget(self.update_btn)

        viz_layout.addLayout(controls_layout)

        # Stats panel
        self.stats_panel = QFrame()
        stats_layout = QVBoxLayout(self.stats_panel)

        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextFormat(Qt.TextFormat.RichText)
        stats_layout.addWidget(self.stats_label)

        # Add panels to splitter
        self.splitter.addWidget(self.viz_panel)
        self.splitter.addWidget(self.stats_panel)

        # Set initial splitter sizes
        self.splitter.setSizes([700, 300])

        layout.addWidget(self.splitter)

    def update_video(self, video_id: int, frame: np.ndarray, time: float, fps: float):
        """Update the current video context."""
        self.current_video_id = video_id
        self.current_frame = frame.copy()
        self.current_time = time
        self.fps = fps
        self.update_visualization()

    def update_visualization(self):
        """Update the visualization display."""
        if not self.current_frame is not None:
            return

        # Start frame timing
        start_time = time.time()

        # Get current frame and results
        frame = self.current_frame.copy()
        results = self.tracking_pipeline.get_results()

        # Apply quality scaling if needed
        if self.current_quality < 1.0:
            h, w = frame.shape[:2]
            new_size = (int(w * self.current_quality), int(h * self.current_quality))
            frame = cv2.resize(frame, new_size)

        # Create visualization frame
        vis_frame = frame.copy()

        # Draw tracking results if enabled
        if self.show_bounding_boxes or self.show_trajectories or self.show_player_ids:
            tracking_results = results.get("tracking", {})
            if tracking_results:
                # Draw object bounding boxes and trajectories
                for obj_id, obj_data in tracking_results.get("objects", {}).items():
                    bbox = obj_data.get("bbox")
                    if bbox and self.show_bounding_boxes:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        if self.show_player_ids:
                            cv2.putText(
                                vis_frame,
                                f"ID: {obj_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                    # Draw trajectory
                    if self.show_trajectories:
                        trajectory = obj_data.get("trajectory", [])
                        if len(trajectory) > 1:
                            points = np.array(trajectory, dtype=np.int32)
                            cv2.polylines(vis_frame, [points], False, (255, 0, 0), 2)

        # Draw ball path and prediction if enabled
        if self.show_ball_path or self.show_ball_prediction:
            ball_results = results.get("ball", {})
            if ball_results:
                bbox = ball_results.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Draw ball trajectory
                if self.show_ball_path:
                    trajectory = ball_results.get("trajectory", [])
                    if len(trajectory) > 1:
                        points = np.array(trajectory, dtype=np.int32)
                        cv2.polylines(vis_frame, [points], False, (0, 255, 255), 2)

                # Draw ball prediction
                if self.show_ball_prediction:
                    prediction = ball_results.get("predicted_trajectory", [])
                    if len(prediction) > 1:
                        points = np.array(prediction[: self.ball_prediction_frames], dtype=np.int32)
                        cv2.polylines(vis_frame, [points], False, (255, 255, 0), 1, cv2.LINE_AA)

        # Draw formation if enabled
        if self.show_formation:
            formation_results = results.get("formation", {})
            if formation_results:
                formation_type = formation_results.get("type")
                if formation_type:
                    cv2.putText(
                        vis_frame,
                        f"Formation: {formation_type}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                # Draw player positions and roles
                positions = formation_results.get("player_positions", [])
                roles = formation_results.get("player_roles", {})
                for i, pos in enumerate(positions):
                    x, y = pos[:2]  # Extract x,y coordinates
                    cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                    # Draw role if available
                    if str(i) in roles:
                        cv2.putText(
                            vis_frame,
                            roles[str(i)],
                            (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

        # Draw heat map if enabled
        if self.show_heat_map:
            tracking_results = results.get("tracking", {})
            if tracking_results:
                positions = []
                for obj_data in tracking_results.get("objects", {}).values():
                    bbox = obj_data.get("bbox")
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        positions.append(
                            (
                                int((x1 + x2) / 2 * self.current_quality),
                                int((y1 + y2) / 2 * self.current_quality),
                            )
                        )

                if positions:
                    draw_heat_map(
                        vis_frame,
                        positions,
                        (0, 0, 255),
                        radius=int(self.heat_map_radius * self.current_quality),
                        alpha=self.heat_map_alpha,
                    )

        # Draw motion vectors if enabled
        if self.show_motion_vectors:
            tracking_results = results.get("tracking", {})
            if tracking_results:
                start_points = []
                end_points = []
                for obj_data in tracking_results.get("objects", {}).values():
                    trajectory = obj_data.get("trajectory", [])
                    if len(trajectory) >= self.motion_vector_smoothing + 1:
                        # Use smoothed motion vectors
                        smoothed_trajectory = trajectory[-self.motion_vector_smoothing - 1 :]
                        start_points.append(smoothed_trajectory[0])
                        end_points.append(smoothed_trajectory[-1])

                if start_points and end_points:
                    draw_motion_vectors(
                        vis_frame,
                        start_points,
                        end_points,
                        (255, 0, 0),
                        thickness=2,
                        arrow_size=10.0,
                    )

        # Resize back to original size if scaled
        if self.current_quality < 1.0:
            vis_frame = cv2.resize(vis_frame, (frame.shape[1], frame.shape[0]))

        # Update frame display
        self._update_frame_display(vis_frame)

        # Update performance stats
        end_time = time.time()
        frame_time = end_time - start_time

        # Update frame time history
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)

        # Calculate current FPS
        current_fps = 1.0 / np.mean(self.frame_times)

        # Adjust quality if needed
        if current_fps < self.min_fps and self.current_quality > self.min_quality:
            self.current_quality = max(self.min_quality, self.current_quality - self.quality_step)
        elif current_fps > self.target_fps and self.current_quality < self.max_quality:
            self.current_quality = min(self.max_quality, self.current_quality + self.quality_step)

        # Update stats display
        if self.show_stats:
            stats = {
                "fps": current_fps,
                "quality": self.current_quality,
                "frame_time": frame_time * 1000,  # Convert to ms
                **results,
            }
            self.stats_updated.emit(stats)

        # Emit updated visualization
        self.visualization_updated.emit()

    def _update_frame_display(self, frame: np.ndarray):
        """Update the visualization frame display."""
        if frame is None:
            return

        # Convert frame to RGB for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QPixmap
        height, width = frame_rgb.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(pixmap)

    def _update_stats_display(self, results: dict[str, Any]):
        """Update the statistics display."""
        stats_html = "<div style='font-family: monospace;'>"

        # Performance stats
        stats_html += "<div style='margin: 10px 0;'>"
        stats_html += "<b>Performance:</b><br/>"
        stats_html += f"FPS: {results.get('fps', 0):.1f}<br/>"
        stats_html += f"Frame Time: {results.get('frame_time', 0):.1f}ms<br/>"
        stats_html += f"Quality Scale: {results.get('quality', 1.0):.2f}<br/>"
        stats_html += "</div>"

        # System resources
        stats_html += "<div style='margin: 10px 0;'>"
        stats_html += "<b>System:</b><br/>"
        stats_html += f"CPU Usage: {psutil.cpu_percent()}%<br/>"
        stats_html += f"Memory Usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB<br/>"

        # GPU stats if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                stats_html += f"GPU Usage: {gpu.load * 100:.1f}%<br/>"
                stats_html += f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB<br/>"
        except Exception:
            pass

        stats_html += "</div>"
        self.stats_label.setText(stats_html)

    def _connect_signals(self):
        """Connect all UI signals."""
        # Connect toggle switches
        self.bbox_toggle.toggled.connect(self._on_bbox_toggle)
        self.traj_toggle.toggled.connect(self._on_traj_toggle)
        self.ball_toggle.toggled.connect(self._on_ball_toggle)
        self.ball_pred_toggle.toggled.connect(self._on_ball_pred_toggle)
        self.form_toggle.toggled.connect(self._on_form_toggle)
        self.heat_map_toggle.toggled.connect(self._on_heat_map_toggle)
        self.motion_vectors_toggle.toggled.connect(self._on_motion_vectors_toggle)

        # Connect auto update toggle
        self.auto_update_toggle.toggled.connect(self._on_auto_update_toggle)

        # Connect update button
        self.update_btn.clicked.connect(self.update_visualization)

    def _on_bbox_toggle(self, checked: bool):
        """Handle bounding box toggle."""
        self.show_bounding_boxes = checked
        self.update_visualization()

    def _on_traj_toggle(self, checked: bool):
        """Handle trajectory toggle."""
        self.show_trajectories = checked
        self.update_visualization()

    def _on_ball_toggle(self, checked: bool):
        """Handle ball path toggle."""
        self.show_ball_path = checked
        self.update_visualization()

    def _on_ball_pred_toggle(self, checked: bool):
        """Handle ball prediction toggle."""
        self.show_ball_prediction = checked
        self.update_visualization()

    def _on_form_toggle(self, checked: bool):
        """Handle formation toggle."""
        self.show_formation = checked
        self.update_visualization()

    def _on_heat_map_toggle(self, checked: bool):
        """Handle heat map toggle."""
        self.show_heat_map = checked
        self.update_visualization()

    def _on_motion_vectors_toggle(self, checked: bool):
        """Handle motion vectors toggle."""
        self.show_motion_vectors = checked
        self.update_visualization()

    def _on_auto_update_toggle(self, checked: bool):
        """Handle auto update toggle."""
        self.auto_update = checked
        self.update_btn.setEnabled(not checked)
        if checked:
            self.update_timer.start(self.update_interval)
        else:
            self.update_timer.stop()

    def _on_heat_map_radius_changed(self, value: int):
        """Handle heat map radius change."""
        self.heat_map_radius = value
        self.update_visualization()

    def _on_heat_map_alpha_changed(self, value: float):
        """Handle heat map alpha change."""
        self.heat_map_alpha = value
        self.update_visualization()

    def _on_motion_smoothing_changed(self, value: int):
        """Handle motion vector smoothing change."""
        self.motion_vector_smoothing = value
        self.update_visualization()

    def _on_ball_prediction_frames_changed(self, value: int):
        """Handle ball prediction frames change."""
        self.ball_prediction_frames = value
        self.update_visualization()

    def _on_min_magnitude_changed(self, value: int):
        """Handle minimum magnitude change."""
        # This method is not used in the current implementation
        pass

    def _on_fade_effect_changed(self, state: int):
        """Handle fade effect change."""
        # This method is not used in the current implementation
        pass

    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
        self.update_visualization()  # Update visualization scaling

    def update_performance_metrics(self, processing_time_ms: float):
        """Update performance metrics display."""
        # Calculate FPS
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_update_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_update_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_update_time = current_time

        # Update processing time
        self.proc_time_label.setText(f"Processing Time: {processing_time_ms:.1f} ms")

        # Update memory usage
        memory_percent = psutil.Process().memory_percent()
        self.memory_label.setText(f"Memory Usage: {memory_percent:.1f}%")

        # Update GPU utilization if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.gpu_label.setText(f"GPU Utilization: {gpu.load*100:.1f}%")
            else:
                self.gpu_label.setText("GPU Utilization: No GPU")
        except Exception:
            self.gpu_label.setText("GPU Utilization: Error")

    def update_team_stats(
        self, home_possession: float, away_possession: float, formation_stability: float
    ):
        """Update team statistics display."""
        self.possession_label.setText(
            f"Possession: Home {home_possession:.1f}% - Away {away_possession:.1f}%"
        )
        self.formation_label.setText(f"Formation Stability: {formation_stability:.1f}%")

    def update_tracking_quality(self, player_stats: dict[str, dict[str, float]]):
        """Update player tracking quality information."""
        html = "<table style='width:100%'>"
        html += "<tr><th>Player</th><th>Role</th><th>Quality</th></tr>"

        for player_id, stats in player_stats.items():
            quality = stats.get("tracking_quality", 0) * 100
            role = stats.get("role", "Unknown")

            # Color code based on quality
            if quality >= 90:
                color = "green"
            elif quality >= 70:
                color = "orange"
            else:
                color = "red"

            html += f"<tr>"
            html += f"<td>{player_id}</td>"
            html += f"<td>{role}</td>"
            html += f"<td style='color:{color}'>{quality:.1f}%</td>"
            html += f"</tr>"

        html += "</table>"
        self.quality_text.setHtml(html)

    def get_visualization_params(self) -> dict:
        """Get current visualization parameters."""
        return {
            "ball_fade_effect": self.fade_effect_cb.isChecked(),
            "ball_trail_length": self.trail_length_sb.value(),
            "heatmap_sigma": self.sigma_sb.value(),
            "motion_min_magnitude": self.min_magnitude_sb.value(),
        }
