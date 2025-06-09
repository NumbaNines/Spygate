"""Widget for transcoding video files."""

import uuid
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..workers.video_transcode_worker import VideoTranscodeWorker


class VideoTranscodeWidget(QWidget):
    """Widget for transcoding video files."""

    def __init__(self):
        """Initialize the widget."""
        super().__init__()

        # Create worker and thread
        self.worker = VideoTranscodeWorker()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Connect worker signals
        self.worker.progress.connect(self._on_progress)
        self.worker.error.connect(self._on_error)
        self.worker.completed.connect(self._on_completed)
        self.worker.cancelled.connect(self._on_cancelled)

        # Start thread
        self.thread.start()

        # Store current clip ID
        self._current_clip_id: Optional[uuid.UUID] = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()

        # Codec selection
        codec_layout = QHBoxLayout()
        codec_layout.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["h264", "h265", "vp9"])
        codec_layout.addWidget(self.codec_combo)
        layout.addLayout(codec_layout)

        # Resolution selection
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(
            [
                "Original",
                "4K (3840x2160)",
                "1440p (2560x1440)",
                "1080p (1920x1080)",
                "720p (1280x720)",
            ]
        )
        res_layout.addWidget(self.res_combo)
        layout.addLayout(res_layout)

        # FPS control
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(30)
        fps_layout.addWidget(self.fps_spin)
        layout.addLayout(fps_layout)

        # Quality/CRF control
        crf_layout = QHBoxLayout()
        crf_layout.addWidget(QLabel("Quality (CRF):"))
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(1, 51)
        self.crf_spin.setValue(23)
        crf_layout.addWidget(self.crf_spin)
        layout.addLayout(crf_layout)

        # Audio toggle
        audio_layout = QHBoxLayout()
        self.audio_check = QCheckBox("Include Audio")
        self.audio_check.setChecked(True)
        audio_layout.addWidget(self.audio_check)
        layout.addLayout(audio_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_transcode)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_transcode)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def set_clip(self, clip_id: uuid.UUID):
        """Set the clip to transcode.

        Args:
            clip_id: UUID of the clip
        """
        self._current_clip_id = clip_id
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)

    @pyqtSlot()
    def start_transcode(self):
        """Start transcoding the current clip."""
        if not self._current_clip_id:
            QMessageBox.warning(self, "Error", "No clip selected")
            return

        # Get resolution
        res_text = self.res_combo.currentText()
        if res_text == "Original":
            width = height = 0  # Will be handled by service
        else:
            width, height = map(int, res_text.split(" ")[1].strip("()").split("x"))

        # Start transcoding
        self.worker.transcode(
            self._current_clip_id,
            width,
            height,
            float(self.fps_spin.value()),
            self.codec_combo.currentText(),
            crf=self.crf_spin.value(),
            has_audio=self.audio_check.isChecked(),
        )

        # Update UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

    @pyqtSlot()
    def cancel_transcode(self):
        """Cancel the current transcoding operation."""
        self.worker.cancel()

    @pyqtSlot(float)
    def _on_progress(self, percent: float):
        """Handle progress updates.

        Args:
            percent: Progress percentage (0-100)
        """
        self.progress_bar.setValue(int(percent))

    @pyqtSlot(str)
    def _on_error(self, message: str):
        """Handle transcoding errors.

        Args:
            message: Error message
        """
        QMessageBox.critical(self, "Error", f"Transcoding failed: {message}")
        self._reset_ui()

    @pyqtSlot()
    def _on_completed(self):
        """Handle transcoding completion."""
        QMessageBox.information(self, "Success", "Transcoding completed successfully")
        self._reset_ui()

    @pyqtSlot()
    def _on_cancelled(self):
        """Handle transcoding cancellation."""
        QMessageBox.information(self, "Cancelled", "Transcoding cancelled")
        self._reset_ui()

    def _reset_ui(self):
        """Reset the UI to its initial state."""
        self.start_btn.setEnabled(bool(self._current_clip_id))
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)

    def closeEvent(self, event):
        """Handle widget close event.

        Args:
            event: Close event
        """
        # Stop worker thread
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)
