from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QFrame,
    QProgressDialog,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDragLeaveEvent, QPalette, QColor, QPixmap, QImage
from ...video.codec_validator import CodecValidator, VideoMetadata
import cv2
import os

class VideoImportWorker(QThread):
    """Worker thread for importing and validating videos."""
    
    progress = pyqtSignal(int, int, str, bool, str)  # current, total, path, success, message
    finished = pyqtSignal(list)  # List of (path, metadata) tuples
    
    def __init__(self, video_files):
        super().__init__()
        self.video_files = video_files
        self.file_paths = video_files  # Alias for test compatibility
        self._should_stop = False
    
    def run(self):
        """Process video files."""
        results = []
        total = len(self.video_files)
        for i, path in enumerate(self.video_files, 1):
            if self._should_stop:
                break
            try:
                is_valid, error_message, metadata = CodecValidator.validate_video(path)
                success = is_valid and metadata is not None
                message = error_message if not success else ""
                results.append((path, is_valid, error_message, metadata))
            except Exception as e:
                results.append((path, False, str(e), None))
            percent = int(100 * i / total) if total > 0 else 100
            self.progress.emit(percent, total, path, success, message)
        self.finished.emit(results)
    
    def stop(self):
        """Stop processing."""
        self._should_stop = True

class VideoListItem(QWidget):
    deleteClicked = pyqtSignal(str)  # Signal emits the video path
    
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Video name label
        self.name_label = QLabel(os.path.basename(video_path))
        self.name_label.setStyleSheet("color: white;")
        layout.addWidget(self.name_label)
        
        # Spacer
        layout.addStretch()
        
        # Delete button
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                color: white;
            }
            QPushButton:hover {
                background-color: #f75c4c;
            }
            QPushButton:pressed {
                background-color: #d74c3c;
            }
        """)
        delete_btn.setFixedWidth(70)
        delete_btn.clicked.connect(lambda: self.deleteClicked.emit(self.video_path))
        layout.addWidget(delete_btn)

class VideoImportWidget(QWidget):
    """Widget for importing videos via drag and drop."""
    
    videosImported = pyqtSignal(list)  # List of (path, metadata) tuples
    videoDeleted = pyqtSignal(str)  # Emits the path of the deleted video
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.imported_videos = []
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # Main drop frame
        self.drop_frame = QFrame()
        self.drop_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #666;
                border-radius: 8px;
                background: #2a2a2a;
            }
        """)
        drop_layout = QVBoxLayout(self.drop_frame)
        drop_layout.setContentsMargins(0, 0, 0, 0)
        drop_layout.setSpacing(0)
        # Main label
        self.label = QLabel("Drop video files here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumHeight(40)
        self.label.setStyleSheet("color: #666;")
        drop_layout.addWidget(self.label)
        # Supported formats label
        self.formats_label = QLabel("Supported formats: MP4, AVI, MOV, MKV")
        self.formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formats_label.setStyleSheet("color: #888; font-size: 11px;")
        drop_layout.addWidget(self.formats_label)
        layout.addWidget(self.drop_frame)
        # List of imported videos
        self.video_list = QListWidget()
        self.video_list.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #666;
                border-radius: 4px;
                color: white;
            }
            QListWidget::item {
                padding: 5px;
                background-color: #333;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #444;
            }
        """)
        self.video_list.setMinimumHeight(150)
        layout.addWidget(self.video_list)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) for url in urls):
                self.drop_frame.setStyleSheet("""
                    QFrame {
                        border: 2px dashed #0078d4;
                        border-radius: 8px;
                        background: #2a2a2a;
                    }
                """)
                self.label.setStyleSheet("color: #0078d4;")
                event.accept()
                return
        event.ignore()
    
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        """Handle drag leave event."""
        self.drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #666;
                border-radius: 8px;
                background: #2a2a2a;
            }
        """)
        self.label.setStyleSheet("color: #666;")
        if event:
            event.accept()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        video_files = [url.toLocalFile() for url in event.mimeData().urls()
                      if url.toLocalFile().lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            return
        
        self.drop_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #666;
                border-radius: 8px;
                background: #2a2a2a;
            }
        """)
        self.label.setStyleSheet("color: #666;")
        
        # Create and start worker thread
        self.worker = VideoImportWorker(video_files)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_import_finished)
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setWindowTitle("Importing Videos")
        self.progress_dialog.setLabelText("Validating videos...")
        self.progress_dialog.setRange(0, len(video_files))
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setMinimumWidth(300)
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Style the progress dialog
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: #2d2d2d;
            }
            QProgressBar {
                border: 1px solid #666;
                border-radius: 3px;
                background-color: #1a1a1a;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3daee9;
                border-radius: 2px;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #3daee9;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4dbef9;
            }
            QPushButton:pressed {
                background-color: #2d9ed9;
            }
        """)
        
        # Connect cancel button
        cancel_button = self.progress_dialog.findChild(QPushButton)
        if cancel_button:
            cancel_button.setText("Cancel Import")
            cancel_button.setFixedWidth(100)
        
        self.progress_dialog.canceled.connect(self.worker.stop)
        self.progress_dialog.show()
        
        # Start processing
        self.worker.start()
        
        event.accept()
    
    def _on_progress(self, current: int, total: int, path: str, success: bool, message: str):
        """Handle progress updates."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(f"Processing: {os.path.basename(path)}")
            if not success and message:
                QMessageBox.warning(self, "Import Error", f"Error importing {path}:\n{message}")
    
    def _on_import_finished(self, results):
        """Handle import completion."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Filter successful imports
        successful_imports = [(path, metadata) for path, is_valid, error_message, metadata in results if is_valid and metadata is not None]
        
        if successful_imports:
            self.imported_videos.extend(successful_imports)
            self.videosImported.emit(successful_imports)
            
            # Add to list widget
            for path, _ in successful_imports:
                item = QListWidgetItem(self.video_list)
                widget = VideoListItem(path)
                widget.deleteClicked.connect(self._on_delete_video)
                item.setSizeHint(widget.sizeHint())
                self.video_list.addItem(item)
                self.video_list.setItemWidget(item, widget)
    
    def _on_delete_video(self, video_path: str):
        """Handle video deletion."""
        # Find and remove from imported_videos list
        self.imported_videos = [(p, m) for p, m in self.imported_videos if p != video_path]
        
        # Find and remove from list widget
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            widget = self.video_list.itemWidget(item)
            if widget.video_path == video_path:
                self.video_list.takeItem(i)
                break
        
        self.videoDeleted.emit(video_path) 