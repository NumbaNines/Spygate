#!/usr/bin/env python3
"""
SpygateAI Training GUI Application
This script provides a GUI interface for organizing datasets and training YOLO models
for the SpygateAI project.
"""

import logging
import os
import sys

from dataset_organizer import DatasetOrganizer
from model_trainer import ModelTrainer
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LogHandler(logging.Handler):
    """Custom logging handler that emits logs to a Qt signal."""

    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)


class WorkerThread(QThread):
    """Worker thread for dataset organization and model training."""

    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            # Set up logging
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.addHandler(LogHandler(self.log_updated))

            # Organize dataset
            self.log_updated.emit("Starting dataset organization...")
            organizer = DatasetOrganizer(base_path=os.getcwd())
            organizer.organize()
            self.log_updated.emit("Dataset organization completed.")

            # Train model
            self.log_updated.emit("Starting model training...")
            trainer = ModelTrainer(
                data_yaml="test_dataset/data.yaml", progress_callback=self.progress_updated.emit
            )
            trainer.train()

            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):
    """Main window of the SpygateAI Training GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Training GUI")
        self.setMinimumSize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add title label
        title_label = QLabel("SpygateAI Training Interface")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        # Add progress section
        progress_label = QLabel("Training Progress (Epochs)")
        layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # 100 epochs
        self.progress_bar.setFormat("%v/%m epochs (%p%)")
        layout.addWidget(self.progress_bar)

        # Add log section
        log_label = QLabel("Training Logs")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Add start button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_process)
        layout.addWidget(self.start_button)

        # Initialize worker thread
        self.worker = None

    def start_process(self):
        """Start the dataset organization and training process."""
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = WorkerThread()
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.log_updated.connect(self.update_log)
        self.worker.finished.connect(self.process_finished)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(value)

    def update_log(self, message):
        """Update the log text area."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def process_finished(self):
        """Handle process completion."""
        self.start_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Process Complete",
            "Dataset organization and model training completed successfully!",
        )

    def handle_error(self, error_message):
        """Handle process errors."""
        self.start_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
