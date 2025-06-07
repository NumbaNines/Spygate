import os
from typing import List, Optional

from PyQt5.QtWidgets import QMessageBox

from ...services.video_service import VideoService
from ...video.codec_validator import CodecValidator, VideoMetadata
from ..dialogs.player_name_dialog import PlayerNameDialog
from ..workers.video_import_worker import VideoImportWorker


class VideoImportWidget:
    def process_files(self, file_paths: List[str]):
        """Process the selected video files.

        Args:
            file_paths: List of file paths to process
        """
        # Validate files first
        valid_files = []
        for file_path in file_paths:
            try:
                metadata = self.codec_validator.validate(file_path)
                if metadata:
                    valid_files.append((file_path, metadata))
            except Exception as e:
                self.import_error.emit(
                    f"Error validating {os.path.basename(file_path)}: {str(e)}"
                )

        if not valid_files:
            QMessageBox.warning(
                self, "Import Error", "No valid video files were found to import."
            )
            return

        # Show player name dialog
        dialog = PlayerNameDialog(self)
        if dialog.exec():
            player_name = dialog.get_player_name()

            # Validate player name
            if player_name == "Opponent" or not player_name:
                QMessageBox.warning(
                    self,
                    "Import Error",
                    "Please provide a name for the opponent player.",
                )
                return

            # Add player name to valid files
            files_with_player = [
                (path, meta, player_name) for path, meta in valid_files
            ]

            # Start import process
            self.start_import(files_with_player)
