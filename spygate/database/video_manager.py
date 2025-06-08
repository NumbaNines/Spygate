"""
Database manager for video-related operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..video.metadata import VideoMetadata
from .schema import AnalysisJob, ImportLog, Player, Tag, Video, video_players


class VideoManager:
    """Manages video-related database operations."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize with database session."""
        self.session = session or Session()

    def get_or_create_player(
        self,
        name: str,
        team: Optional[str] = None,
        is_self: bool = False,
        gamertag: Optional[str] = None,
    ) -> Player:
        """
        Get an existing player or create a new one.

        Args:
            name: Player name
            team: Optional team name
            is_self: Whether this is the user's own gameplay
            gamertag: Optional gamertag

        Returns:
            Player: The player object
        """
        # Try to find existing player
        query = self.session.query(Player)
        if is_self:
            player = query.filter_by(is_self=True).first()
        else:
            player = query.filter_by(name=name, is_self=False).first()

        if player:
            # Update existing player if needed
            if team and team != player.team:
                player.team = team
            if gamertag and gamertag != player.gamertag:
                player.gamertag = gamertag
            player.updated_at = datetime.utcnow()
        else:
            # Create new player
            player = Player(name=name, team=team, is_self=is_self, gamertag=gamertag)
            self.session.add(player)

        return player

    def create_video(
        self, metadata: VideoMetadata, players: List[Dict[str, Any]]
    ) -> Tuple[Video, List[str]]:
        """
        Create a new video entry in the database.

        Args:
            metadata: VideoMetadata object with extracted information
            players: List of player information dictionaries, each containing:
                    - name: Player name
                    - team: Optional team name
                    - is_self: Whether this is the user's own gameplay
                    - gamertag: Optional gamertag
                    - is_primary: Whether this is the primary player

        Returns:
            Tuple[Video, List[str]]: Created video object and list of warnings
        """
        warnings = []

        try:
            # Create video object
            video = Video(
                filename=metadata.file_name,
                file_path=str(metadata.file_path),
                duration=metadata.duration,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
                codec=metadata.codec,
                bitrate=metadata.bit_rate,
                has_audio=metadata.has_audio,
                audio_codec=metadata.audio_codec,
                import_status="pending",
            )

            # Add players
            for player_info in players:
                player = self.get_or_create_player(
                    name=player_info["name"],
                    team=player_info.get("team"),
                    is_self=player_info.get("is_self", False),
                    gamertag=player_info.get("gamertag"),
                )

                # Add to video with primary flag
                self.session.execute(
                    video_players.insert().values(
                        video_id=video.id,
                        player_id=player.id,
                        is_primary=player_info.get("is_primary", False),
                    )
                )

            self.session.add(video)
            self.session.commit()

            # Log the creation
            self._log_operation(
                video.id, "create", "success", "Video entry created successfully"
            )

        except IntegrityError as e:
            self.session.rollback()
            if "UNIQUE constraint failed: videos.file_path" in str(e):
                warnings.append(
                    f"Video at {metadata.file_path} already exists in database"
                )
            else:
                raise

        return video, warnings

    def update_video_status(
        self, video_id: int, status: str, error: Optional[str] = None
    ) -> None:
        """
        Update the import status of a video.

        Args:
            video_id: ID of the video to update
            status: New status ('pending', 'processing', 'completed', 'failed')
            error: Optional error message if status is 'failed'
        """
        video = self.session.query(Video).get(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        video.import_status = status
        video.error_message = error

        self.session.commit()

        # Log the status update
        self._log_operation(
            video_id,
            "status_update",
            "success" if status != "failed" else "failed",
            f"Status updated to {status}",
            {"error": error} if error else None,
        )

    def get_video_by_hash(self, file_hash: str) -> Optional[Video]:
        """
        Get a video by its file hash.

        Args:
            file_hash: SHA-256 hash of the video file

        Returns:
            Optional[Video]: Video object if found, None otherwise
        """
        return self.session.query(Video).filter_by(file_hash=file_hash).first()

    def get_video_by_path(self, file_path: str) -> Optional[Video]:
        """
        Get a video by its file path.

        Args:
            file_path: Path to the video file

        Returns:
            Optional[Video]: Video object if found, None otherwise
        """
        return self.session.query(Video).filter_by(file_path=file_path).first()

    def get_videos_by_player(self, player_id: int) -> List[Video]:
        """
        Get all videos associated with a player.

        Args:
            player_id: ID of the player

        Returns:
            List[Video]: List of video objects
        """
        player = self.session.query(Player).get(player_id)
        if not player:
            raise ValueError(f"No player found with ID {player_id}")

        return player.videos

    def get_player_videos_by_name(self, player_name: str) -> List[Video]:
        """
        Get all videos associated with a player by their name.

        Args:
            player_name: Name of the player

        Returns:
            List[Video]: List of video objects
        """
        player = self.session.query(Player).filter_by(name=player_name).first()
        if not player:
            return []

        return player.videos

    def add_video_tag(self, video_id: int, tag_name: str) -> None:
        """
        Add a tag to a video.

        Args:
            video_id: ID of the video
            tag_name: Name of the tag to add
        """
        video = self.session.query(Video).get(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        # Get or create tag
        tag = self.session.query(Tag).filter_by(name=tag_name).first()
        if not tag:
            tag = Tag(name=tag_name)
            self.session.add(tag)

        # Add tag to video if not already present
        if tag not in video.tags:
            video.tags.append(tag)
            self.session.commit()

            # Log the operation
            self._log_operation(
                video_id, "add_tag", "success", f"Added tag '{tag_name}'"
            )

    def remove_video_tag(self, video_id: int, tag_name: str) -> None:
        """
        Remove a tag from a video.

        Args:
            video_id: ID of the video
            tag_name: Name of the tag to remove
        """
        video = self.session.query(Video).get(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        tag = self.session.query(Tag).filter_by(name=tag_name).first()
        if tag and tag in video.tags:
            video.tags.remove(tag)
            self.session.commit()

            # Log the operation
            self._log_operation(
                video_id, "remove_tag", "success", f"Removed tag '{tag_name}'"
            )

    def get_video_tags(self, video_id: int) -> List[str]:
        """
        Get all tags associated with a video.

        Args:
            video_id: ID of the video

        Returns:
            List[str]: List of tag names
        """
        video = self.session.query(Video).get(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        return [tag.name for tag in video.tags]

    def update_video_notes(self, video_id: int, notes: str) -> None:
        """
        Update the notes for a video.

        Args:
            video_id: ID of the video
            notes: New notes text
        """
        video = self.session.query(Video).get(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        video.notes = notes
        self.session.commit()

        # Log the operation
        self._log_operation(video_id, "update_notes", "success", "Updated video notes")

    def mark_video_deleted(self, video_id: int) -> None:
        """
        Mark a video as deleted (soft delete).

        Args:
            video_id: ID of the video
        """
        video = self.session.query(Video).get(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        video.is_deleted = True
        video.delete_date = datetime.utcnow()
        self.session.commit()

        # Log the operation
        self._log_operation(
            video_id, "mark_deleted", "success", "Marked video as deleted"
        )

    def get_recent_videos(self, limit: int = 10) -> List[Video]:
        """
        Get the most recently imported videos.

        Args:
            limit: Maximum number of videos to return

        Returns:
            List[Video]: List of video objects
        """
        return (
            self.session.query(Video)
            .filter_by(is_deleted=False)
            .order_by(Video.import_date.desc())
            .limit(limit)
            .all()
        )

    def get_videos_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Video]:
        """
        Get videos imported within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List[Video]: List of video objects
        """
        return (
            self.session.query(Video)
            .filter(
                Video.import_date >= start_date,
                Video.import_date <= end_date,
                Video.is_deleted == False,
            )
            .order_by(Video.import_date.desc())
            .all()
        )

    def update_analysis_progress(
        self,
        job_id: int,
        progress: float,
        status: Optional[str] = None,
        results: Optional[Dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the progress of an analysis job.

        Args:
            job_id: ID of the job to update
            progress: Progress value (0.0 to 1.0)
            status: Optional new status
            results: Optional results data
            error: Optional error message
        """
        job = self.session.query(AnalysisJob).get(job_id)
        if not job:
            raise ValueError(f"No analysis job found with ID {job_id}")

        job.progress = progress

        if status:
            job.status = status
            if status == "completed":
                job.completed_at = datetime.utcnow()
            elif status == "processing" and not job.started_at:
                job.started_at = datetime.utcnow()

        if results is not None:
            job.results = results

        if error is not None:
            job.error_message = error

        self.session.commit()

    def _log_operation(
        self,
        video_id: int,
        operation: str,
        status: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> None:
        """Log a video operation."""
        log = ImportLog(
            video_id=video_id,
            operation=operation,
            status=status,
            message=message,
            details=details,
        )
        self.session.add(log)
        self.session.commit()
