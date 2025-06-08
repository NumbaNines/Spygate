from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..models import db
from ..models.database import Clip, Collection, ShareHistory, Tag, WatchHistory


class ClipService:
    def __init__(self):
        self.db = db

    def create_clip(self, data: dict[str, Any]) -> Clip:
        """Create a new clip"""
        with self.db.get_session() as session:
            clip = Clip(
                title=data["title"],
                description=data.get("description"),
                filename=data["filename"],
                duration=data["duration"],
                player_name=data.get("player_name"),
                thumbnail_path=data.get("thumbnail_path"),
            )
            session.add(clip)
            session.commit()
            session.refresh(clip)
            return clip

    def get_clip(self, clip_id: int) -> Optional[Clip]:
        """Get a clip by ID"""
        with self.db.get_session() as session:
            return session.query(Clip).filter(Clip.id == clip_id).first()

    def get_clips(
        self,
        player_name: Optional[str] = None,
        tag_names: Optional[list[str]] = None,
        collection_id: Optional[int] = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Clip]:
        """Get clips with optional filtering and sorting"""
        with self.db.get_session() as session:
            query = session.query(Clip)

            # Apply filters
            if player_name:
                query = query.filter(Clip.player_name == player_name)

            if tag_names:
                query = query.join(Clip.tags).filter(Tag.name.in_(tag_names))

            if collection_id:
                query = query.join(Clip.collections).filter(Collection.id == collection_id)

            # Apply sorting
            sort_column = getattr(Clip, sort_by)
            if sort_desc:
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)

            # Apply pagination
            query = query.offset(offset).limit(limit)

            return query.all()

    def update_watch_progress(self, clip_id: int, progress: float) -> None:
        """Update the watch progress for a clip"""
        with self.db.get_session() as session:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            if clip:
                clip.watch_progress = progress
                clip.last_watched = datetime.utcnow()

                # Record in watch history
                history = WatchHistory(
                    clip_id=clip_id, timestamp=progress, action="progress_update"
                )
                session.add(history)
                session.commit()

    def record_watch_action(self, clip_id: int, action: str, timestamp: float) -> None:
        """Record a watch action (play, pause, seek)"""
        with self.db.get_session() as session:
            history = WatchHistory(clip_id=clip_id, timestamp=timestamp, action=action)
            session.add(history)
            session.commit()

    def share_clip(self, clip_id: int, platform: str, channel: str, shared_by: str) -> None:
        """Record a clip share"""
        with self.db.get_session() as session:
            share = ShareHistory(
                clip_id=clip_id, platform=platform, channel=channel, shared_by=shared_by
            )
            session.add(share)
            session.commit()

    def add_to_collection(self, clip_id: int, collection_id: int) -> None:
        """Add a clip to a collection"""
        with self.db.get_session() as session:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            collection = session.query(Collection).filter(Collection.id == collection_id).first()
            if clip and collection:
                clip.collections.append(collection)
                session.commit()

    def remove_from_collection(self, clip_id: int, collection_id: int) -> None:
        """Remove a clip from a collection"""
        with self.db.get_session() as session:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            collection = session.query(Collection).filter(Collection.id == collection_id).first()
            if clip and collection and collection in clip.collections:
                clip.collections.remove(collection)
                session.commit()

    def add_tags(self, clip_id: int, tag_names: list[str]) -> None:
        """Add tags to a clip"""
        with self.db.get_session() as session:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            if clip:
                for name in tag_names:
                    tag = session.query(Tag).filter(Tag.name == name).first()
                    if not tag:
                        tag = Tag(name=name)
                        session.add(tag)
                    if tag not in clip.tags:
                        clip.tags.append(tag)
                session.commit()

    def remove_tags(self, clip_id: int, tag_names: list[str]) -> None:
        """Remove tags from a clip"""
        with self.db.get_session() as session:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            if clip:
                tags = session.query(Tag).filter(Tag.name.in_(tag_names)).all()
                for tag in tags:
                    if tag in clip.tags:
                        clip.tags.remove(tag)
                session.commit()

    def increment_view_count(self, clip_id: int) -> None:
        """Increment the view count for a clip"""
        with self.db.get_session() as session:
            clip = session.query(Clip).filter(Clip.id == clip_id).first()
            if clip:
                clip.view_count += 1
                session.commit()
