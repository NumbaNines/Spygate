from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import DatabaseSession
from .models import Clip, MotionEvent, Player, Tag


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = DatabaseSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_player(
    name: str, team: str = None, position: str = None, session: Session = None
) -> Player:
    """Create a new player."""
    if session is None:
        session = DatabaseSession()

    player = Player(name=name, team=team, position=position)
    session.add(player)
    session.commit()
    return player


def get_player(player_id: int, session: Session = None) -> Optional[Player]:
    """Get a player by ID."""
    if session is None:
        session = DatabaseSession()

    return session.query(Player).filter(Player.id == player_id).first()


def create_clip(
    file_path: str,
    title: str = None,
    player_id: int = None,
    tags: list[str] = None,
    session: Session = None,
) -> Clip:
    """Create a new clip."""
    if session is None:
        session = DatabaseSession()

    clip = Clip(file_path=file_path, title=title, player_id=player_id)

    if tags:
        for tag_name in tags:
            tag = session.query(Tag).filter(Tag.name == tag_name).first()
            if not tag:
                tag = Tag(name=tag_name)
                session.add(tag)
            clip.tags.append(tag)

    session.add(clip)
    session.commit()
    return clip


def get_clip(clip_id: int, session: Session = None) -> Optional[Clip]:
    """Get a clip by ID."""
    if session is None:
        session = DatabaseSession()

    return session.query(Clip).filter(Clip.id == clip_id).first()


def create_motion_event(
    clip_id: int,
    start_time: float,
    end_time: float,
    event_type: str,
    confidence: float,
    event_metadata: dict = None,
    session: Session = None,
) -> MotionEvent:
    """Create a new motion event."""
    if session is None:
        session = DatabaseSession()

    event = MotionEvent(
        clip_id=clip_id,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type,
        confidence=confidence,
        event_metadata=event_metadata,
    )
    session.add(event)
    session.commit()
    return event


def get_motion_events(clip_id: int, session: Session = None) -> list[MotionEvent]:
    """Get all motion events for a clip."""
    if session is None:
        session = DatabaseSession()

    return session.query(MotionEvent).filter(MotionEvent.clip_id == clip_id).all()


def create_tag(name: str, description: str = None, session: Session = None) -> Tag:
    """Create a new tag."""
    if session is None:
        session = DatabaseSession()

    tag = Tag(name=name, description=description)
    session.add(tag)
    session.commit()
    return tag


def get_tag(tag_id: int, session: Session = None) -> Optional[Tag]:
    """Get a tag by ID."""
    if session is None:
        session = DatabaseSession()

    return session.query(Tag).filter(Tag.id == tag_id).first()


def get_clips_by_tag(tag_name: str, session: Session = None) -> list[Clip]:
    """Get all clips with a specific tag."""
    if session is None:
        session = DatabaseSession()

    tag = session.query(Tag).filter(Tag.name == tag_name).first()
    return tag.clips if tag else []


def get_clips_by_player(player_name: str, include_opponent: bool = True) -> list[Clip]:
    """Get all clips for a player."""
    with get_db_session() as session:
        query = session.query(Clip).join(Player).filter(Player.name == player_name)
        if not include_opponent:
            query = query.filter(Clip.is_opponent == False)
        return query.all()


def get_clips_by_tags(tags: list[str], match_all: bool = False) -> list[Clip]:
    """Get clips that match the given tags."""
    with get_db_session() as session:
        query = session.query(Clip).join(Clip.tags).filter(Tag.name.in_(tags))
        if match_all and len(tags) > 1:
            # If match_all is True, ensure clips have all specified tags
            query = query.group_by(Clip.id).having(func.count(Tag.id) == len(tags))
        return query.distinct().all()


def get_motion_events_for_clip(clip_id: int, event_type: Optional[str] = None) -> list[MotionEvent]:
    """Get all motion events for a clip."""
    with get_db_session() as session:
        query = session.query(MotionEvent).filter(MotionEvent.clip_id == clip_id)
        if event_type:
            query = query.filter(MotionEvent.event_type == event_type)
        return query.order_by(MotionEvent.timestamp).all()
