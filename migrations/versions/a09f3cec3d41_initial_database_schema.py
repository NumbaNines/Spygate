"""Initial database schema

Revision ID: a09f3cec3d41
Revises:
Create Date: 2025-06-07 03:25:59.185347

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = "a09f3cec3d41"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables."""
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("email", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )

    # Create clips table
    op.create_table(
        "clips",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("codec", sa.String(), nullable=True),
        sa.Column("bitrate", sa.Integer(), nullable=True),
        sa.Column("player_name", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create transcoded_clips table
    op.create_table(
        "transcoded_clips",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "original_clip_id", sa.Integer(), sa.ForeignKey("clips.id"), nullable=False
        ),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("codec", sa.String(), nullable=False),
        sa.Column("bitrate", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create tags table
    op.create_table(
        "tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Create clip_tags table
    op.create_table(
        "clip_tags",
        sa.Column("clip_id", sa.Integer(), sa.ForeignKey("clips.id")),
        sa.Column("tag_id", sa.Integer(), sa.ForeignKey("tags.id")),
        sa.PrimaryKeyConstraint("clip_id", "tag_id"),
    )

    # Create playbooks table
    op.create_table(
        "playbooks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("game_version", sa.String(), nullable=False),
        sa.Column("formation_count", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create analysis_status enum table for SQLite
    op.create_table(
        "analysis_status",
        sa.Column("name", sa.String(50), primary_key=True),
    )
    op.execute("INSERT INTO analysis_status (name) VALUES ('pending')")
    op.execute("INSERT INTO analysis_status (name) VALUES ('in_progress')")
    op.execute("INSERT INTO analysis_status (name) VALUES ('completed')")
    op.execute("INSERT INTO analysis_status (name) VALUES ('failed')")
    op.execute("INSERT INTO analysis_status (name) VALUES ('cancelled')")

    # Create analysis_jobs table
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "clip_id",
            sa.String(36),
            sa.ForeignKey("clips.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(50),
            sa.ForeignKey("analysis_status.name"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
        sa.Column("progress", sa.Float, nullable=False, server_default=sa.text("0")),
        sa.Column("error_message", sa.String(1000)),
        sa.Column("detected_situations", sa.JSON),
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("analysis_jobs")
    op.drop_table("analysis_status")
    op.drop_table("transcoded_clips")
    op.drop_table("playbooks")
    op.drop_table("clip_tags")
    op.drop_table("tags")
    op.drop_table("clips")
    op.drop_table("users")
