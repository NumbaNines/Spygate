"""Video import schema update

Revision ID: video_import_schema_update
Revises: a09f3cec3d41
Create Date: 2025-06-07 15:30:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "video_import_schema_update"
down_revision: Union[str, None] = "a09f3cec3d41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update schema for video import feature."""
    # Create video_tags association table
    op.create_table(
        "video_tags",
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("tag_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["videos.id"],
        ),
        sa.ForeignKeyConstraint(
            ["tag_id"],
            ["tags.id"],
        ),
        sa.PrimaryKeyConstraint("video_id", "tag_id"),
    )

    # Add new columns to videos table
    op.add_column("videos", sa.Column("file_path", sa.String(), nullable=False, unique=True))
    op.add_column("videos", sa.Column("width", sa.Integer(), nullable=True))
    op.add_column("videos", sa.Column("height", sa.Integer(), nullable=True))
    op.add_column("videos", sa.Column("fps", sa.Float(), nullable=True))
    op.add_column("videos", sa.Column("codec", sa.String(), nullable=True))
    op.add_column("videos", sa.Column("bitrate", sa.Integer(), nullable=True))
    op.add_column("videos", sa.Column("has_audio", sa.Boolean(), default=False))
    op.add_column("videos", sa.Column("audio_codec", sa.String(), nullable=True))
    op.add_column("videos", sa.Column("preview_gif_path", sa.String(), nullable=True))
    op.add_column("videos", sa.Column("import_status", sa.String(), default="pending"))
    op.add_column("videos", sa.Column("error_message", sa.Text(), nullable=True))
    op.add_column("videos", sa.Column("metadata", sa.JSON(), nullable=True))

    # Create analysis_jobs table
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), default="pending"),
        sa.Column("progress", sa.Float(), default=0.0),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["videos.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create import_logs table
    op.create_table(
        "import_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["videos.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Revert schema changes."""
    # Drop new tables
    op.drop_table("import_logs")
    op.drop_table("analysis_jobs")
    op.drop_table("video_tags")

    # Remove new columns from videos table
    op.drop_column("videos", "metadata")
    op.drop_column("videos", "error_message")
    op.drop_column("videos", "import_status")
    op.drop_column("videos", "preview_gif_path")
    op.drop_column("videos", "audio_codec")
    op.drop_column("videos", "has_audio")
    op.drop_column("videos", "bitrate")
    op.drop_column("videos", "codec")
    op.drop_column("videos", "fps")
    op.drop_column("videos", "height")
    op.drop_column("videos", "width")
    op.drop_column("videos", "file_path")
