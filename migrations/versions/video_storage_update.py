"""Video storage schema update

Revision ID: video_storage_update
Revises: video_import_schema_update
Create Date: 2025-06-07 16:30:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "video_storage_update"
down_revision: Union[str, None] = "video_import_schema_update"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update schema for video storage management."""
    # Modify videos table
    with op.batch_alter_table("videos") as batch_op:
        # Add new columns
        batch_op.add_column(sa.Column("file_hash", sa.String(64), nullable=True, unique=True))
        batch_op.add_column(sa.Column("file_size", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("original_filename", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("is_deleted", sa.Boolean(), nullable=True, default=False))
        batch_op.add_column(sa.Column("delete_date", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("notes", sa.Text(), nullable=True))

        # Rename columns
        batch_op.alter_column("upload_date", new_column_name="import_date")
        batch_op.alter_column("bitrate", new_column_name="bit_rate")

        # Drop old columns
        batch_op.drop_column("import_status")
        batch_op.drop_column("error_message")
        batch_op.drop_column("metadata")
        batch_op.drop_column("thumbnail_path")
        batch_op.drop_column("preview_gif_path")

        # Modify column types and constraints
        batch_op.alter_column(
            "file_path",
            existing_type=sa.String(),
            type_=sa.String(255),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "player_name",
            existing_type=sa.String(),
            type_=sa.String(100),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "codec",
            existing_type=sa.String(),
            type_=sa.String(50),
            existing_nullable=True,
            nullable=False,
        )
        batch_op.alter_column(
            "audio_codec",
            existing_type=sa.String(),
            type_=sa.String(50),
            existing_nullable=True,
        )

    # Modify analysis_jobs table
    with op.batch_alter_table("analysis_jobs") as batch_op:
        # Rename columns
        batch_op.alter_column("started_at", new_column_name="start_time")
        batch_op.alter_column("completed_at", new_column_name="end_time")

        # Drop old columns
        batch_op.drop_column("progress")
        batch_op.drop_column("results")

        # Modify column types and constraints
        batch_op.alter_column(
            "job_type",
            existing_type=sa.String(),
            type_=sa.String(50),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "status",
            existing_type=sa.String(),
            type_=sa.String(20),
            existing_nullable=True,
            nullable=False,
        )
        batch_op.alter_column(
            "start_time",
            existing_type=sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        )


def downgrade() -> None:
    """Revert schema changes."""
    # Revert videos table changes
    with op.batch_alter_table("videos") as batch_op:
        # Drop new columns
        batch_op.drop_column("file_hash")
        batch_op.drop_column("file_size")
        batch_op.drop_column("original_filename")
        batch_op.drop_column("is_deleted")
        batch_op.drop_column("delete_date")
        batch_op.drop_column("notes")

        # Restore old columns
        batch_op.add_column(sa.Column("import_status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("error_message", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("metadata", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("thumbnail_path", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("preview_gif_path", sa.String(), nullable=True))

        # Revert column renames
        batch_op.alter_column("import_date", new_column_name="upload_date")
        batch_op.alter_column("bit_rate", new_column_name="bitrate")

        # Revert column types and constraints
        batch_op.alter_column(
            "file_path",
            existing_type=sa.String(255),
            type_=sa.String(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "player_name",
            existing_type=sa.String(100),
            type_=sa.String(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "codec",
            existing_type=sa.String(50),
            type_=sa.String(),
            existing_nullable=False,
            nullable=True,
        )
        batch_op.alter_column(
            "audio_codec",
            existing_type=sa.String(50),
            type_=sa.String(),
            existing_nullable=True,
        )

    # Revert analysis_jobs table changes
    with op.batch_alter_table("analysis_jobs") as batch_op:
        # Restore old columns
        batch_op.add_column(sa.Column("progress", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("results", sa.JSON(), nullable=True))

        # Revert column renames
        batch_op.alter_column("start_time", new_column_name="started_at")
        batch_op.alter_column("end_time", new_column_name="completed_at")

        # Revert column types and constraints
        batch_op.alter_column(
            "job_type",
            existing_type=sa.String(50),
            type_=sa.String(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "status",
            existing_type=sa.String(20),
            type_=sa.String(),
            existing_nullable=False,
            nullable=True,
        )
        batch_op.alter_column(
            "started_at",
            existing_type=sa.DateTime(),
            nullable=True,
            server_default=None,
        )
