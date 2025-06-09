"""Add player support to video schema

Revision ID: player_support_update
Revises: video_storage_update
Create Date: 2024-03-19 10:30:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "player_support_update"
down_revision: Union[str, None] = "video_storage_update"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add player support to video schema."""
    # Create players table
    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("team", sa.String(100)),
        sa.Column("is_self", sa.Boolean(), default=False),
        sa.Column("gamertag", sa.String(100)),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create video_players association table
    op.create_table(
        "video_players",
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), default=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("video_id", "player_id"),
    )

    # Create default "Self" player
    op.execute(
        """
        INSERT INTO players (name, team, is_self, created_at, updated_at)
        VALUES ('Self', NULL, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """
    )

    # Migrate existing player_name data to new structure
    op.execute(
        """
        INSERT INTO video_players (video_id, player_id, is_primary)
        SELECT v.id,
               CASE
                   WHEN v.player_name = 'Self' THEN 1
                   ELSE (
                       INSERT INTO players (name, is_self, created_at, updated_at)
                       VALUES (v.player_name, FALSE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                       RETURNING id
                   )
               END,
               TRUE
        FROM videos v
        WHERE v.player_name IS NOT NULL
    """
    )

    # Remove old player_name column
    op.drop_column("videos", "player_name")


def downgrade() -> None:
    """Revert player support changes."""
    # Add back player_name column
    op.add_column("videos", sa.Column("player_name", sa.String(100)))

    # Migrate data back to player_name column
    op.execute(
        """
        UPDATE videos v
        SET player_name = (
            SELECT p.name
            FROM video_players vp
            JOIN players p ON p.id = vp.player_id
            WHERE vp.video_id = v.id AND vp.is_primary = TRUE
            LIMIT 1
        )
    """
    )

    # Make player_name not nullable
    op.alter_column("videos", "player_name", nullable=False)

    # Drop association table and players table
    op.drop_table("video_players")
    op.drop_table("players")
