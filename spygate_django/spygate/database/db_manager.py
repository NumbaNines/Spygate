import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for motion detection system."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self.conn = None
        self.cursor = None

        # Initialize database
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database connection and create necessary tables."""
        try:
            # Create database directory if needed
            if self.db_path != ":memory:":
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

            # Create tables
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS motion_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    motion_detected BOOLEAN NOT NULL,
                    bounding_boxes TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """
            )

            # Create indices
            self.cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_motion_data_timestamp
                ON motion_data(timestamp)
            """
            )

            self.conn.commit()
            logger.info(f"Initialized database at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def insert_motion_data(self, data: dict[str, Any]) -> None:
        """Insert motion detection result into database.

        Args:
            data: Dictionary containing motion detection data
        """
        try:
            self.cursor.execute(
                """
                INSERT INTO motion_data
                (timestamp, motion_detected, bounding_boxes, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    data["timestamp"],
                    data["motion_detected"],
                    data["bounding_boxes"],
                    data["metadata"],
                ),
            )
            self.conn.commit()

        except Exception as e:
            logger.error(f"Failed to insert motion data: {str(e)}")
            self.conn.rollback()
            raise

    def get_motion_data(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        motion_only: bool = False,
    ) -> list:
        """Retrieve motion detection results from database.

        Args:
            start_time: ISO format timestamp to start from
            end_time: ISO format timestamp to end at
            motion_only: If True, only return records with motion_detected=True

        Returns:
            List of motion detection records
        """
        try:
            query = "SELECT * FROM motion_data WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if motion_only:
                query += " AND motion_detected = 1"

            query += " ORDER BY timestamp DESC"

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()

            # Convert rows to dictionaries with parsed JSON
            results = []
            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "timestamp": row[1],
                        "motion_detected": bool(row[2]),
                        "bounding_boxes": json.loads(row[3]),
                        "metadata": json.loads(row[4]),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve motion data: {str(e)}")
            raise

    def clear_old_data(self, before_timestamp: str) -> int:
        """Delete motion detection records older than specified timestamp.

        Args:
            before_timestamp: ISO format timestamp

        Returns:
            Number of records deleted
        """
        try:
            self.cursor.execute("DELETE FROM motion_data WHERE timestamp < ?", (before_timestamp,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()

            logger.info(f"Deleted {deleted_count} old records")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear old data: {str(e)}")
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __del__(self):
        """Ensure database connection is closed on deletion."""
        self.close()
