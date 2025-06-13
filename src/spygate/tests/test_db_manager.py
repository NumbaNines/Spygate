import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from spygate.database.db_manager import DatabaseManager


def create_test_data():
    """Create sample motion detection data for testing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "motion_detected": True,
        "bounding_boxes": json.dumps([[100, 100, 200, 200]]),
        "metadata": json.dumps({"confidence": 0.95}),
    }


def test_db_initialization():
    """Test database initialization."""
    # Test in-memory database
    db = DatabaseManager()
    assert db.db_path == ":memory:"
    assert db.conn is not None
    assert db.cursor is not None

    # Test file-based database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "test.db")
        db = DatabaseManager(db_path)
        assert db.db_path == db_path
        assert Path(db_path).exists()

        # Test table creation
        db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in db.cursor.fetchall()]
        assert "motion_data" in tables

        # Test index creation
        db.cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = [row[0] for row in db.cursor.fetchall()]
        assert "idx_motion_data_timestamp" in indices


def test_data_insertion():
    """Test inserting motion detection data."""
    db = DatabaseManager()
    data = create_test_data()

    # Test single insertion
    db.insert_motion_data(data)

    # Verify insertion
    db.cursor.execute("SELECT COUNT(*) FROM motion_data")
    count = db.cursor.fetchone()[0]
    assert count == 1

    # Test multiple insertions
    for _ in range(5):
        db.insert_motion_data(data)

    db.cursor.execute("SELECT COUNT(*) FROM motion_data")
    count = db.cursor.fetchone()[0]
    assert count == 6


def test_data_retrieval():
    """Test retrieving motion detection data."""
    db = DatabaseManager()

    # Insert test data with different timestamps
    base_time = datetime.now()
    for i in range(5):
        data = create_test_data()
        data["timestamp"] = (base_time + timedelta(minutes=i)).isoformat()
        data["motion_detected"] = i % 2 == 0  # Alternate between True/False
        db.insert_motion_data(data)

    # Test retrieving all data
    results = db.get_motion_data()
    assert len(results) == 5

    # Test time range filtering
    start_time = (base_time + timedelta(minutes=1)).isoformat()
    end_time = (base_time + timedelta(minutes=3)).isoformat()
    results = db.get_motion_data(start_time, end_time)
    assert len(results) == 3

    # Test motion-only filtering
    results = db.get_motion_data(motion_only=True)
    assert len(results) == 3
    assert all(r["motion_detected"] for r in results)

    # Verify data format
    result = results[0]
    assert isinstance(result["id"], int)
    assert isinstance(result["timestamp"], str)
    assert isinstance(result["motion_detected"], bool)
    assert isinstance(result["bounding_boxes"], list)
    assert isinstance(result["metadata"], dict)


def test_clear_old_data():
    """Test clearing old motion detection data."""
    db = DatabaseManager()
    base_time = datetime.now()

    # Insert test data with different timestamps
    for i in range(5):
        data = create_test_data()
        data["timestamp"] = (base_time + timedelta(minutes=i)).isoformat()
        db.insert_motion_data(data)

    # Clear data older than 3 minutes
    cutoff_time = (base_time + timedelta(minutes=3)).isoformat()
    deleted_count = db.clear_old_data(cutoff_time)
    assert deleted_count == 3

    # Verify remaining data
    results = db.get_motion_data()
    assert len(results) == 2

    # Verify timestamps of remaining data
    for result in results:
        assert result["timestamp"] >= cutoff_time


def test_error_handling():
    """Test error handling in database operations."""
    db = DatabaseManager()

    # Test invalid data insertion
    with pytest.raises(Exception):
        db.insert_motion_data({})  # Missing required fields

    # Test invalid query parameters
    with pytest.raises(Exception):
        db.get_motion_data(start_time="invalid_timestamp")

    # Test database closure
    db.close()
    assert db.conn is None
    assert db.cursor is None

    # Test operations after closure
    with pytest.raises(Exception):
        db.insert_motion_data(create_test_data())
