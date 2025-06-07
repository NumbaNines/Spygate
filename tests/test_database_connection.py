import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from database.config import DatabaseSession, get_database_url
from database.models import Base, Player


def test_database_connection():
    """Test database connection and basic operations."""
    try:
        # Create a session
        session = DatabaseSession()

        # Try to create a test player
        test_player = Player(name="Test Player")
        session.add(test_player)
        session.commit()

        # Query the player back
        queried_player = session.query(Player).filter_by(name="Test Player").first()
        print(f"Database connection successful!")
        print(f"Database URL: {get_database_url()}")
        print(f"Test player created with ID: {queried_player.id}")

        # Clean up
        session.delete(queried_player)
        session.commit()
        session.close()

        return True
    except Exception as e:
        print(f"Error testing database connection: {str(e)}")
        return False


if __name__ == "__main__":
    test_database_connection()
