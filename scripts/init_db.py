#!/usr/bin/env python
"""Initialize the database with a default user."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spygate.database.config import get_db, init_db
from spygate.database.models import User
from spygate.database.utils import hash_password


def create_default_user():
    """Create a default user for testing."""
    with get_db() as session:
        # Check if default user exists
        default_user = session.query(User).filter(User.username == "admin").first()
        if not default_user:
            # Create default user
            default_user = User(
                username="admin",
                password_hash=hash_password("admin"),
                email="admin@example.com",
                is_active=True,
            )
            session.add(default_user)
            session.commit()
            print("Created default user 'admin' with password 'admin'")
        else:
            print("Default user 'admin' already exists")


def main():
    """Initialize the database and create default user."""
    print("Initializing database...")
    init_db()
    print("Database initialized")

    print("Creating default user...")
    create_default_user()
    print("Setup complete!")


if __name__ == "__main__":
    main()
