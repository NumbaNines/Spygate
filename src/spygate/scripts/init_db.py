"""Initialize the database with a default user."""

from spygate.database import User, get_db, hash_password, init_db


def create_default_user():
    """Create a default user if none exists."""
    db = get_db()

    # Check if any users exist
    if db.query(User).first() is None:
        # Create default user
        default_user = User(
            username="admin",
            password_hash=hash_password("admin"),
            email="admin@spygate.local",
            is_active=True,
        )
        db.add(default_user)
        db.commit()
        print("Created default user: admin/admin")
    else:
        print("Users already exist, skipping default user creation")


if __name__ == "__main__":
    # Initialize database schema
    init_db()
    # Create default user
    create_default_user()
