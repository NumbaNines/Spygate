"""Database session configuration."""

from sqlalchemy.orm import scoped_session, sessionmaker

# Create session factory
session_factory = sessionmaker()

# Create scoped session
Session = scoped_session(session_factory)


# Ensure we have a remove method
def cleanup():
    """Clean up any active sessions."""
    Session.remove()
