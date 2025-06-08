import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from .database import Base

load_dotenv()


def get_database_url() -> str:
    """Get database URL from environment variables"""
    db_type = os.getenv("DB_TYPE", "sqlite")

    if db_type == "postgresql":
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "spygate")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    else:
        # SQLite default
        db_path = os.getenv("DB_PATH", "spygate.db")
        return f"sqlite:///{db_path}"


class Database:
    _instance: Optional["Database"] = None
    _engine: Optional[Engine] = None
    _SessionLocal: Optional[sessionmaker] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize database connection and session factory"""
        if self._engine is None:
            db_url = get_database_url()

            # Configure engine based on database type
            if db_url.startswith("sqlite"):
                self._engine = create_engine(
                    db_url,
                    connect_args={"check_same_thread": False},
                    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
                )
            else:
                self._engine = create_engine(
                    db_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
                )

            # Create session factory
            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

            # Create tables
            Base.metadata.create_all(bind=self._engine)

    def get_session(self):
        """Get a new database session"""
        if self._SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return self._SessionLocal()

    def dispose(self):
        """Dispose of the current engine and all its database connections"""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None


# Create global instance
db = Database()
