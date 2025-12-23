"""
Database Connection and Session Management
Handles PostgreSQL connection with pgvector support
Provides session management for all database operations
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from backend.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create declarative base for ORM models
Base = declarative_base()

# Database engine configuration
engine = None
SessionLocal = None


def init_database():
    """
    Initialize database engine and session factory
    Called once at application startup
    """
    global engine, SessionLocal
    
    try:
        # Create SQLAlchemy engine with connection pooling
        engine = create_engine(
            config.database.url,
            poolclass=QueuePool,
            pool_size=10,  # Number of permanent connections
            max_overflow=20,  # Additional connections when pool is full
            pool_timeout=30,  # Timeout for getting connection from pool
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Verify connections before using
            echo=config.is_development(),  # Log SQL queries in development
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection established")
        
        # Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        
        logger.info("✅ Database session factory created")
        
        # Register pgvector extension handler
        register_vector_type()
        
        # Register engine event listeners (AFTER engine is created)
        _register_engine_events()
        
        return engine
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def register_vector_type():
    """
    Register pgvector custom type handler
    Allows SQLAlchemy to work with vector columns
    """
    from pgvector.sqlalchemy import Vector
    logger.info("✅ pgvector type handler registered")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI routes
    Provides database session with automatic cleanup
    
    Usage in FastAPI:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            # Use db here
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions
    Use when not in FastAPI route context
    
    Usage:
        with get_db_session() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()


def create_tables():
    """
    Create all tables defined in models
    Should only be used in development - use Alembic migrations in production
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        raise


def drop_tables():
    """
    Drop all tables
    WARNING: Use with extreme caution - deletes all data
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("⚠️  All tables dropped")
    except Exception as e:
        logger.error(f"❌ Failed to drop tables: {e}")
        raise


def check_database_connection() -> bool:
    """
    Check if database connection is working
    Returns True if connection is healthy, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"❌ Database connection check failed: {e}")
        return False


def get_table_count() -> dict:
    """
    Get count of records in each table
    Useful for monitoring and debugging
    """
    from backend.database.models import User, Session as SessionModel, Conversation, Memory, SafetyLog
    
    with get_db_session() as db:
        return {
            "users": db.query(User).count(),
            "sessions": db.query(SessionModel).count(),
            "conversations": db.query(Conversation).count(),
            "memories": db.query(Memory).count(),
            "safety_logs": db.query(SafetyLog).count(),
        }


# Event listeners are registered inside init_database() after engine is created
# DO NOT put @event.listens_for decorators at module level - engine is None!


def _register_engine_events():
    """
    Register event listeners for connection pool management.
    Called after engine is created in init_database().
    """
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Called when a new database connection is created"""
        logger.debug("New database connection established")

    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        """Called when a connection is retrieved from the pool"""
        logger.debug("Connection checked out from pool")

    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn, connection_record):
        """Called when a connection is returned to the pool"""
        logger.debug("Connection checked back into pool")


if __name__ == "__main__":
    """Test database connection"""
    print("=" * 70)
    print("DATABASE CONNECTION TEST")
    print("=" * 70)
    
    # Initialize database
    init_database()
    
    # Check connection
    if check_database_connection():
        print("✅ Database connection successful")
        
        # Print table counts
        try:
            counts = get_table_count()
            print("\n📊 Table Record Counts:")
            for table, count in counts.items():
                print(f"   {table}: {count}")
        except Exception as e:
            print(f"⚠️  Could not fetch table counts (tables may not exist yet): {e}")
    else:
        print("❌ Database connection failed")
    
    print("=" * 70)
