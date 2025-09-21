# db.py - FIXED VERSION
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
import logging

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Enhanced SQLite configuration for better reliability
engine = create_engine(
    "sqlite:///data/evaluations.db",
    echo=False,
    future=True,
    poolclass=StaticPool,
    connect_args={
        "check_same_thread": False,
        "timeout": 20
    },
    pool_pre_ping=True,
    pool_recycle=3600
)

# Fixed session configuration
SessionLocal = sessionmaker(
    bind=engine, 
    autoflush=False, 
    autocommit=False, 
    future=True,
    expire_on_commit=False
)

Base = declarative_base()

# Add logging for database operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db():
    """Proper database session management"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
