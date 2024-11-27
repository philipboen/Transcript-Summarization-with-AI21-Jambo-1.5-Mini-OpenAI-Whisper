import os
import uuid
from sqlalchemy import create_engine, Column, String, Text, ARRAY, FLOAT, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from sqlalchemy_utils import database_exists, create_database
from dotenv import load_dotenv
from sqlalchemy.orm import relationship

load_dotenv()

# Database credentials
database_url = os.getenv("DB_URL")

# Create the engine
engine = create_engine(database_url)

# Check and create database
if not database_exists(engine.url):
    create_database(engine.url)

# Session local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define models
Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    v_id = Column(String(50), nullable=False)
    v_url = Column(String(255), nullable=False)
    transcript = Column(Text)
    status = Column(String(20), default='pending')  # pending, processing, completed, error

class Audio(Base):
    __tablename__ = 'audios'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    status = Column(String(20), default='pending')  # pending, processing, completed, error
    transcript = Column(Text)
    chunks = relationship("AudioChunk", back_populates="audio")

class AudioChunk(Base):
    __tablename__ = "audio_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audio_id = Column(UUID(as_uuid=True), ForeignKey("audios.id"))
    chunk_text = Column(Text)
    embedding = Column(ARRAY(FLOAT))
    audio = relationship("Audio", back_populates="chunks")

# Ensure the vector extension is enabled
with engine.begin() as connection:
    connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))

try:
    # Create tables
    Base.metadata.create_all(engine)
except Exception as e:
    print(f"Error creating tables: {e}")
