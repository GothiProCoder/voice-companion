"""
SQLAlchemy ORM Models
Defines all database tables as Python classes
Matches the schema created in scripts/init_db.py
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, TIMESTAMP, 
    ForeignKey, ARRAY, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

from backend.database.database import Base


class User(Base):
    """
    User profiles and preferences
    """
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    
    # Language preferences
    preferred_language = Column(String(10), default='hi')
    
    # User preferences (character settings, UI preferences, etc.)
    preferences = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=func.now())
    last_active = Column(TIMESTAMP, default=func.now(), onupdate=func.now())
    
    # Account status
    is_active = Column(Boolean, default=True)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    safety_logs = relationship("SafetyLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, language={self.preferred_language})>"


class Session(Base):
    """
    Conversation sessions - groups multiple conversation turns
    Supports short-term memory (context window)
    """
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Session timing
    session_start = Column(TIMESTAMP, default=func.now())
    session_end = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    
    # Session stats
    message_count = Column(Integer, default=0)
    avg_response_time_ms = Column(Integer, nullable=True)
    
    # Session context (short-term memory storage)
    session_context = Column(JSONB, default={})
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class Conversation(Base):
    """
    Core table - stores every conversation turn
    Contains: transcription, prosody features, emotions, embeddings
    """
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Conversation content
    user_input_text = Column(Text, nullable=False)
    ai_response_text = Column(Text, nullable=False)
    
    # Language detection and handling
    detected_language = Column(String(10), index=True)  # Input language from Whisper
    response_language = Column(String(10))  # Output language chosen by Gemini
    is_code_mixed = Column(Boolean, default=False, index=True)
    code_mix_languages = Column(ARRAY(String(10)))  # Array: ['hi', 'en'] for Hinglish
    
    # Emotion & sentiment analysis
    detected_emotion = Column(String(50), index=True)  # joy, sadness, anger, fear, etc.
    emotion_confidence = Column(Float)
    sentiment = Column(String(20))  # positive, negative, neutral
    
    # Intent classification
    detected_intent = Column(String(100))  # greeting, question, complaint, etc.
    intent_confidence = Column(Float)
    
    # Prosody features (librosa + OpenSMILE acoustic analysis)
    prosody_features = Column(JSONB)  # Stores complete acoustic features JSON
    
    # Audio metadata
    audio_duration_seconds = Column(Float)
    audio_file_path = Column(Text)
    response_audio_path = Column(Text)
    
    # TTS generation
    tts_prompt = Column(Text)  # Emotional tone description for TTS
    
    # Performance metrics
    response_generation_time_ms = Column(Integer)
    
    # Safety & moderation
    safety_check_passed = Column(Boolean, default=True)
    safety_flags = Column(JSONB)  # Details of any safety violations
    
    # Timestamp
    created_at = Column(TIMESTAMP, default=func.now(), index=True)
    
    # Semantic search embedding (IndicBERT 768-dimensional vector)
    embedding = Column(Vector(768))
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    session = relationship("Session", back_populates="conversations")
    safety_logs = relationship("SafetyLog", back_populates="conversation")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversations_embedding', 'embedding', postgresql_using='hnsw', 
              postgresql_with={'m': 16, 'ef_construction': 64}, 
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, emotion={self.detected_emotion}, language={self.detected_language})>"


class Memory(Base):
    """
    Multi-layer memory system
    Supports: Long-term facts, Episodic summaries, Semantic search
    """
    __tablename__ = "memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Memory content
    memory_text = Column(Text, nullable=False)
    
    # Memory type: 'long_term' or 'episodic'
    # short_term memory = recent conversations queried from conversations table
    memory_type = Column(String(20), nullable=False, index=True)
    
    # Memory categorization
    category = Column(String(100), index=True)  # personal, work, relationships, preferences
    
    # Importance and prioritization
    importance_score = Column(Float, default=0.5, index=True)  # 0.0 to 1.0 scale
    decay_factor = Column(Float, default=1.0)  # Memory fading: starts at 1.0, decreases over time
    is_pinned = Column(Boolean, default=False, index=True)  # Critical memories don't fade
    
    # Emotional context
    emotional_tone = Column(String(50))  # Emotion associated with memory
    emotional_intensity = Column(Float)  # 0.0 to 1.0
    
    # Temporal information
    created_at = Column(TIMESTAMP, default=func.now(), index=True)
    last_accessed = Column(TIMESTAMP, default=func.now(), onupdate=func.now(), index=True)
    access_count = Column(Integer, default=0)  # How many times retrieved
    
    # Source tracking
    source_conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    
    # Semantic embedding for similarity search (IndicBERT)
    embedding = Column(Vector(768))
    
    # Soft delete
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    user = relationship("User", back_populates="memories")
    
    # Indexes
    __table_args__ = (
        Index('idx_memories_embedding', 'embedding', postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
    
    def __repr__(self):
        return f"<Memory(id={self.id}, type={self.memory_type}, category={self.category}, importance={self.importance_score})>"


class SafetyLog(Base):
    """
    Audit trail for all safety events
    Critical for: Production compliance, security monitoring
    """
    __tablename__ = "safety_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    
    # Event classification
    event_type = Column(String(50), nullable=False, index=True)  # jailbreak, self_harm, hate_speech, etc.
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    
    # Detection details
    layer = Column(String(20))  # layer_1 (input), layer_2 (content), layer_3 (output)
    matched_patterns = Column(ARRAY(Text))  # Patterns that triggered detection
    confidence_score = Column(Float)
    
    # Content reference (hashed for privacy)
    flagged_content_hash = Column(Text)
    
    # Action taken
    action_taken = Column(String(50))  # blocked, warned, escalated, allowed_with_warning
    
    # Crisis intervention (self-harm, suicide detection)
    crisis_response_triggered = Column(Boolean, default=False, index=True)
    helpline_provided = Column(String(100))  # Which helpline was shown
    
    # Timestamp
    detected_at = Column(TIMESTAMP, default=func.now(), index=True)
    
    # Additional data (renamed from 'metadata' - reserved by SQLAlchemy)
    extra_data = Column(JSONB)
    
    # Relationships
    user = relationship("User", back_populates="safety_logs")
    conversation = relationship("Conversation", back_populates="safety_logs")
    
    def __repr__(self):
        return f"<SafetyLog(id={self.id}, event={self.event_type}, severity={self.severity})>"


if __name__ == "__main__":
    """Test model definitions"""
    print("=" * 70)
    print("DATABASE MODELS TEST")
    print("=" * 70)
    
    from backend.database.database import init_database, create_tables
    
    # Initialize database
    init_database()
    
    # Create tables from models
    print("\nCreating tables from ORM models...")
    create_tables()
    
    print("\n✅ All models defined successfully!")
    print("\n📋 Defined Models:")
    print("   1. User")
    print("   2. Session")
    print("   3. Conversation (with vector embeddings)")
    print("   4. Memory (with vector embeddings)")
    print("   5. SafetyLog")
    
    print("=" * 70)
