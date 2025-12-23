"""
GuppShupp Database Initialization Script
Production-Grade Schema with Multi-Layer Memory System
Creates PostgreSQL database with pgvector extension
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (default 'postgres' database)
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"✅ Database '{DB_NAME}' created successfully")
        else:
            print(f"ℹ️  Database '{DB_NAME}' already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        sys.exit(1)


def setup_extensions():
    """Install required PostgreSQL extensions"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        # Install pgvector extension for semantic search
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✅ pgvector extension installed")
        
        # Install uuid extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        print("✅ uuid-ossp extension installed")
        
        # Install pgcrypto for future encryption needs
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        print("✅ pgcrypto extension installed")
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error setting up extensions: {e}")
        sys.exit(1)


def create_tables():
    """Create all required tables with production-grade schema"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        
        # ==========================================
        # TABLE 1: USERS
        # Purpose: Store user profiles and preferences
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE,
                
                -- Language preferences
                preferred_language VARCHAR(10) DEFAULT 'hi',
                
                -- User preferences (character settings, UI preferences, etc.)
                preferences JSONB DEFAULT '{}'::jsonb,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Account status
                is_active BOOLEAN DEFAULT TRUE
            );
            
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active DESC);
        """)
        print("✅ Table 'users' created")
        
        
        # ==========================================
        # TABLE 2: SESSIONS
        # Purpose: Group conversations into chat sessions
        # Supports: Short-term memory (context window)
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                
                -- Session timing
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_end TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                
                -- Session stats
                message_count INTEGER DEFAULT 0,
                avg_response_time_ms INTEGER,
                
                -- Session context (short-term memory storage)
                session_context JSONB DEFAULT '{}'::jsonb
            );
            
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);
            CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(session_start DESC);
        """)
        print("✅ Table 'sessions' created")
        
        
        # ==========================================
        # TABLE 3: CONVERSATIONS
        # Purpose: Core table - stores every conversation turn
        # Contains: Transcription, prosody features, emotions, embeddings
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                
                -- Conversation content
                user_input_text TEXT NOT NULL,
                ai_response_text TEXT NOT NULL,
                
                -- Language detection and handling
                detected_language VARCHAR(10),  -- Input language from Whisper
                response_language VARCHAR(10),  -- Output language chosen by Gemini
                is_code_mixed BOOLEAN DEFAULT FALSE,
                code_mix_languages VARCHAR(10)[],  -- Array: ['hi', 'en'] for Hinglish
                
                -- Emotion & sentiment analysis
                detected_emotion VARCHAR(50),  -- joy, sadness, anger, fear, surprise, disgust, neutral
                emotion_confidence FLOAT,
                sentiment VARCHAR(20),  -- positive, negative, neutral
                
                -- Intent classification
                detected_intent VARCHAR(100),  -- greeting, question, complaint, expressing_emotion, etc.
                intent_confidence FLOAT,
                
                -- Prosody features (librosa + OpenSMILE acoustic analysis)
                -- Stores the complete JSON with all acoustic features
                prosody_features JSONB,
                
                -- Audio metadata
                audio_duration_seconds FLOAT,
                audio_file_path TEXT,
                response_audio_path TEXT,
                
                -- TTS generation
                tts_prompt TEXT,  -- Emotional tone description for TTS (e.g., "warm, reassuring tone")
                
                -- Performance metrics
                response_generation_time_ms INTEGER,
                
                -- Safety & moderation
                safety_check_passed BOOLEAN DEFAULT TRUE,
                safety_flags JSONB,  -- Details of any safety violations
                
                -- Timestamp
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Semantic search embedding (IndicBERT 768-dimensional vector)
                embedding vector(768)
            );
            
            -- Indexes for fast retrieval
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversations_emotion ON conversations(detected_emotion);
            CREATE INDEX IF NOT EXISTS idx_conversations_language ON conversations(detected_language);
            CREATE INDEX IF NOT EXISTS idx_conversations_code_mixed ON conversations(is_code_mixed);
            
            -- Vector similarity search index (HNSW algorithm for fast nearest neighbor search)
            CREATE INDEX IF NOT EXISTS idx_conversations_embedding ON conversations 
            USING hnsw (embedding vector_cosine_ops);
        """)
        print("✅ Table 'conversations' created")
        
        
        # ==========================================
        # TABLE 4: MEMORIES
        # Purpose: Multi-layer memory system
        # Supports: Long-term facts, Episodic summaries, Semantic search
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                
                -- Memory content
                memory_text TEXT NOT NULL,
                
                -- Memory type (only long-term and episodic stored here)
                -- short-term memory = recent conversations queried from conversations table
                memory_type VARCHAR(20) NOT NULL,  -- 'long_term' or 'episodic'
                
                -- Memory categorization
                category VARCHAR(100),  -- 'personal', 'work', 'relationships', 'preferences', etc.
                
                -- Importance and prioritization
                importance_score FLOAT DEFAULT 0.5,  -- 0.0 to 1.0 scale
                decay_factor FLOAT DEFAULT 1.0,  -- Memory fading: starts at 1.0, decreases over time
                is_pinned BOOLEAN DEFAULT FALSE,  -- Critical memories (name, key facts) don't fade
                
                -- Emotional context
                emotional_tone VARCHAR(50),  -- Emotion associated with this memory
                emotional_intensity FLOAT,  -- 0.0 to 1.0
                
                -- Temporal information
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,  -- How many times retrieved (boosts importance)
                
                -- Source tracking
                source_conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
                
                -- Semantic embedding for similarity search (IndicBERT)
                embedding vector(768),
                
                -- Soft delete (don't hard-delete user memories)
                is_active BOOLEAN DEFAULT TRUE
            );
            
            -- Indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance_score DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_is_active ON memories(is_active);
            CREATE INDEX IF NOT EXISTS idx_memories_is_pinned ON memories(is_pinned);
            
            -- Vector search index for semantic similarity
            CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories 
            USING hnsw (embedding vector_cosine_ops);
        """)
        print("✅ Table 'memories' created")
        
        
        # ==========================================
        # TABLE 5: SAFETY_LOGS
        # Purpose: Audit trail for all safety events
        # Critical for: Production compliance, security monitoring
        # ==========================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
                
                -- Event classification
                event_type VARCHAR(50) NOT NULL,  -- 'jailbreak', 'self_harm', 'hate_speech', 'violence', etc.
                severity VARCHAR(20) NOT NULL,  -- 'low', 'medium', 'high', 'critical'
                
                -- Detection details
                layer VARCHAR(20),  -- 'layer_1' (input), 'layer_2' (content), 'layer_3' (output)
                matched_patterns TEXT[],  -- Array of patterns that triggered detection
                confidence_score FLOAT,
                
                -- Content reference (hashed for privacy)
                flagged_content_hash TEXT,
                
                -- Action taken
                action_taken VARCHAR(50),  -- 'blocked', 'warned', 'escalated', 'allowed_with_warning'
                
                -- Crisis intervention (self-harm, suicide detection)
                crisis_response_triggered BOOLEAN DEFAULT FALSE,
                helpline_provided VARCHAR(100),  -- Which helpline was shown to user
                
                -- Timestamp
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Additional metadata
                metadata JSONB
            );
            
            -- Indexes for security monitoring and auditing
            CREATE INDEX IF NOT EXISTS idx_safety_logs_user_id ON safety_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_safety_logs_event_type ON safety_logs(event_type);
            CREATE INDEX IF NOT EXISTS idx_safety_logs_severity ON safety_logs(severity);
            CREATE INDEX IF NOT EXISTS idx_safety_logs_detected_at ON safety_logs(detected_at DESC);
            CREATE INDEX IF NOT EXISTS idx_safety_logs_crisis ON safety_logs(crisis_response_triggered);
        """)
        print("✅ Table 'safety_logs' created")
        
        
        conn.commit()
        cursor.close()
        conn.close()
        print("\n🎉 All tables created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        sys.exit(1)


def verify_setup():
    """Verify that all tables and extensions are properly installed"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        # Check extensions
        cursor.execute("""
            SELECT extname 
            FROM pg_extension 
            WHERE extname IN ('vector', 'uuid-ossp', 'pgcrypto')
            ORDER BY extname
        """)
        extensions = cursor.fetchall()
        print(f"\n✅ Extensions installed: {[ext[0] for ext in extensions]}")
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        print(f"✅ Tables created: {[table[0] for table in tables]}")
        
        # Check vector indexes
        cursor.execute("""
            SELECT tablename, indexname 
            FROM pg_indexes 
            WHERE indexname LIKE '%embedding%'
            ORDER BY tablename
        """)
        vector_indexes = cursor.fetchall()
        print(f"✅ Vector search indexes: {len(vector_indexes)} created")
        
        # Count total indexes
        cursor.execute("""
            SELECT COUNT(*) 
            FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        total_indexes = cursor.fetchone()[0]
        print(f"✅ Total indexes created: {total_indexes}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error verifying setup: {e}")


def print_schema_summary():
    """Print summary of the database schema"""
    print("\n" + "=" * 70)
    print("DATABASE SCHEMA SUMMARY")
    print("=" * 70)
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    GUPPSHUPP DATABASE SCHEMA                      ║
║                      5 Tables - Production Ready                  ║
╚═══════════════════════════════════════════════════════════════════╝

📋 TABLE 1: users
   Purpose: User profiles and preferences
   Key Features: Account management, language preferences

📋 TABLE 2: sessions
   Purpose: Group conversations into chat sessions
   Key Features: Short-term memory context, session tracking

📋 TABLE 3: conversations (CORE TABLE)
   Purpose: Every conversation turn with full context
   Key Features:
   - Transcription (user input + AI response)
   - Language detection (input + output languages)
   - Code-mixing support
   - Prosody features (librosa + OpenSMILE acoustic data)
   - Emotion detection (joy, sadness, anger, fear, etc.)
   - Intent classification
   - TTS prompt (emotional tone for voice synthesis)
   - Semantic embeddings (vector search)
   - Safety flags

📋 TABLE 4: memories (MULTI-LAYER MEMORY)
   Purpose: Long-term facts and episodic summaries
   Key Features:
   - memory_type: 'long_term' | 'episodic'
   - Importance scoring (0-1)
   - Memory decay (fading over time)
   - Pinned memories (never fade)
   - Semantic embeddings (similarity search)
   - Access tracking
   
   Memory Layers:
   → Short-term: Query recent conversations from sessions
   → Long-term: Extracted facts stored here
   → Episodic: Conversation summaries stored here

📋 TABLE 5: safety_logs
   Purpose: Security audit trail
   Key Features:
   - 3-layer guardrail tracking
   - Crisis intervention logging
   - Severity classification
   - Action tracking

═══════════════════════════════════════════════════════════════════

✨ SEMANTIC SEARCH ENABLED
   - conversations.embedding (768-dim vectors)
   - memories.embedding (768-dim vectors)
   - HNSW indexes for fast similarity search

✨ PRODUCTION FEATURES
   - pgvector for semantic search
   - JSONB for flexible data (prosody, preferences, context)
   - Comprehensive indexing for performance
   - Foreign key constraints for data integrity
   - Soft deletes (is_active flags)
   - Audit trails (safety_logs)

═══════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    print("=" * 70)
    print("GUPPSHUPP DATABASE INITIALIZATION")
    print("=" * 70)
    print()
    
    # Step 1: Create database
    print("Step 1: Creating database...")
    create_database()
    print()
    
    # Step 2: Setup extensions
    print("Step 2: Installing PostgreSQL extensions...")
    setup_extensions()
    print()
    
    # Step 3: Create tables
    print("Step 3: Creating tables with production-grade schema...")
    create_tables()
    print()
    
    # Step 4: Verify setup
    print("Step 4: Verifying setup...")
    verify_setup()
    print()
    
    # Step 5: Print schema summary
    print_schema_summary()
    
    print("=" * 70)
    print("✅ DATABASE INITIALIZATION COMPLETE!")
    print("=" * 70)
    print("\n🚀 Ready to build GuppShupp!")
    print("   Next step: Run 'python backend/config.py' to configure services\n")
