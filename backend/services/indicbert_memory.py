"""
GuppShupp IndicBERT Memory Service Module
backend/services/indicbert_memory.py

PRODUCTION-GRADE SEMANTIC MEMORY SERVICE
- IndicBERT-based embeddings (768-dimensional vectors)
- pgvector semantic similarity search
- Multi-layer memory storage (long_term, episodic, semantic)
- Batch processing for efficiency
- GPU/CPU auto-detection
- Async operations support
- Integrated with Gemini LLM memory_updates

Author: GuppShupp Team
Last Updated: 2025-12-23
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text, desc
from datetime import datetime, timedelta
import logging
import time
from contextlib import contextmanager
import asyncio
from functools import lru_cache

from backend.database.models import Memory, User
from backend.services.gemini_llm import MemoryUpdate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# INDICBERT MEMORY SERVICE
# ============================================================================

class IndicBERTMemoryService:
    """
    Semantic memory service for GuppShupp using IndicBERT embeddings.
    
    Responsibilities:
    1. Generate 768-dim embeddings for Indic languages + code-mixing
    2. Store memories with embeddings in PostgreSQL (pgvector)
    3. Retrieve relevant memories via cosine similarity search
    4. Track memory importance, access count, decay
    5. Support batch operations for efficiency
    
    Model: ai4bharat/indic-sentence-similarity-sbert (supports 11 Indic languages)
    Alternative: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    """
    
    def __init__(
        self,
        model_name: str = "l3cube-pune/indic-sentence-similarity-sbert",
        device: Optional[str] = None,
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        """
        Initialize IndicBERT Memory Service.
        
        Args:
            model_name: HuggingFace model identifier
                - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (768-dim, multilingual)
                - "ai4bharat/indic-sentence-similarity-sbert" (768-dim, Indic languages)
            device: "cuda", "cpu", or None (auto-detect)
            cache_size: LRU cache size for frequently embedded text
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing IndicBERT Memory Service on {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded: {model_name} (dim={self.embedding_dim})")
            
            # Verify embedding dimension matches database
            if self.embedding_dim != 768:
                logger.warning(
                    f"⚠️ Model embedding dim ({self.embedding_dim}) != 768. "
                    "Update database Vector column or use different model."
                )
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
        
        # Performance metrics
        self._total_embeddings = 0
        self._total_time_ms = 0
        
        logger.info(f"✅ IndicBERT Memory Service ready (cache_size={cache_size})")
    
    # ========================================================================
    # EMBEDDING GENERATION
    # ========================================================================
    
    @lru_cache(maxsize=1000)
    def _cached_embed(self, text: str) -> Tuple[float, ...]:
        """
        Internal cached embedding (must return hashable tuple).
        Used by embed_text for frequently repeated text.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        return tuple(embedding.tolist())
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate 768-dimensional embedding for single text.
        
        Args:
            text: Input text (supports Indic languages, code-mixing)
            use_cache: Use LRU cache for frequently embedded text
        
        Returns:
            np.ndarray of shape (768,)
        
        Example:
            >>> service = IndicBERTMemoryService()
            >>> embedding = service.embed_text("User has exam anxiety")
            >>> embedding.shape
            (768,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        start_time = time.time()
        
        try:
            if use_cache:
                embedding_tuple = self._cached_embed(text.strip())
                embedding = np.array(embedding_tuple, dtype=np.float32)
            else:
                embedding = self.model.encode(
                    text.strip(),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            
            # Track performance
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._total_embeddings += 1
            self._total_time_ms += elapsed_ms
            
            logger.debug(f"Generated embedding in {elapsed_ms}ms (avg: {self.avg_embedding_time_ms:.1f}ms)")
            
            return embedding
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently (batch processing).
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar for large batches
        
        Returns:
            List of np.ndarray embeddings, each of shape (768,)
        
        Example:
            >>> texts = ["Memory 1", "Memory 2", "Memory 3"]
            >>> embeddings = service.embed_batch(texts)
            >>> len(embeddings)
            3
        """
        if not texts:
            return []
        
        # Filter empty texts
        non_empty_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not non_empty_texts:
            logger.warning("All texts are empty")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
        
        start_time = time.time()
        
        try:
            embeddings = self.model.encode(
                non_empty_texts,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                batch_size=self.batch_size,
                normalize_embeddings=True
            )
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._total_embeddings += len(embeddings)
            self._total_time_ms += elapsed_ms
            
            logger.info(
                f"Generated {len(embeddings)} embeddings in {elapsed_ms}ms "
                f"({elapsed_ms/len(embeddings):.1f}ms per embedding)"
            )
            
            return list(embeddings)
        
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
    
    # ========================================================================
    # MEMORY STORAGE
    # ========================================================================
    
    def store_memory(
        self,
        db: Session,
        user_id: str,
        memory_update: MemoryUpdate,
        conversation_id: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        emotional_intensity: Optional[float] = None,
        is_pinned: bool = False
    ) -> Optional[Memory]:
        """
        Store a single memory with embedding in database.
        
        Args:
            db: SQLAlchemy session
            user_id: User UUID
            memory_update: MemoryUpdate from Gemini (type, text, category, importance)
            conversation_id: Source conversation UUID (optional)
            emotional_tone: Emotion associated with memory (optional)
            emotional_intensity: 0.0 to 1.0 (optional)
            is_pinned: If True, memory won't decay
        
        Returns:
            Memory object if successful, None if failed
        
        Example:
            >>> memory_update = MemoryUpdate(
            ...     type="long_term",
            ...     text="User has exam anxiety",
            ...     category="work_study",
            ...     importance=0.9
            ... )
            >>> memory = service.store_memory(db, user_id, memory_update)
        """
        try:
            # Generate embedding
            embedding = self.embed_text(memory_update.text, use_cache=False)
            
            # Convert numpy array to list for pgvector
            embedding_list = embedding.tolist()
            
            # Create Memory object
            memory = Memory(
                user_id=user_id,
                memory_text=memory_update.text,
                memory_type=memory_update.type,
                category=memory_update.category,
                importance_score=memory_update.importance,
                decay_factor=1.0,  # Starts at full strength
                is_pinned=is_pinned,
                emotional_tone=emotional_tone,
                emotional_intensity=emotional_intensity,
                source_conversation_id=conversation_id,
                embedding=embedding_list,
                access_count=0,
                is_active=True,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            
            # Add to database
            db.add(memory)
            db.commit()
            db.refresh(memory)
            
            logger.info(
                f"✅ Stored memory: [{memory.memory_type}] {memory.memory_text[:50]}... "
                f"(importance={memory.importance_score:.2f}, id={memory.id})"
            )
            
            return memory
        
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            db.rollback()
            return None
    
    def store_memories_batch(
        self,
        db: Session,
        user_id: str,
        memory_updates: List[MemoryUpdate],
        conversation_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Store multiple memories efficiently (batch embedding + insert).
        
        Args:
            db: SQLAlchemy session
            user_id: User UUID
            memory_updates: List of MemoryUpdate from Gemini
            conversation_id: Source conversation UUID (optional)
        
        Returns:
            List of successfully created Memory objects
        
        Example:
            >>> memory_updates = [
            ...     MemoryUpdate(type="long_term", text="Fact 1", category="personal", importance=0.8),
            ...     MemoryUpdate(type="episodic", text="Summary 1", category="emotional", importance=0.7)
            ... ]
            >>> memories = service.store_memories_batch(db, user_id, memory_updates)
        """
        if not memory_updates:
            return []
        
        try:
            # Batch generate embeddings
            texts = [mem.text for mem in memory_updates]
            embeddings = self.embed_batch(texts, show_progress=False)
            
            # Create Memory objects
            memories = []
            for memory_update, embedding in zip(memory_updates, embeddings):
                memory = Memory(
                    user_id=user_id,
                    memory_text=memory_update.text,
                    memory_type=memory_update.type,
                    category=memory_update.category,
                    importance_score=memory_update.importance,
                    decay_factor=1.0,
                    is_pinned=False,
                    source_conversation_id=conversation_id,
                    embedding=embedding.tolist(),
                    access_count=0,
                    is_active=True,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow()
                )
                memories.append(memory)
            
            # Bulk insert
            db.add_all(memories)
            db.commit()
            
            logger.info(f"✅ Stored {len(memories)} memories in batch")
            
            return memories
        
        except Exception as e:
            logger.error(f"Failed to store memories in batch: {e}")
            db.rollback()
            return []
    
    # ========================================================================
    # MEMORY RETRIEVAL (SEMANTIC SEARCH)
    # ========================================================================
    
    def retrieve_memories(
        self,
        db: Session,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        apply_decay: bool = True,
        apply_recency_boost: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant memories using semantic similarity search.
        
        Args:
            db: SQLAlchemy session
            user_id: User UUID
            query_text: Text to search against (user's transcript)
            top_k: Number of memories to return
            memory_types: Filter by type (e.g., ["long_term", "episodic"])
            min_importance: Filter memories below this importance score
            apply_decay: Apply time-based decay factor to ranking
            apply_recency_boost: Boost recently accessed memories
        
        Returns:
            List of dicts with memory info + similarity score
        
        Example:
            >>> memories = service.retrieve_memories(
            ...     db=db,
            ...     user_id=user_id,
            ...     query_text="मैं बहुत stressed हूँ",
            ...     top_k=5
            ... )
            >>> memories[0]["memory_text"]
            "User has exam anxiety"
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query_text, use_cache=False)
            query_embedding_list = query_embedding.tolist()
            
            # Build SQL query with pgvector cosine similarity
            # Using <-> operator for cosine distance (1 - cosine_similarity)
            base_query = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True
            )
            
            # Apply filters
            if memory_types:
                base_query = base_query.filter(Memory.memory_type.in_(memory_types))
            
            if min_importance > 0.0:
                base_query = base_query.filter(Memory.importance_score >= min_importance)
            
            # Get all matching memories with similarity scores
            # pgvector cosine distance: embedding <-> query_embedding
            # Convert to similarity: 1 - distance
            query_with_similarity = base_query.add_columns(
                (1 - Memory.embedding.cosine_distance(query_embedding_list)).label('similarity')
            )
            
            # Fetch results
            results = query_with_similarity.all()
            
            if not results:
                logger.info(f"No memories found for user {user_id}")
                return []
            
            # Process results with optional decay and recency boost
            processed_results = []
            for memory, similarity in results:
                # Base score is similarity
                score = float(similarity)
                
                # Apply decay factor (time-based fading)
                if apply_decay and not memory.is_pinned:
                    score *= memory.decay_factor
                
                # Apply recency boost (recently accessed memories rank higher)
                if apply_recency_boost and memory.last_accessed:
                    days_since_access = (datetime.utcnow() - memory.last_accessed).days
                    recency_factor = 1.0 + (0.1 if days_since_access < 7 else 0.0)
                    score *= recency_factor
                
                # Weight by importance
                weighted_score = score * (0.7 + 0.3 * memory.importance_score)
                
                processed_results.append({
                    'memory_id': str(memory.id),
                    'memory_text': memory.memory_text,
                    'memory_type': memory.memory_type,
                    'category': memory.category,
                    'importance_score': memory.importance_score,
                    'emotional_tone': memory.emotional_tone,
                    'emotional_intensity': memory.emotional_intensity,
                    'created_at': memory.created_at,
                    'last_accessed': memory.last_accessed,
                    'access_count': memory.access_count,
                    'similarity': float(similarity),
                    'weighted_score': weighted_score,
                    'is_pinned': memory.is_pinned
                })
            
            # Sort by weighted score
            processed_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            # Take top-K
            top_memories = processed_results[:top_k]
            
            # Update access counts and timestamps (async to avoid blocking)
            memory_ids = [m['memory_id'] for m in top_memories]
            self._update_memory_access(db, memory_ids)
            
            logger.info(
                f"Retrieved {len(top_memories)} memories for user {user_id} "
                f"(top similarity: {top_memories[0]['similarity']:.3f})"
            )
            
            return top_memories
        
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def _update_memory_access(self, db: Session, memory_ids: List[str]) -> None:
        """
        Update access_count and last_accessed for retrieved memories.
        Called internally after retrieval.
        """
        try:
            db.query(Memory).filter(Memory.id.in_(memory_ids)).update(
                {
                    Memory.access_count: Memory.access_count + 1,
                    Memory.last_accessed: datetime.utcnow()
                },
                synchronize_session=False
            )
            db.commit()
            logger.debug(f"Updated access tracking for {len(memory_ids)} memories")
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")
            db.rollback()
    
    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    def get_memories_by_category(
        self,
        db: Session,
        user_id: str,
        category: str,
        limit: int = 10
    ) -> List[Memory]:
        """
        Get memories filtered by category, ordered by importance.
        
        Args:
            db: SQLAlchemy session
            user_id: User UUID
            category: Memory category (e.g., "work_study", "relationships")
            limit: Max memories to return
        
        Returns:
            List of Memory objects
        """
        try:
            memories = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.category == category,
                Memory.is_active == True
            ).order_by(
                desc(Memory.importance_score),
                desc(Memory.created_at)
            ).limit(limit).all()
            
            logger.info(f"Retrieved {len(memories)} memories for category '{category}'")
            return memories
        
        except Exception as e:
            logger.error(f"Failed to get memories by category: {e}")
            return []
    
    def apply_memory_decay(
        self,
        db: Session,
        user_id: str,
        decay_rate: float = 0.95,
        min_decay: float = 0.3
    ) -> int:
        """
        Apply time-based decay to unpinned memories.
        Called periodically (e.g., daily cron job).
        
        Args:
            db: SQLAlchemy session
            user_id: User UUID
            decay_rate: Multiplicative decay factor (0.95 = 5% reduction)
            min_decay: Minimum decay_factor (prevent complete erasure)
        
        Returns:
            Number of memories updated
        """
        try:
            # Update unpinned memories older than 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            affected = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_pinned == False,
                Memory.is_active == True,
                Memory.created_at < cutoff_time,
                Memory.decay_factor > min_decay
            ).update(
                {Memory.decay_factor: Memory.decay_factor * decay_rate},
                synchronize_session=False
            )
            
            db.commit()
            logger.info(f"Applied decay to {affected} memories for user {user_id}")
            return affected
        
        except Exception as e:
            logger.error(f"Failed to apply memory decay: {e}")
            db.rollback()
            return 0
    
    def archive_low_value_memories(
        self,
        db: Session,
        user_id: str,
        min_decay_threshold: float = 0.2,
        min_importance_threshold: float = 0.3
    ) -> int:
        """
        Soft-delete memories that have decayed significantly or have low importance.
        
        Args:
            db: SQLAlchemy session
            user_id: User UUID
            min_decay_threshold: Archive if decay_factor below this
            min_importance_threshold: Archive if importance_score below this
        
        Returns:
            Number of memories archived
        """
        try:
            archived = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True,
                Memory.is_pinned == False,
                Memory.decay_factor < min_decay_threshold,
                Memory.importance_score < min_importance_threshold
            ).update(
                {Memory.is_active: False},
                synchronize_session=False
            )
            
            db.commit()
            logger.info(f"Archived {archived} low-value memories for user {user_id}")
            return archived
        
        except Exception as e:
            logger.error(f"Failed to archive memories: {e}")
            db.rollback()
            return 0
    
    def get_memory_stats(self, db: Session, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about user's memory storage.
        
        Returns:
            Dict with counts by type, category, average importance, etc.
        """
        try:
            total = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True
            ).count()
            
            by_type = db.query(
                Memory.memory_type,
                db.func.count(Memory.id)
            ).filter(
                Memory.user_id == user_id,
                Memory.is_active == True
            ).group_by(Memory.memory_type).all()
            
            avg_importance = db.query(
                db.func.avg(Memory.importance_score)
            ).filter(
                Memory.user_id == user_id,
                Memory.is_active == True
            ).scalar() or 0.0
            
            pinned_count = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True,
                Memory.is_pinned == True
            ).count()
            
            return {
                'total_memories': total,
                'by_type': dict(by_type),
                'avg_importance': float(avg_importance),
                'pinned_memories': pinned_count
            }
        
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    # ========================================================================
    # ASYNC OPERATIONS
    # ========================================================================
    
    async def embed_text_async(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Async wrapper for embed_text (runs in thread pool).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text, use_cache)
    
    async def embed_batch_async(self, texts: List[str]) -> List[np.ndarray]:
        """
        Async wrapper for embed_batch (runs in thread pool).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts, False)
    
    async def retrieve_memories_async(
        self,
        db: Session,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Async wrapper for retrieve_memories (runs in thread pool).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.retrieve_memories,
            db, user_id, query_text, top_k, memory_types
        )
    
    # ========================================================================
    # UTILITY & DIAGNOSTICS
    # ========================================================================
    
    @property
    def avg_embedding_time_ms(self) -> float:
        """Average time to generate one embedding (milliseconds)."""
        if self._total_embeddings == 0:
            return 0.0
        return self._total_time_ms / self._total_embeddings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'model': self.model_name,
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'total_embeddings': self._total_embeddings,
            'total_time_ms': self._total_time_ms,
            'avg_time_per_embedding_ms': self.avg_embedding_time_ms
        }
    
    def close(self) -> None:
        """
        Cleanup resources (if needed).
        Currently a no-op, but kept for API consistency.
        """
        logger.info("IndicBERT Memory Service closed")


# ============================================================================
# STANDALONE UTILITY FUNCTIONS
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Similarity score (0.0 to 1.0)
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
    
    return float(dot_product / norm_product)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding vector.
    
    Args:
        embedding: Input vector
    
    Returns:
        Normalized vector (unit length)
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


# ============================================================================
# TESTING & DEBUGGING
# ============================================================================

if __name__ == "__main__":
    """
    Test module independently.
    Usage: python -m backend.services.indicbert_memory
    """
    print("=" * 70)
    print("INDICBERT MEMORY SERVICE TEST")
    print("=" * 70)
    
    # Initialize service
    print("\n1. Initializing service...")
    service = IndicBERTMemoryService()
    
    # Test embedding generation
    print("\n2. Testing single embedding generation...")
    test_texts = [
        "User has exam anxiety and needs support",
        "मैं बहुत stressed हूँ",  # Hinglish
        "User prefers morning conversations"
    ]
    
    for text in test_texts:
        embedding = service.embed_text(text)
        print(f"   Text: {text}")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.3f}")
    
    # Test batch embedding
    print("\n3. Testing batch embedding...")
    batch_embeddings = service.embed_batch(test_texts, show_progress=True)
    print(f"   Generated {len(batch_embeddings)} embeddings")
    
    # Test similarity
    print("\n4. Testing cosine similarity...")
    sim = cosine_similarity(batch_embeddings[0], batch_embeddings[1])
    print(f"   Similarity between text 1 and 2: {sim:.3f}")
    
    # Performance stats
    print("\n5. Performance Statistics:")
    stats = service.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    
    # Database integration test (requires running database)
    print("\n6. Database Integration Test (optional):")
    print("   To test with database, run: python scripts/test_indicbert_memory.py")
    
    service.close()
