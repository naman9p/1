"""
Vector Store Service

ChromaDB integration for vector storage and semantic search.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import time

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.candidate import CandidateDocument

# Lazy import
_chroma_client = None
_chroma_collection = None


class VectorStoreService:
    """
    Service for vector storage and retrieval using ChromaDB.

    Features:
    - Persistent storage
    - Efficient similarity search
    - Metadata filtering
    - Batch operations
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the vector store service.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self._client = None
        self._collection = None
        self.logger = get_logger(__name__)

    def initialize(self) -> bool:
        """
        Initialize ChromaDB client and collection.

        Returns:
            True if initialization successful
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.logger.info("Initializing ChromaDB...")

            # Create persistent client
            db_path = Path(self.settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

            self.logger.info(
                f"ChromaDB initialized. Collection: {self.settings.chroma_collection}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self._client is not None and self._collection is not None

    @property
    def collection_size(self) -> int:
        """Get number of vectors in collection."""
        if not self.is_initialized:
            return 0
        return self._collection.count()

    def upsert_candidate(
        self,
        candidate_id: str,
        embedding: List[float],
        document: CandidateDocument,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upsert a candidate into the vector store.

        Args:
            candidate_id: Unique candidate identifier
            embedding: Embedding vector
            document: Candidate document
            metadata: Additional metadata

        Returns:
            True if successful
        """
        if not self.is_initialized:
            self.logger.error("Vector store not initialized")
            return False

        try:
            # Prepare metadata
            meta = {
                "candidate_id": candidate_id,
                "seniority_level": document.seniority_level,
                "total_years_experience": getattr(
                    document, "total_years_experience", None
                ),
                **(metadata or {}),
            }

            # Filter out None values
            meta = {k: v for k, v in meta.items() if v is not None}

            # Upsert
            self._collection.upsert(
                ids=[candidate_id],
                embeddings=[embedding],
                documents=[document.full_document],
                metadatas=[meta],
            )

            self.logger.debug(f"Upserted candidate: {candidate_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to upsert candidate {candidate_id}: {e}")
            return False

    def upsert_candidates_batch(
        self,
        candidates: List[Tuple[str, List[float], CandidateDocument, Dict[str, Any]]],
        batch_size: int = 100,
    ) -> Tuple[int, int]:
        """
        Upsert multiple candidates in batches.

        Args:
            candidates: List of (candidate_id, embedding, document, metadata) tuples
            batch_size: Batch size for upserts

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not self.is_initialized:
            self.logger.error("Vector store not initialized")
            return 0, len(candidates)

        successful = 0
        failed = 0

        self.logger.info(f"Upserting {len(candidates)} candidates in batches of {batch_size}")
        start_time = time.time()

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for candidate_id, embedding, document, metadata in batch:
                ids.append(candidate_id)
                embeddings.append(embedding)
                documents.append(document.full_document)

                meta = {
                    "candidate_id": candidate_id,
                    "seniority_level": document.seniority_level,
                    **(metadata or {}),
                }
                meta = {k: v for k, v in meta.items() if v is not None}
                metadatas.append(meta)

            try:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                successful += len(batch)
            except Exception as e:
                self.logger.error(f"Batch upsert failed: {e}")
                failed += len(batch)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Batch upsert completed: {successful} successful, {failed} failed in {elapsed:.2f}s"
        )

        return successful, failed

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar candidates.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Metadata filters

        Returns:
            List of search results with distances and metadata
        """
        if not self.is_initialized:
            self.logger.error("Vector store not initialized")
            return []

        try:
            # Prepare where filter
            where_filter = None
            if filter_metadata:
                where_filter = filter_metadata

            # Query
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["embeddings", "documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, candidate_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "candidate_id": candidate_id,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "document": results["documents"][0][i] if results["documents"] else None,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                        "embedding": results["embeddings"][0][i] if results["embeddings"] else None,
                    })

            self.logger.debug(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific candidate by ID.

        Args:
            candidate_id: Candidate identifier

        Returns:
            Candidate data or None if not found
        """
        if not self.is_initialized:
            return None

        try:
            results = self._collection.get(
                ids=[candidate_id],
                include=["embeddings", "documents", "metadatas"],
            )

            if results and results["ids"] and results["ids"][0]:
                return {
                    "candidate_id": results["ids"][0],
                    "document": results["documents"][0] if results["documents"] else None,
                    "metadata": results["metadatas"][0] if results["metadatas"] else None,
                    "embedding": results["embeddings"][0] if results["embeddings"] else None,
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to get candidate {candidate_id}: {e}")
            return None

    def delete_candidate(self, candidate_id: str) -> bool:
        """
        Delete a candidate from the vector store.

        Args:
            candidate_id: Candidate identifier

        Returns:
            True if successful
        """
        if not self.is_initialized:
            return False

        try:
            self._collection.delete(ids=[candidate_id])
            self.logger.debug(f"Deleted candidate: {candidate_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete candidate {candidate_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.

        Returns:
            True if successful
        """
        if not self.is_initialized:
            return False

        try:
            # Delete and recreate collection
            self._client.delete_collection(self.settings.chroma_collection)
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info("Collection cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset the vector store (delete all data).

        Returns:
            True if successful
        """
        try:
            import shutil

            db_path = Path(self.settings.chroma_db_path)
            if db_path.exists():
                shutil.rmtree(db_path)

            self._client = None
            self._collection = None

            return self.initialize()

        except Exception as e:
            self.logger.error(f"Failed to reset vector store: {e}")
            return False


# Global instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store(settings: Optional[Settings] = None) -> VectorStoreService:
    """
    Get or create the vector store service singleton.

    Args:
        settings: Optional settings override

    Returns:
        VectorStoreService instance
    """
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStoreService(settings)

    return _vector_store
