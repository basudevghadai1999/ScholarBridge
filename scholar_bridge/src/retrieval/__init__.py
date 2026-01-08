"""
Retrieval module for hybrid search capabilities.

This module provides hybrid retrieval combining BM25 keyword search
with vector similarity search, plus reranking for optimal results.

Components:
- BM25Index: Keyword-based search using BM25 algorithm
- HybridRetriever: Combines BM25 and vector search with RRF fusion
- Reranker: LLM-based relevance reranking and diversity filtering
"""

from typing import TypedDict, List, Optional
from dataclasses import dataclass


class RetrievalResult(TypedDict):
    """A single retrieval result with score."""
    chunk_id: str
    text: str
    score: float
    source: str  # "bm25", "vector", or "hybrid"
    metadata: dict


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    top_k_bm25: int = 20
    top_k_vector: int = 20
    final_top_k: int = 10
    rrf_k: int = 60  # RRF constant


@dataclass
class RerankConfig:
    """Configuration for reranking."""
    enabled: bool = True
    diversity_threshold: float = 0.7
    relevance_threshold: float = 0.5
    max_context_tokens: int = 4000


__all__ = [
    "RetrievalResult",
    "RetrievalConfig",
    "RerankConfig",
    "BM25Index",
    "HybridRetriever",
    "Reranker",
]

# Import after defining types to avoid circular imports
from .bm25_index import BM25Index
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker
