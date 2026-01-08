"""
Chunking module for advanced semantic text chunking.

This module provides intelligent chunking strategies for academic papers,
including section-aware splitting, configurable overlap, and metadata enrichment.

Components:
- SemanticChunker: Section-aware chunking with recursive splitting
- OverlapManager: Handles overlap calculation between chunks
- MetadataEnricher: Enriches chunks with section labels, page numbers, and semantic tags
"""

from typing import TypedDict, Optional, List


class ChunkMetadata(TypedDict):
    """Metadata associated with each chunk."""
    source: str
    page: int
    section: str  # "abstract", "methods", "results", etc.
    chunk_index: int
    parent_chunk_id: Optional[str]  # For hierarchical retrieval
    overlap_with_prev: int  # Characters overlapping with previous chunk
    overlap_with_next: int  # Characters overlapping with next chunk
    semantic_tags: List[str]  # LLM-generated topic tags


class Chunk(TypedDict):
    """A text chunk with associated metadata."""
    id: str
    text: str
    metadata: ChunkMetadata


# Section detection patterns for academic papers
SECTION_PATTERNS = {
    "abstract": r"^abstract\s*$|^summary\s*$",
    "introduction": r"^1\.?\s*introduction|^introduction\s*$",
    "methods": r"^2\.?\s*method|^materials?\s+and\s+methods?|^methodology",
    "results": r"^3\.?\s*results?|^findings",
    "discussion": r"^4\.?\s*discussion|^analysis",
    "conclusion": r"^5\.?\s*conclusion|^concluding",
    "references": r"^references?\s*$|^bibliography"
}


__all__ = [
    "Chunk",
    "ChunkMetadata",
    "SECTION_PATTERNS",
    "SemanticChunker",
    "ChunkConfig",
    "OverlapManager",
    "OverlapConfig",
    "MetadataEnricher",
    "EnricherConfig",
]

# Import after defining types to avoid circular imports
from .semantic_chunker import SemanticChunker, ChunkConfig
from .overlap_manager import OverlapManager, OverlapConfig
from .metadata_enricher import MetadataEnricher, EnricherConfig
