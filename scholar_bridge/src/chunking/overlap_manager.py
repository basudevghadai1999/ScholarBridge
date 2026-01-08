"""
Overlap Manager for chunk overlap calculation and application.

Provides utilities for calculating and applying overlap between consecutive
text chunks to preserve context across chunk boundaries.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from . import Chunk, ChunkMetadata


@dataclass
class OverlapConfig:
    """Configuration for overlap management."""
    overlap_percent: int = 20  # Default 20% overlap
    min_overlap_chars: int = 50  # Minimum overlap in characters
    max_overlap_chars: int = 500  # Maximum overlap in characters


class OverlapManager:
    """
    Manages overlap calculation and application between text chunks.
    
    Handles the logic for determining how much text should overlap between
    consecutive chunks to maintain context continuity, especially important
    for academic papers where concepts span multiple paragraphs.
    """
    
    def __init__(self, config: Optional[OverlapConfig] = None):
        """
        Initialize the overlap manager.
        
        Args:
            config: Overlap configuration. Uses defaults if not provided.
        """
        self.config = config or OverlapConfig()
    
    def calculate_overlap(
        self,
        chunk_a: Chunk,
        chunk_b: Chunk
    ) -> int:
        """
        Calculate the overlap size between two consecutive chunks.
        
        The overlap is calculated as a percentage of the average chunk size,
        bounded by min and max overlap settings.
        
        Args:
            chunk_a: The first (preceding) chunk.
            chunk_b: The second (following) chunk.
            
        Returns:
            Number of characters that should overlap between the chunks.
        """
        len_a = len(chunk_a["text"])
        len_b = len(chunk_b["text"])
        
        # Calculate overlap based on average size
        avg_size = (len_a + len_b) / 2
        overlap = int(avg_size * self.config.overlap_percent / 100)
        
        # Apply bounds
        overlap = max(overlap, self.config.min_overlap_chars)
        overlap = min(overlap, self.config.max_overlap_chars)
        
        # Don't overlap more than the smaller chunk
        min_chunk_size = min(len_a, len_b)
        overlap = min(overlap, min_chunk_size // 2)
        
        return overlap
    
    def apply_overlap(
        self,
        chunks: List[Chunk],
        overlap_percent: Optional[int] = None
    ) -> List[Chunk]:
        """
        Apply overlap to a list of chunks.
        
        Modifies chunks to include overlapping content from adjacent chunks
        and updates metadata to track overlap amounts.
        
        Args:
            chunks: List of chunks to process.
            overlap_percent: Override the default overlap percentage.
            
        Returns:
            New list of chunks with overlap applied.
        """
        if not chunks:
            return []
        
        if len(chunks) == 1:
            # Single chunk - no overlap needed
            return [self._copy_chunk_with_overlap(chunks[0], 0, 0)]
        
        # Use provided overlap_percent or default
        effective_percent = overlap_percent if overlap_percent is not None else self.config.overlap_percent
        
        result = []
        
        for i, chunk in enumerate(chunks):
            overlap_prev = 0
            overlap_next = 0
            new_text = chunk["text"]
            
            # Calculate and apply overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_size = self._calculate_overlap_size(
                    prev_chunk["text"], 
                    chunk["text"], 
                    effective_percent
                )
                
                if overlap_size > 0:
                    overlap_content = self._get_overlap_content(
                        prev_chunk["text"], 
                        overlap_size, 
                        from_end=True
                    )
                    
                    # Only prepend if not already present
                    if not new_text.startswith(overlap_content.strip()):
                        new_text = overlap_content.rstrip() + " " + new_text
                        overlap_prev = len(overlap_content)
            
            # Calculate overlap with next chunk (for metadata tracking)
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_size = self._calculate_overlap_size(
                    chunk["text"], 
                    next_chunk["text"], 
                    effective_percent
                )
                overlap_next = overlap_size
            
            # Create new chunk with updated text and metadata
            new_chunk = self._copy_chunk_with_overlap(
                chunk, 
                overlap_prev, 
                overlap_next, 
                new_text
            )
            result.append(new_chunk)
        
        return result
    
    def _calculate_overlap_size(
        self,
        text_a: str,
        text_b: str,
        overlap_percent: int
    ) -> int:
        """
        Calculate overlap size between two text segments.
        
        Args:
            text_a: First text segment.
            text_b: Second text segment.
            overlap_percent: Percentage of overlap.
            
        Returns:
            Number of characters to overlap.
        """
        avg_size = (len(text_a) + len(text_b)) / 2
        overlap = int(avg_size * overlap_percent / 100)
        
        # Apply bounds
        overlap = max(overlap, self.config.min_overlap_chars)
        overlap = min(overlap, self.config.max_overlap_chars)
        
        # Don't overlap more than half of the smaller text
        min_size = min(len(text_a), len(text_b))
        overlap = min(overlap, min_size // 2)
        
        return overlap
    
    def _get_overlap_content(
        self,
        text: str,
        overlap_size: int,
        from_end: bool = True
    ) -> str:
        """
        Extract overlap content from text.
        
        Tries to break at word boundaries for cleaner overlap.
        
        Args:
            text: Source text.
            overlap_size: Target overlap size in characters.
            from_end: If True, extract from end; otherwise from start.
            
        Returns:
            Extracted overlap content.
        """
        if overlap_size <= 0:
            return ""
        
        if from_end:
            # Extract from end, try to break at word boundary
            raw_content = text[-overlap_size:]
            
            # Find first space to start at word boundary
            space_idx = raw_content.find(' ')
            if space_idx > 0 and space_idx < len(raw_content) // 2:
                return raw_content[space_idx + 1:]
            return raw_content
        else:
            # Extract from start, try to break at word boundary
            raw_content = text[:overlap_size]
            
            # Find last space to end at word boundary
            space_idx = raw_content.rfind(' ')
            if space_idx > len(raw_content) // 2:
                return raw_content[:space_idx]
            return raw_content
    
    def _copy_chunk_with_overlap(
        self,
        chunk: Chunk,
        overlap_prev: int,
        overlap_next: int,
        new_text: Optional[str] = None
    ) -> Chunk:
        """
        Create a copy of a chunk with updated overlap metadata.
        
        Args:
            chunk: Original chunk.
            overlap_prev: Overlap with previous chunk.
            overlap_next: Overlap with next chunk.
            new_text: Optional new text content.
            
        Returns:
            New chunk with updated metadata.
        """
        new_metadata: ChunkMetadata = {
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "section": chunk["metadata"]["section"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "parent_chunk_id": chunk["metadata"]["parent_chunk_id"],
            "overlap_with_prev": overlap_prev,
            "overlap_with_next": overlap_next,
            "semantic_tags": chunk["metadata"]["semantic_tags"].copy(),
        }
        
        return {
            "id": chunk["id"],
            "text": new_text if new_text is not None else chunk["text"],
            "metadata": new_metadata,
        }
    
    def get_overlap_stats(self, chunks: List[Chunk]) -> dict:
        """
        Get statistics about overlap in a list of chunks.
        
        Args:
            chunks: List of chunks to analyze.
            
        Returns:
            Dictionary with overlap statistics.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_overlap_prev": 0,
                "avg_overlap_next": 0,
                "total_overlap_chars": 0,
            }
        
        total_prev = sum(c["metadata"]["overlap_with_prev"] for c in chunks)
        total_next = sum(c["metadata"]["overlap_with_next"] for c in chunks)
        
        return {
            "total_chunks": len(chunks),
            "avg_overlap_prev": total_prev / len(chunks),
            "avg_overlap_next": total_next / len(chunks),
            "total_overlap_chars": total_prev + total_next,
        }
    
    @staticmethod
    def from_config_dict(config_dict: dict) -> "OverlapManager":
        """
        Create an OverlapManager from a configuration dictionary.
        
        Args:
            config_dict: Dictionary with overlap configuration.
            
        Returns:
            Configured OverlapManager instance.
        """
        config = OverlapConfig(
            overlap_percent=config_dict.get("overlap_percent", 20),
            min_overlap_chars=config_dict.get("min_overlap_chars", 50),
            max_overlap_chars=config_dict.get("max_overlap_chars", 500),
        )
        return OverlapManager(config)
  