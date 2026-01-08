"""
Semantic Chunker for academic papers.

Provides section-aware chunking with recursive splitting and configurable overlap.
Designed specifically for academic paper structure (Abstract, Methods, Results, etc.).
"""

import re
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import Chunk, ChunkMetadata, SECTION_PATTERNS


@dataclass
class ChunkConfig:
    """Configuration for semantic chunking."""
    chunk_size: int = 1000  # Target chunk size in characters
    overlap_percent: int = 20  # Overlap percentage between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    section_detection: bool = True  # Enable section detection


class SemanticChunker:
    """
    Section-aware chunker for academic papers.
    
    Detects paper structure (Abstract, Introduction, Methods, etc.) and
    performs recursive splitting within sections while maintaining
    configurable overlap between chunks.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize the semantic chunker.
        
        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkConfig()
        self._compiled_patterns = {
            section: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for section, pattern in SECTION_PATTERNS.items()
        }
    
    def chunk(
        self,
        text: str,
        source: str = "unknown",
        doc_type: str = "paper"
    ) -> List[Chunk]:
        """
        Chunk text into semantically meaningful segments.
        
        Args:
            text: The full text to chunk.
            source: Source identifier (e.g., filename, URL).
            doc_type: Document type ("paper" for academic papers).
            
        Returns:
            List of Chunk objects with metadata.
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Detect sections
        sections = self.detect_sections(text) if self.config.section_detection else []
        
        # Step 2: Split within sections
        if sections:
            raw_chunks = self._recursive_split_with_sections(text, sections)
        else:
            raw_chunks = self._recursive_split(text, "body")
        
        # Step 3: Apply overlap
        chunks_with_overlap = self._apply_overlap(raw_chunks)
        
        # Step 4: Create Chunk objects with metadata
        chunks = self._create_chunks(chunks_with_overlap, source)
        
        return chunks
    
    def detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect section boundaries in academic paper text.
        
        Args:
            text: The full text to analyze.
            
        Returns:
            List of tuples: (section_name, start_index, end_index)
        """
        sections = []
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            for section_name, pattern in self._compiled_patterns.items():
                if pattern.match(stripped):
                    sections.append((section_name, current_pos, i))
                    break
            current_pos += len(line) + 1  # +1 for newline
        
        if not sections:
            return []
        
        # Convert line indices to character positions and set end positions
        result = []
        lines_cumulative = []
        pos = 0
        for line in lines:
            lines_cumulative.append(pos)
            pos += len(line) + 1
        
        for i, (section_name, _, line_idx) in enumerate(sections):
            start_pos = lines_cumulative[line_idx] if line_idx < len(lines_cumulative) else 0
            
            # End position is start of next section or end of text
            if i + 1 < len(sections):
                next_line_idx = sections[i + 1][2]
                end_pos = lines_cumulative[next_line_idx] if next_line_idx < len(lines_cumulative) else len(text)
            else:
                end_pos = len(text)
            
            result.append((section_name, start_pos, end_pos))
        
        return result

    
    def _recursive_split_with_sections(
        self,
        text: str,
        sections: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, str]]:
        """
        Split text recursively while respecting section boundaries.
        
        Args:
            text: The full text.
            sections: List of (section_name, start, end) tuples.
            
        Returns:
            List of (chunk_text, section_name) tuples.
        """
        chunks = []
        
        # Handle text before first section
        if sections and sections[0][1] > 0:
            pre_section_text = text[:sections[0][1]].strip()
            if pre_section_text:
                chunks.extend(self._recursive_split(pre_section_text, "preamble"))
        
        # Process each section
        for section_name, start, end in sections:
            section_text = text[start:end].strip()
            if section_text:
                section_chunks = self._recursive_split(section_text, section_name)
                chunks.extend(section_chunks)
        
        return chunks
    
    def _recursive_split(
        self,
        text: str,
        section: str
    ) -> List[Tuple[str, str]]:
        """
        Recursively split text into chunks of target size.
        
        Uses a hierarchy of separators: paragraphs -> sentences -> words.
        
        Args:
            text: Text to split.
            section: Section name for metadata.
            
        Returns:
            List of (chunk_text, section_name) tuples.
        """
        if len(text) <= self.config.chunk_size:
            return [(text, section)] if text.strip() else []
        
        # Try splitting by paragraphs first
        separators = [
            '\n\n',  # Double newline (paragraphs)
            '\n',    # Single newline
            '. ',    # Sentence boundary
            ', ',    # Clause boundary
            ' ',     # Word boundary
        ]
        
        for separator in separators:
            parts = text.split(separator)
            if len(parts) > 1:
                chunks = self._merge_splits(parts, separator, section)
                if chunks:
                    return chunks
        
        # Fallback: hard split at chunk_size
        return self._hard_split(text, section)
    
    def _merge_splits(
        self,
        parts: List[str],
        separator: str,
        section: str
    ) -> List[Tuple[str, str]]:
        """
        Merge split parts back together to reach target chunk size.
        
        Args:
            parts: List of text parts.
            separator: The separator used to split.
            section: Section name.
            
        Returns:
            List of (chunk_text, section_name) tuples.
        """
        chunks = []
        current_chunk = ""
        
        for part in parts:
            test_chunk = current_chunk + separator + part if current_chunk else part
            
            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append((current_chunk.strip(), section))
                
                # If part itself is too large, recursively split it
                if len(part) > self.config.chunk_size:
                    sub_chunks = self._recursive_split(part, section)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunks.append((current_chunk.strip(), section))
        elif current_chunk and chunks:
            # Append small remainder to previous chunk if possible
            prev_text, prev_section = chunks[-1]
            chunks[-1] = (prev_text + separator + current_chunk.strip(), prev_section)
        
        return chunks
    
    def _hard_split(
        self,
        text: str,
        section: str
    ) -> List[Tuple[str, str]]:
        """
        Hard split text at chunk_size boundaries (last resort).
        
        Args:
            text: Text to split.
            section: Section name.
            
        Returns:
            List of (chunk_text, section_name) tuples.
        """
        chunks = []
        for i in range(0, len(text), self.config.chunk_size):
            chunk_text = text[i:i + self.config.chunk_size].strip()
            if chunk_text:
                chunks.append((chunk_text, section))
        return chunks

    
    def _apply_overlap(
        self,
        chunks: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, int, int]]:
        """
        Apply overlap between consecutive chunks.
        
        Args:
            chunks: List of (chunk_text, section_name) tuples.
            
        Returns:
            List of (chunk_text, section_name, overlap_prev, overlap_next) tuples.
        """
        if not chunks:
            return []
        
        overlap_chars = int(self.config.chunk_size * self.config.overlap_percent / 100)
        result = []
        
        for i, (text, section) in enumerate(chunks):
            overlap_prev = 0
            overlap_next = 0
            new_text = text
            
            # Add overlap from previous chunk (append to beginning)
            if i > 0:
                prev_text = chunks[i - 1][0]
                overlap_content = prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text
                # Only add if it doesn't duplicate
                if not new_text.startswith(overlap_content):
                    new_text = overlap_content + " " + new_text
                    overlap_prev = len(overlap_content)
            
            # Calculate overlap with next (for metadata, actual overlap added in next iteration)
            if i < len(chunks) - 1:
                next_text = chunks[i + 1][0]
                overlap_content = text[-overlap_chars:] if len(text) > overlap_chars else text
                overlap_next = len(overlap_content)
            
            result.append((new_text, section, overlap_prev, overlap_next))
        
        return result
    
    def _create_chunks(
        self,
        chunks_data: List[Tuple[str, str, int, int]],
        source: str
    ) -> List[Chunk]:
        """
        Create Chunk objects with full metadata.
        
        Args:
            chunks_data: List of (text, section, overlap_prev, overlap_next) tuples.
            source: Source identifier.
            
        Returns:
            List of Chunk objects.
        """
        chunks = []
        
        for i, (text, section, overlap_prev, overlap_next) in enumerate(chunks_data):
            chunk_id = f"{source}_{section}_{i}_{uuid.uuid4().hex[:8]}"
            
            metadata: ChunkMetadata = {
                "source": source,
                "page": 0,  # Will be enriched by MetadataEnricher
                "section": section,
                "chunk_index": i,
                "parent_chunk_id": None,
                "overlap_with_prev": overlap_prev,
                "overlap_with_next": overlap_next,
                "semantic_tags": [],  # Will be enriched by MetadataEnricher
            }
            
            chunk: Chunk = {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
            }
            
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def from_config_dict(config_dict: dict) -> "SemanticChunker":
        """
        Create a SemanticChunker from a configuration dictionary.
        
        Args:
            config_dict: Dictionary with chunking configuration.
            
        Returns:
            Configured SemanticChunker instance.
        """
        config = ChunkConfig(
            chunk_size=config_dict.get("chunk_size", 1000),
            overlap_percent=config_dict.get("overlap_percent", 20),
            min_chunk_size=config_dict.get("min_chunk_size", 100),
            section_detection=config_dict.get("section_detection", True),
        )
        return SemanticChunker(config)
