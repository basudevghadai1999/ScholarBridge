"""
Metadata Enricher for chunk metadata enhancement.

Provides utilities for enriching chunk metadata with page numbers,
section labels, hierarchical IDs, and optional LLM-based semantic tagging.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re
import hashlib

from . import Chunk, ChunkMetadata, SECTION_PATTERNS


@dataclass
class EnricherConfig:
    """Configuration for metadata enrichment."""
    semantic_tagging: bool = False  # LLM-based tagging (expensive)
    hierarchical_ids: bool = True  # Generate hierarchical chunk IDs
    extract_page_numbers: bool = True  # Extract page numbers from text


class MetadataEnricher:
    """
    Enriches chunk metadata with additional information.
    
    Handles page number extraction, section label assignment,
    hierarchical ID generation, and optional LLM-based semantic tagging.
    """
    
    def __init__(
        self, 
        config: Optional[EnricherConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the metadata enricher.
        
        Args:
            config: Enricher configuration. Uses defaults if not provided.
            llm_client: Optional LLM client for semantic tagging.
        """
        self.config = config or EnricherConfig()
        self.llm_client = llm_client
        self._compiled_patterns = {
            section: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for section, pattern in SECTION_PATTERNS.items()
        }
    
    def enrich(
        self,
        chunks: List[Chunk],
        pdf_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Enrich chunks with additional metadata.
        
        Args:
            chunks: List of chunks to enrich.
            pdf_metadata: Optional PDF metadata with page information.
            
        Returns:
            List of enriched chunks.
        """
        if not chunks:
            return []
        
        enriched = []
        
        for i, chunk in enumerate(chunks):
            enriched_chunk = self._enrich_chunk(chunk, i, len(chunks), pdf_metadata)
            enriched.append(enriched_chunk)
        
        # Build parent-child relationships
        if self.config.hierarchical_ids:
            enriched = self._assign_parent_relationships(enriched)
        
        return enriched
    
    def _enrich_chunk(
        self,
        chunk: Chunk,
        index: int,
        total_chunks: int,
        pdf_metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """
        Enrich a single chunk with metadata.
        
        Args:
            chunk: The chunk to enrich.
            index: Index of the chunk in the list.
            total_chunks: Total number of chunks.
            pdf_metadata: Optional PDF metadata.
            
        Returns:
            Enriched chunk.
        """
        new_metadata = dict(chunk["metadata"])
        
        # Extract page number
        if self.config.extract_page_numbers and pdf_metadata:
            page = self._extract_page_number(chunk, pdf_metadata)
            new_metadata["page"] = page
        
        # Assign section label if not already set
        if not new_metadata.get("section") or new_metadata["section"] == "body":
            section = self._detect_section(chunk["text"])
            if section:
                new_metadata["section"] = section
        
        # Generate hierarchical ID
        if self.config.hierarchical_ids:
            new_id = self._generate_hierarchical_id(
                chunk["metadata"]["source"],
                new_metadata["section"],
                index,
                chunk["text"]
            )
        else:
            new_id = chunk["id"]
        
        return {
            "id": new_id,
            "text": chunk["text"],
            "metadata": new_metadata,
        }
    
    def _extract_page_number(
        self,
        chunk: Chunk,
        pdf_metadata: Dict[str, Any]
    ) -> int:
        """
        Extract page number for a chunk from PDF metadata.
        
        Args:
            chunk: The chunk to get page number for.
            pdf_metadata: PDF metadata containing page mappings.
            
        Returns:
            Page number (0 if not found).
        """
        # Check if pdf_metadata contains page_map (char_offset -> page)
        page_map = pdf_metadata.get("page_map", {})
        
        if not page_map:
            # Try to extract from chunk text patterns like "Page X" or "[X]"
            page_pattern = re.search(r'\bpage\s*(\d+)\b', chunk["text"], re.IGNORECASE)
            if page_pattern:
                return int(page_pattern.group(1))
            return 0
        
        # Use character offset mapping if available
        chunk_start = pdf_metadata.get("chunk_offsets", {}).get(chunk["id"], 0)
        
        # Find the page for this offset
        for offset, page in sorted(page_map.items()):
            if chunk_start >= int(offset):
                return page
        
        return 0
    
    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect section type from chunk text.
        
        Args:
            text: Chunk text to analyze.
            
        Returns:
            Section name or None if not detected.
        """
        # Check first few lines for section headers
        lines = text.split('\n')[:5]
        
        for line in lines:
            stripped = line.strip()
            for section_name, pattern in self._compiled_patterns.items():
                if pattern.match(stripped):
                    return section_name
        
        # Heuristic detection based on content
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['we propose', 'this paper', 'we present', 'in this work']):
            return "introduction"
        if any(kw in text_lower for kw in ['experiment', 'dataset', 'training', 'evaluation']):
            return "methods"
        if any(kw in text_lower for kw in ['accuracy', 'performance', 'table', 'figure', 'results show']):
            return "results"
        if any(kw in text_lower for kw in ['limitation', 'future work', 'in conclusion', 'we have shown']):
            return "conclusion"
        
        return None
    
    def _generate_hierarchical_id(
        self,
        source: str,
        section: str,
        index: int,
        text: str
    ) -> str:
        """
        Generate a hierarchical chunk ID.
        
        Format: {source_hash}_{section}_{index}_{content_hash}
        
        Args:
            source: Source document identifier.
            section: Section name.
            index: Chunk index.
            text: Chunk text for content hash.
            
        Returns:
            Hierarchical chunk ID.
        """
        # Create short hash of source
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        
        # Create short hash of content for uniqueness
        content_hash = hashlib.md5(text.encode()).hexdigest()[:6]
        
        # Sanitize section name
        section_clean = section.replace(" ", "_").lower() if section else "unknown"
        
        return f"{source_hash}_{section_clean}_{index:04d}_{content_hash}"
    
    def _assign_parent_relationships(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Assign parent-child relationships between chunks.
        
        Chunks in the same section share a parent relationship.
        
        Args:
            chunks: List of chunks to process.
            
        Returns:
            Chunks with parent_chunk_id assigned.
        """
        # Group chunks by section
        section_first_chunk: Dict[str, str] = {}
        result = []
        
        for chunk in chunks:
            section = chunk["metadata"]["section"]
            
            if section not in section_first_chunk:
                # This is the first chunk of this section (parent)
                section_first_chunk[section] = chunk["id"]
                parent_id = None
            else:
                # This chunk's parent is the first chunk of its section
                parent_id = section_first_chunk[section]
            
            # Update metadata with parent ID
            new_metadata = dict(chunk["metadata"])
            new_metadata["parent_chunk_id"] = parent_id
            
            result.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": new_metadata,
            })
        
        return result

    
    async def enrich_with_semantic_tags(
        self,
        chunks: List[Chunk],
        batch_size: int = 5
    ) -> List[Chunk]:
        """
        Enrich chunks with LLM-generated semantic tags.
        
        This is an expensive operation that calls the LLM for each batch.
        
        Args:
            chunks: List of chunks to tag.
            batch_size: Number of chunks to process per LLM call.
            
        Returns:
            Chunks with semantic_tags populated.
        """
        if not self.config.semantic_tagging or not self.llm_client:
            return chunks
        
        result = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            tagged_batch = await self._tag_batch(batch)
            result.extend(tagged_batch)
        
        return result
    
    async def _tag_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate semantic tags for a batch of chunks.
        
        Args:
            chunks: Batch of chunks to tag.
            
        Returns:
            Chunks with semantic tags.
        """
        if not self.llm_client:
            return chunks
        
        # Build prompt for tagging
        prompt = self._build_tagging_prompt(chunks)
        
        try:
            response = await self.llm_client.generate(prompt)
            tags_map = self._parse_tags_response(response, len(chunks))
        except Exception:
            # On error, return chunks without tags
            tags_map = {i: [] for i in range(len(chunks))}
        
        result = []
        for i, chunk in enumerate(chunks):
            new_metadata = dict(chunk["metadata"])
            new_metadata["semantic_tags"] = tags_map.get(i, [])
            
            result.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": new_metadata,
            })
        
        return result
    
    def _build_tagging_prompt(self, chunks: List[Chunk]) -> str:
        """
        Build prompt for semantic tagging.
        
        Args:
            chunks: Chunks to tag.
            
        Returns:
            Prompt string.
        """
        chunks_text = "\n\n".join([
            f"[Chunk {i}]\n{chunk['text'][:500]}..."
            for i, chunk in enumerate(chunks)
        ])
        
        return f"""Analyze the following text chunks from an academic paper and generate 2-4 semantic tags for each chunk.
Tags should capture the main topics, concepts, or themes discussed.

{chunks_text}

Respond in this exact format for each chunk:
Chunk 0: tag1, tag2, tag3
Chunk 1: tag1, tag2
...
"""
    
    def _parse_tags_response(
        self,
        response: str,
        num_chunks: int
    ) -> Dict[int, List[str]]:
        """
        Parse LLM response to extract tags.
        
        Args:
            response: LLM response text.
            num_chunks: Expected number of chunks.
            
        Returns:
            Mapping of chunk index to tags.
        """
        tags_map = {}
        
        for line in response.split('\n'):
            match = re.match(r'Chunk\s*(\d+):\s*(.+)', line, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                tags = [t.strip() for t in match.group(2).split(',')]
                tags_map[idx] = tags
        
        # Fill in missing chunks with empty tags
        for i in range(num_chunks):
            if i not in tags_map:
                tags_map[i] = []
        
        return tags_map
    
    @staticmethod
    def from_config_dict(
        config_dict: dict,
        llm_client: Optional[Any] = None
    ) -> "MetadataEnricher":
        """
        Create a MetadataEnricher from a configuration dictionary.
        
        Args:
            config_dict: Dictionary with enricher configuration.
            llm_client: Optional LLM client for semantic tagging.
            
        Returns:
            Configured MetadataEnricher instance.
        """
        config = EnricherConfig(
            semantic_tagging=config_dict.get("semantic_tagging", False),
            hierarchical_ids=config_dict.get("hierarchical_ids", True),
            extract_page_numbers=config_dict.get("extract_page_numbers", True),
        )
        return MetadataEnricher(config, llm_client)
