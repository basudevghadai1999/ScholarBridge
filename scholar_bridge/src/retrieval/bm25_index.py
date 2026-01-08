"""
BM25 Index for keyword-based search.

Provides BM25 ranking algorithm for keyword search alongside vector similarity.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from . import RetrievalResult


@dataclass
class BM25Config:
    """Configuration for BM25 index."""
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization


class BM25Index:
    """
    BM25 keyword search index.
    
    Uses the BM25Okapi algorithm for ranking documents by keyword relevance.
    Designed to work alongside vector search for hybrid retrieval.
    """
    
    def __init__(self, config: Optional[BM25Config] = None):
        """
        Initialize the BM25 index.
        
        Args:
            config: BM25 configuration. Uses defaults if not provided.
        """
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required. Install with: pip install rank-bm25")
        
        self.config = config or BM25Config()
        self._index: Optional[BM25Okapi] = None
        self._chunks: List[Dict[str, Any]] = []
        self._tokenized_corpus: List[List[str]] = []
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'text', and 'metadata'.
        """
        if not chunks:
            self._index = None
            self._chunks = []
            return
        
        self._chunks = chunks
        self._tokenized_corpus = [
            self._tokenize(chunk["text"]) for chunk in chunks
        ]
        
        self._index = BM25Okapi(
            self._tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b
        )
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Search the index for relevant chunks.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            
        Returns:
            List of RetrievalResult sorted by relevance score.
        """
        if self._index is None or not self._chunks:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)
        
        # Get top-k indices
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]
        
        results = []
        for idx, score in top_indices:
            if score > 0:  # Only include positive scores
                chunk = self._chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    score=float(score),
                    source="bm25",
                    metadata=chunk.get("metadata", {}),
                ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Handles academic text with special consideration for:
        - Technical terms and acronyms
        - Hyphenated words
        - Numbers and equations
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens.
        """
        # Lowercase
        text = text.lower()
        
        # Keep alphanumeric, hyphens, and underscores
        # Split on whitespace and punctuation (except hyphens in words)
        tokens = re.findall(r'\b[\w-]+\b', text)
        
        # Filter out very short tokens and pure numbers
        tokens = [
            t for t in tokens 
            if len(t) > 1 and not t.isdigit()
        ]
        
        # Remove common stopwords for academic text
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our',
            'they', 'their', 'which', 'who', 'whom', 'what', 'where', 'when',
        }
        
        tokens = [t for t in tokens if t not in stopwords]
        
        return tokens
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics.
        """
        if not self._chunks:
            return {
                "num_documents": 0,
                "avg_doc_length": 0,
                "total_tokens": 0,
            }
        
        total_tokens = sum(len(tokens) for tokens in self._tokenized_corpus)
        
        return {
            "num_documents": len(self._chunks),
            "avg_doc_length": total_tokens / len(self._chunks),
            "total_tokens": total_tokens,
        }
    
    @staticmethod
    def from_config_dict(config_dict: dict) -> "BM25Index":
        """
        Create a BM25Index from a configuration dictionary.
        
        Args:
            config_dict: Dictionary with BM25 configuration.
            
        Returns:
            Configured BM25Index instance.
        """
        config = BM25Config(
            k1=config_dict.get("k1", 1.5),
            b=config_dict.get("b", 0.75),
        )
        return BM25Index(config)
