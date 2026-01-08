"""
Hybrid Retriever combining BM25 and vector search.

Uses Reciprocal Rank Fusion (RRF) to merge results from keyword
and semantic search for optimal retrieval quality.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from . import RetrievalResult, RetrievalConfig
from .bm25_index import BM25Index


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    top_k_bm25: int = 20
    top_k_vector: int = 20
    final_top_k: int = 10
    rrf_k: int = 60


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrieval methods, providing better coverage than either alone.
    """
    
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store: Any,  # ChromaDB collection
        config: Optional[HybridConfig] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            bm25_index: BM25 index for keyword search.
            vector_store: ChromaDB collection for vector search.
            config: Hybrid retrieval configuration.
        """
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.config = config or HybridConfig()
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query string.
            top_k: Number of results to return (overrides config).
            
        Returns:
            List of RetrievalResult sorted by combined relevance.
        """
        final_k = top_k or self.config.final_top_k
        
        # Get BM25 results
        bm25_results = self.bm25_index.search(
            query, 
            top_k=self.config.top_k_bm25
        )
        
        # Get vector search results
        vector_results = self._vector_search(
            query,
            top_k=self.config.top_k_vector
        )
        
        # Merge using RRF
        merged = self.rrf_fusion(
            bm25_results,
            vector_results,
            k=self.config.rrf_k
        )
        
        return merged[:final_k]
    
    def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            
        Returns:
            List of RetrievalResult from vector search.
        """
        if self.vector_store is None:
            return []
        
        try:
            # ChromaDB query
            results = self.vector_store.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieval_results = []
            
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0] if results.get("documents") else []
                metadatas = results["metadatas"][0] if results.get("metadatas") else []
                distances = results["distances"][0] if results.get("distances") else []
                
                for i, chunk_id in enumerate(ids):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    distance = distances[i] if i < len(distances) else 0
                    score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    retrieval_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        text=documents[i] if i < len(documents) else "",
                        score=score,
                        source="vector",
                        metadata=metadatas[i] if i < len(metadatas) else {},
                    ))
            
            return retrieval_results
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def rrf_fusion(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        RRF Score = Σ 1 / (k + rank_i)
        
        Args:
            bm25_results: Results from BM25 search.
            vector_results: Results from vector search.
            k: RRF constant (default 60).
            
        Returns:
            Merged and re-ranked results.
        """
        scores: Dict[str, float] = defaultdict(float)
        chunks: Dict[str, RetrievalResult] = {}
        
        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] += self.config.bm25_weight / (k + rank + 1)
            if chunk_id not in chunks:
                chunks[chunk_id] = result
        
        # Score vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] += self.config.vector_weight / (k + rank + 1)
            if chunk_id not in chunks:
                chunks[chunk_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build final results with updated scores
        results = []
        for chunk_id in sorted_ids:
            result = chunks[chunk_id].copy()
            result["score"] = scores[chunk_id]
            result["source"] = "hybrid"
            results.append(result)
        
        return results

    
    def get_retrieval_stats(
        self,
        bm25_results: List[RetrievalResult],
        vector_results: List[RetrievalResult],
        merged_results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval results.
        
        Args:
            bm25_results: Results from BM25 search.
            vector_results: Results from vector search.
            merged_results: Merged results after RRF.
            
        Returns:
            Dictionary with retrieval statistics.
        """
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        vector_ids = {r["chunk_id"] for r in vector_results}
        merged_ids = {r["chunk_id"] for r in merged_results}
        
        return {
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
            "merged_count": len(merged_results),
            "overlap_count": len(bm25_ids & vector_ids),
            "bm25_only": len(bm25_ids - vector_ids),
            "vector_only": len(vector_ids - bm25_ids),
            "avg_bm25_score": sum(r["score"] for r in bm25_results) / len(bm25_results) if bm25_results else 0,
            "avg_vector_score": sum(r["score"] for r in vector_results) / len(vector_results) if vector_results else 0,
        }
    
    @staticmethod
    def from_config_dict(
        config_dict: dict,
        bm25_index: BM25Index,
        vector_store: Any
    ) -> "HybridRetriever":
        """
        Create a HybridRetriever from a configuration dictionary.
        
        Args:
            config_dict: Dictionary with retrieval configuration.
            bm25_index: BM25 index instance.
            vector_store: ChromaDB collection.
            
        Returns:
            Configured HybridRetriever instance.
        """
        config = HybridConfig(
            bm25_weight=config_dict.get("bm25_weight", 0.3),
            vector_weight=config_dict.get("vector_weight", 0.7),
            top_k_bm25=config_dict.get("top_k_bm25", 20),
            top_k_vector=config_dict.get("top_k_vector", 20),
            final_top_k=config_dict.get("final_top_k", 10),
            rrf_k=config_dict.get("rrf_k", 60),
        )
        return HybridRetriever(bm25_index, vector_store, config)
