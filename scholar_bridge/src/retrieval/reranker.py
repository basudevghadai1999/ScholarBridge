"""
Reranker for relevance-based result reordering.

Provides LLM-based reranking and diversity filtering for retrieved chunks.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict

from . import RetrievalResult, RerankConfig


class Reranker:
    """
    LLM-based reranker for retrieved chunks.
    
    Reorders chunks by relevance to the query and filters
    for diversity to avoid redundant context.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[RerankConfig] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            llm_client: LLM client for relevance scoring.
            config: Reranking configuration.
        """
        self.llm_client = llm_client
        self.config = config or RerankConfig()
    
    async def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Rerank results by relevance to query.
        
        Args:
            results: Retrieved results to rerank.
            query: Original search query.
            top_k: Number of results to return.
            
        Returns:
            Reranked results.
        """
        if not self.config.enabled or not results:
            return results[:top_k] if top_k else results
        
        # Score each result for relevance
        scored_results = await self._score_relevance(results, query)
        
        # Filter by relevance threshold
        filtered = [
            r for r in scored_results 
            if r["score"] >= self.config.relevance_threshold
        ]
        
        # Apply diversity filtering
        diverse_results = self._filter_diversity(filtered)
        
        # Limit to context window
        limited = self._limit_context(diverse_results)
        
        return limited[:top_k] if top_k else limited
    
    async def _score_relevance(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        Score results for relevance using LLM.
        
        Args:
            results: Results to score.
            query: Search query.
            
        Returns:
            Results with updated relevance scores.
        """
        if not self.llm_client:
            return results
        
        scored = []
        
        for result in results:
            try:
                score = await self._get_relevance_score(result["text"], query)
                new_result = result.copy()
                new_result["score"] = score
                scored.append(new_result)
            except Exception:
                scored.append(result)
        
        # Sort by new scores
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
    
    async def _get_relevance_score(self, text: str, query: str) -> float:
        """
        Get relevance score for a single chunk.
        
        Args:
            text: Chunk text.
            query: Search query.
            
        Returns:
            Relevance score between 0 and 1.
        """
        prompt = f"""Rate the relevance of this text to the query on a scale of 0-10.
        
Query: {query}

Text: {text[:500]}...

Respond with only a number between 0 and 10."""

        try:
            response = await self.llm_client.generate(prompt)
            score = float(response.strip()) / 10.0
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5  # Default score on error
    
    def _filter_diversity(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Filter results for diversity to avoid redundancy.
        
        Args:
            results: Results to filter.
            
        Returns:
            Diverse subset of results.
        """
        if not results:
            return []
        
        diverse = [results[0]]
        
        for result in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            for selected in diverse:
                similarity = self._text_similarity(result["text"], selected["text"])
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse.append(result)
        
        return diverse
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using word overlap.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _limit_context(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Limit results to fit within context window.
        
        Args:
            results: Results to limit.
            
        Returns:
            Results fitting within token limit.
        """
        limited = []
        total_tokens = 0
        
        for result in results:
            # Rough token estimate (4 chars per token)
            chunk_tokens = len(result["text"]) // 4
            
            if total_tokens + chunk_tokens <= self.config.max_context_tokens:
                limited.append(result)
                total_tokens += chunk_tokens
            else:
                break
        
        return limited
    
    @staticmethod
    def from_config_dict(
        config_dict: dict,
        llm_client: Optional[Any] = None
    ) -> "Reranker":
        """
        Create a Reranker from a configuration dictionary.
        
        Args:
            config_dict: Dictionary with reranking configuration.
            llm_client: Optional LLM client.
            
        Returns:
            Configured Reranker instance.
        """
        config = RerankConfig(
            enabled=config_dict.get("enabled", True),
            diversity_threshold=config_dict.get("diversity_threshold", 0.7),
            relevance_threshold=config_dict.get("relevance_threshold", 0.5),
            max_context_tokens=config_dict.get("max_context_tokens", 4000),
        )
        return Reranker(llm_client, config)
