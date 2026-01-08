# Multi-Agent RAG System - Design Document

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUPERVISOR AGENT                                   │
│                    (Orchestrates workflow, delegates tasks)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   RESEARCH    │          │  RETRIEVAL    │          │   SYNTHESIS   │
│    AGENTS     │          │    AGENTS     │          │    AGENTS     │
├───────────────┤          ├───────────────┤          ├───────────────┤
│ • Website     │          │ • Query       │          │ • Simplifier  │
│   Analyzer    │          │   Planner     │          │ • Writer      │
│ • ArXiv       │          │ • Retriever   │          │ • Fact        │
│   Searcher    │          │ • Reranker    │          │   Checker     │
│ • Paper       │          │               │          │               │
│   Filter      │          │               │          │               │
└───────────────┘          └───────────────┘          └───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        RAG ENGINE             │
                    ├───────────────────────────────┤
                    │ • Semantic Chunker            │
                    │ • Vector Store (ChromaDB)     │
                    │ • BM25 Index                  │
                    │ • Hybrid Retriever            │
                    │ • Context Reranker            │
                    └───────────────────────────────┘
```

## Component Design

### 1. Supervisor Agent

```python
class SupervisorState(TypedDict):
    messages: List[dict]
    current_task: str
    agent_outputs: Dict[str, Any]
    next_agent: str
    iteration: int
    max_iterations: int
    final_output: Optional[str]
```

The supervisor:
- Receives initial request (URL + mode)
- Decides which agent to invoke next
- Aggregates agent outputs
- Determines when workflow is complete

### 2. Agentic RAG Components

#### 2.1 Query Planner Agent
Decomposes complex research questions into targeted sub-queries.

```python
class QueryPlannerOutput(TypedDict):
    sub_queries: List[str]           # Decomposed queries
    query_types: List[str]           # "factual", "conceptual", "methodological"
    priority_order: List[int]        # Execution order
    reasoning: str                   # Why this decomposition
```

#### 2.2 Retriever Agent
Executes hybrid retrieval with self-reflection.

```python
class RetrieverState(TypedDict):
    query: str
    retrieved_chunks: List[dict]
    relevance_scores: List[float]
    needs_more_context: bool
    retrieval_attempts: int
```

#### 2.3 Reranker Agent
Reorders and filters retrieved chunks.

```python
class RerankerOutput(TypedDict):
    reranked_chunks: List[dict]
    diversity_score: float
    coverage_assessment: str
```

### 3. Advanced Chunking Strategy

#### 3.1 Semantic Chunker

```python
class SemanticChunker:
    def __init__(self, config: ChunkConfig):
        self.chunk_size = config.chunk_size          # 1000 chars
        self.overlap_percent = config.overlap        # 20%
        self.min_chunk_size = config.min_size        # 100 chars
        
    def chunk(self, text: str, doc_type: str = "paper") -> List[Chunk]:
        # 1. Detect sections (Abstract, Introduction, Methods, etc.)
        # 2. Split within sections using recursive splitter
        # 3. Apply overlap between chunks
        # 4. Enrich with metadata
        pass
```

#### 3.2 Chunk Schema

```python
class Chunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata
    
class ChunkMetadata(TypedDict):
    source: str
    page: int
    section: str                    # "abstract", "methods", "results", etc.
    chunk_index: int
    parent_chunk_id: Optional[str]  # For hierarchical retrieval
    overlap_with_prev: int          # Characters overlapping with previous
    overlap_with_next: int          # Characters overlapping with next
    semantic_tags: List[str]        # LLM-generated topic tags
```

#### 3.3 Section Detection Patterns

```python
SECTION_PATTERNS = {
    "abstract": r"^abstract\s*$|^summary\s*$",
    "introduction": r"^1\.?\s*introduction|^introduction\s*$",
    "methods": r"^2\.?\s*method|^materials?\s+and\s+methods?|^methodology",
    "results": r"^3\.?\s*results?|^findings",
    "discussion": r"^4\.?\s*discussion|^analysis",
    "conclusion": r"^5\.?\s*conclusion|^concluding",
    "references": r"^references?\s*$|^bibliography"
}
```

### 4. Hybrid Retrieval

#### 4.1 Retrieval Pipeline

```
Query → Query Expansion → [BM25 Search] + [Vector Search] → RRF Fusion → Rerank → Top-K
```

#### 4.2 Reciprocal Rank Fusion

```python
def rrf_fusion(bm25_results: List, vector_results: List, k: int = 60) -> List:
    """
    RRF Score = Σ 1 / (k + rank_i)
    """
    scores = defaultdict(float)
    
    for rank, doc in enumerate(bm25_results):
        scores[doc.id] += 1 / (k + rank + 1)
        
    for rank, doc in enumerate(vector_results):
        scores[doc.id] += 1 / (k + rank + 1)
        
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5. LangGraph Workflow

```python
# New graph structure
workflow = StateGraph(MultiAgentState)

# Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("website_analyzer", website_analyzer_node)
workflow.add_node("arxiv_searcher", arxiv_searcher_node)
workflow.add_node("paper_filter", paper_filter_node)
workflow.add_node("query_planner", query_planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("writer", writer_node)

# Entry
workflow.set_entry_point("supervisor")

# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "website_analyzer": "website_analyzer",
        "arxiv_searcher": "arxiv_searcher",
        "paper_filter": "paper_filter",
        "rag_pipeline": "query_planner",
        "writer": "writer",
        "end": END
    }
)

# RAG sub-pipeline
workflow.add_edge("query_planner", "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_conditional_edges(
    "reranker",
    check_context_quality,
    {
        "sufficient": "synthesizer",
        "insufficient": "retriever"  # Loop back
    }
)
workflow.add_edge("synthesizer", "supervisor")

# Other edges back to supervisor
workflow.add_edge("website_analyzer", "supervisor")
workflow.add_edge("arxiv_searcher", "supervisor")
workflow.add_edge("paper_filter", "supervisor")
workflow.add_edge("writer", END)
```

## File Structure

```
scholar_bridge/src/
├── agents/
│   ├── supervisor.py          # NEW: Orchestration agent
│   ├── query_planner.py       # NEW: Query decomposition
│   ├── retriever.py           # NEW: Hybrid retrieval agent
│   ├── reranker.py            # NEW: Context reranking
│   ├── synthesizer.py         # NEW: Context synthesis
│   ├── graph.py               # MODIFIED: New multi-agent graph
│   ├── website_analysis.py    # EXISTING
│   ├── arxiv_search.py        # EXISTING
│   ├── paper_filter.py        # EXISTING
│   ├── simplifier.py          # EXISTING
│   ├── writer.py              # MODIFIED: Accept synthesized context
│   └── rag_engine.py          # MODIFIED: Add hybrid retrieval
├── chunking/
│   ├── __init__.py            # NEW
│   ├── semantic_chunker.py    # NEW: Section-aware chunking
│   ├── overlap_manager.py     # NEW: Overlap calculation
│   └── metadata_enricher.py   # NEW: Chunk metadata
└── retrieval/
    ├── __init__.py            # NEW
    ├── bm25_index.py          # NEW: Keyword search
    ├── hybrid_retriever.py    # NEW: RRF fusion
    └── reranker.py            # NEW: Cross-encoder reranking
```

## Configuration Updates

### model_config.yaml additions

```yaml
chunking:
  chunk_size: 1000
  overlap_percent: 20
  min_chunk_size: 100
  section_detection: true
  semantic_tagging: false  # LLM-based, expensive

retrieval:
  bm25_weight: 0.3
  vector_weight: 0.7
  top_k_bm25: 20
  top_k_vector: 20
  final_top_k: 10
  rrf_k: 60
  
reranking:
  enabled: true
  diversity_threshold: 0.7
  relevance_threshold: 0.5
  max_context_tokens: 4000

agents:
  max_retrieval_loops: 3
  supervisor_max_iterations: 10
```

## API Changes

No changes to external API. The `/api/run` endpoint continues to accept:
```json
{
  "url": "https://example.com",
  "mode": "deep"
}
```

Internal processing changes are transparent to the frontend.
