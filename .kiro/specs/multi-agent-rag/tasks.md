                                                                                                                                      # Multi-Agent RAG System - Implementation Tasks

## Phase 1: Advanced Chunking Infrastructure

### Task 1.1: Create Chunking Module Structure
- [x] Create `scholar_bridge/src/chunking/__init__.py`
- [x] Create `scholar_bridge/src/chunking/semantic_chunker.py`
- [x] Create `scholar_bridge/src/chunking/overlap_manager.py`
- [x] Create `scholar_bridge/src/chunking/metadata_enricher.py`

### Task 1.2: Implement Semantic Chunker
- [x] Add section detection regex patterns for academic papers
- [x] Implement `detect_sections(text)` to identify paper structure
- [x] Implement `recursive_split_with_sections(text, sections)` 
- [x] Add configurable chunk_size and overlap_percent parameters
- [x] Return `List[Chunk]` with full metadata

### Task 1.3: Implement Overlap Manager
- [x] Create `OverlapManager` class
- [x] Implement `calculate_overlap(chunk_a, chunk_b)` 
- [x] Implement `apply_overlap(chunks, overlap_percent)`
- [x] Track overlap_with_prev and overlap_with_next in metadata
- [x] Handle edge cases (first/last chunks, small chunks)

### Task 1.4: Implement Metadata Enricher
- [x] Create `MetadataEnricher` class
- [x] Extract page numbers from PDF metadata
- [x] Assign section labels based on detection
- [x] Generate chunk IDs with hierarchical structure
- [x] Optional: LLM-based semantic tagging (configurable)

### Task 1.5: Update Configuration
- [x] Add `chunking` section to `model_config.yaml`
- [x] Add chunk_size, overlap_percent, min_chunk_size settings
- [x] Add section_detection toggle
- [x] Add semantic_tagging toggle

---

## Phase 2: Hybrid Retrieval System

### Task 2.1: Create Retrieval Module Structure
- [x] Create `scholar_bridge/src/retrieval/__init__.py`
- [x] Create `scholar_bridge/src/retrieval/bm25_index.py`
- [x] Create `scholar_bridge/src/retrieval/hybrid_retriever.py`
- [x] Add `rank_bm25` to requirements.txt

### Task 2.2: Implement BM25 Index
- [x] Create `BM25Index` class
- [x] Implement `build_index(chunks)` using rank_bm25
- [x] Implement `search(query, top_k)` returning scored results
- [x] Store index alongside ChromaDB collection
- [x] Handle tokenization for academic text

### Task 2.3: Implement Hybrid Retriever
- [x] Create `HybridRetriever` class
- [x] Implement `retrieve(query, collection_name)` 
- [x] Call both BM25 and vector search
- [x] Implement `rrf_fusion(bm25_results, vector_results, k=60)`
- [x] Return merged and scored results
- [x] Add configurable weights for BM25 vs vector

### Task 2.4: Implement Reranker
- [x] Create `scholar_bridge/src/retrieval/reranker.py`
- [x] Implement LLM-based relevance scoring
- [x] Implement diversity filtering (avoid redundant chunks)
- [x] Apply relevance threshold filtering
- [x] Manage context window limits

### Task 2.5: Update RAG Engine
- [-] Modify `rag_engine.py` to use new SemanticChunker
- [ ] Integrate BM25Index alongside ChromaDB
- [ ] Replace simple query with HybridRetriever
- [ ] Add reranking step before returning context

### Task 2.6: Update Retrieval Configuration
- [ ] Add `retrieval` section to `model_config.yaml`
- [ ] Add bm25_weight, vector_weight settings
- [ ] Add top_k settings for each retrieval method
- [ ] Add rrf_k parameter
- [ ] Add reranking settings

---

## Phase 3: Agentic RAG Components

### Task 3.1: Create Query Planner Agent
- [ ] Create `scholar_bridge/src/agents/query_planner.py`
- [ ] Define `QueryPlannerOutput` TypedDict
- [ ] Implement `plan_queries(research_goal, paper_title)`
- [ ] Decompose into sub-queries with types (factual, conceptual, methodological)
- [ ] Add prompt template to `prompt_templates.yaml`

### Task 3.2: Create Retriever Agent
- [ ] Create `scholar_bridge/src/agents/retriever_agent.py`
- [ ] Define `RetrieverState` TypedDict
- [ ] Implement `retrieve_with_reflection(query, collection)`
- [ ] Add self-assessment of retrieval quality
- [ ] Implement retry logic for insufficient context
- [ ] Add prompt template for reflection

### Task 3.3: Create Reranker Agent
- [ ] Create `scholar_bridge/src/agents/reranker_agent.py`
- [ ] Define `RerankerOutput` TypedDict
- [ ] Implement `rerank_and_assess(chunks, query)`
- [ ] Assess coverage and diversity
- [ ] Signal if more retrieval needed
- [ ] Add prompt template

### Task 3.4: Create Synthesizer Agent
- [ ] Create `scholar_bridge/src/agents/synthesizer.py`
- [ ] Implement `synthesize_context(chunks, query, paper_info)`
- [ ] Combine retrieved chunks into coherent narrative
- [ ] Extract key facts, metrics, and quotes
- [ ] Format for downstream writer agent
- [ ] Add prompt template

### Task 3.5: Update Agent Prompts
- [ ] Add `query_planner_agent` section to `prompt_templates.yaml`
- [ ] Add `retriever_agent` section
- [ ] Add `reranker_agent` section
- [ ] Add `synthesizer_agent` section

---

## Phase 4: Multi-Agent Orchestration

### Task 4.1: Create Supervisor Agent
- [ ] Create `scholar_bridge/src/agents/supervisor.py`
- [ ] Define `SupervisorState` TypedDict
- [ ] Define `MultiAgentState` TypedDict (extends ScholarState)
- [ ] Implement `supervisor_node(state)` decision logic
- [ ] Implement `route_to_agent(state)` conditional routing
- [ ] Add supervisor prompt template

### Task 4.2: Define Agent Communication Protocol
- [ ] Define standard agent output format
- [ ] Implement `agent_outputs` accumulator in state
- [ ] Create helper functions for state updates
- [ ] Add logging for agent handoffs

### Task 4.3: Refactor Existing Agents
- [ ] Update `website_analysis.py` to return standardized output
- [ ] Update `arxiv_search.py` to return standardized output
- [ ] Update `paper_filter.py` to return standardized output
- [ ] Update `simplifier.py` to accept synthesized context
- [ ] Update `writer.py` to accept enriched context

### Task 4.4: Build New LangGraph Workflow
- [ ] Modify `graph.py` with new MultiAgentState
- [ ] Add all new agent nodes
- [ ] Implement supervisor routing logic
- [ ] Add RAG sub-pipeline (query_planner → retriever → reranker → synthesizer)
- [ ] Add conditional edges for retrieval loops
- [ ] Connect all agents back to supervisor
- [ ] Set terminal conditions

### Task 4.5: Update Agent Configuration
- [ ] Add `agents` section to `model_config.yaml`
- [ ] Add max_retrieval_loops setting
- [ ] Add supervisor_max_iterations setting

---

## Phase 5: Integration and Testing

### Task 5.1: Integration Testing
- [ ] Test chunking pipeline with sample PDF
- [ ] Test hybrid retrieval with sample queries
- [ ] Test full agent workflow end-to-end
- [ ] Verify state propagation between agents
- [ ] Test fallback behaviors

### Task 5.2: Update Dependencies
- [ ] Add `rank_bm25` to requirements.txt
- [ ] Verify all imports work correctly
- [ ] Test with fresh virtual environment

### Task 5.3: Documentation
- [ ] Update architecture.md with new system design
- [ ] Add inline code comments for complex logic
- [ ] Update steering files if needed

---

## Task Dependencies

```
Phase 1 (Chunking) ──┐
                     ├──► Phase 2 (Retrieval) ──┐
                     │                          │
                     │                          ├──► Phase 4 (Orchestration) ──► Phase 5 (Integration)
                     │                          │
Phase 3 (Agents) ────┴──────────────────────────┘
```

- Phase 1 and Phase 3 can be done in parallel
- Phase 2 depends on Phase 1 (chunking)
- Phase 4 depends on Phase 2 and Phase 3
- Phase 5 depends on Phase 4

## Estimated Effort

| Phase | Tasks | Complexity | Estimate |
|-------|-------|------------|----------|
| Phase 1 | 5 | Medium | 2-3 hours |
| Phase 2 | 6 | High | 3-4 hours |
| Phase 3 | 5 | Medium | 2-3 hours |
| Phase 4 | 5 | High | 3-4 hours |
| Phase 5 | 3 | Low | 1-2 hours |
| **Total** | **24** | | **11-16 hours** |
