# Project Structure

```
scholar_bridge/           # Python backend
├── server.py             # FastAPI server entry point
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
├── config/
│   ├── model_config.yaml     # LLM and ArXiv settings
│   └── prompt_templates.yaml # Agent prompts (YAML format)
├── data/
│   └── outputs/          # Generated blog posts
└── src/
    ├── agents/           # LangGraph agent implementations
    │   ├── graph.py          # Main workflow graph (StateGraph)
    │   ├── website_analysis.py
    │   ├── arxiv_search.py
    │   ├── paper_filter.py
    │   ├── simplifier.py
    │   ├── writer.py
    │   ├── rag_engine.py     # PDF ingestion & ChromaDB
    │   └── react_researcher.py
    ├── llm/
    │   └── gemini_client.py  # Gemini API wrapper
    └── utils/
        └── json_parser.py    # JSON extraction from LLM responses

frontend/                 # React frontend
├── index.html
├── package.json
├── vite.config.js        # Vite config with API proxy
└── src/
    ├── main.jsx          # React entry point
    ├── App.jsx           # Main application component
    └── index.css         # Global styles
```

## Architecture Patterns

- **Agent Pattern**: Each agent is a class with async methods, uses `GeminiClient` for LLM calls
- **State Machine**: LangGraph `StateGraph` with `ScholarState` TypedDict manages workflow
- **Config-Driven Prompts**: All agent prompts stored in YAML, loaded at runtime
- **API Proxy**: Frontend proxies `/api/*` to backend at `localhost:8000`

## Key Files to Know

- `scholar_bridge/src/agents/graph.py` - Central workflow definition, add new nodes here
- `scholar_bridge/config/prompt_templates.yaml` - Modify agent behavior via prompts
- `frontend/src/App.jsx` - Single-page React app, all UI logic
