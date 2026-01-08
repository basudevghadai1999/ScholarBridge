# Technology Stack & Commands

## Backend (Python)

- **Framework**: FastAPI with Uvicorn server
- **Orchestration**: LangGraph for stateful multi-agent workflows
- **LLM**: Google Gemini (gemini-2.0-flash) via `google-generativeai` SDK
- **Vector DB**: ChromaDB (local embedded) with Google Gecko embeddings
- **PDF Processing**: PyMuPDF (fitz)
- **Web Scraping**: BeautifulSoup4, requests
- **External APIs**: ArXiv API for paper search

## Frontend (JavaScript/React)

- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS (via CDN/inline)
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Markdown**: react-markdown

## Environment Variables

Required in `.env`:
- `GEMINI_API_KEY` - Google Gemini API key

## Common Commands

### Backend
```bash
# Install dependencies
pip install -r scholar_bridge/requirements.txt

# Run API server (port 8000)
python scholar_bridge/server.py

# Run CLI directly
python scholar_bridge/main.py --url "https://example.com"
```

### Frontend
```bash
# Install dependencies
cd frontend && npm install

# Run dev server (port 5173, proxies /api to backend)
cd frontend && npm run dev

# Build for production
cd frontend && npm run build
```

## Configuration Files

- `scholar_bridge/config/model_config.yaml` - LLM settings, ArXiv search params
- `scholar_bridge/config/prompt_templates.yaml` - Agent system prompts
