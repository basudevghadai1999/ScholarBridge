# ScholarBridge

**Research-to-Content Pipeline**: Automatically convert academic research into engaging blog content tailored to your brand's niche and voice.

## Overview

ScholarBridge bridges the gap between academic research and business content by automatically finding relevant papers on ArXiv and translating complex findings into accessible, brand-aligned blog posts.

## Core Workflow

1. **Website Analysis** - Provide a target website URL to extract brand niche and voice
2. **ArXiv Search** - Automatically search for relevant academic papers
3. **Paper Filtering** - Filter papers for business relevance
4. **Content Processing** - Process papers in Fast or Deep mode
5. **Blog Generation** - Generate thought leadership content matching your brand voice

## Processing Modes

- **Fast Mode**: Quick analysis using paper abstracts only
- **Deep Mode**: Downloads full PDFs, ingests into ChromaDB, uses ReAct agent for deeper insights

## Technology Stack

### Backend (Python)
- **Framework**: FastAPI with Uvicorn server
- **Orchestration**: LangGraph for stateful multi-agent workflows
- **LLM**: Google Gemini (gemini-2.0-flash)
- **Vector DB**: ChromaDB with Google Gecko embeddings
- **PDF Processing**: PyMuPDF (fitz)
- **Web Scraping**: BeautifulSoup4, requests
- **External APIs**: ArXiv API

### Frontend (React)
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Markdown**: react-markdown

## Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/basudevghadai1999/ScholarBridge.git
cd ScholarBridge
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. Install backend dependencies:
```bash
pip install -r scholar_bridge/requirements.txt
```

4. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

## Usage

### Running the Full Application

1. Start the backend server (port 8000):
```bash
python scholar_bridge/server.py
```

2. In a new terminal, start the frontend dev server (port 5173):
```bash
cd frontend
npm run dev
```

3. Open your browser to `http://localhost:5173`

### CLI Usage

Run the pipeline directly from command line:
```bash
python scholar_bridge/main.py --url "https://example.com"
```

## Project Structure

```
scholar_bridge/           # Python backend
├── server.py             # FastAPI server entry point
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
├── config/
│   ├── model_config.yaml     # LLM and ArXiv settings
│   └── prompt_templates.yaml # Agent prompts
├── data/
│   └── outputs/          # Generated blog posts
└── src/
    ├── agents/           # LangGraph agent implementations
    ├── llm/              # Gemini API wrapper
    ├── chunking/         # Semantic chunking & metadata
    ├── retrieval/        # Hybrid retrieval & reranking
    └── utils/            # Helper utilities

frontend/                 # React frontend
├── src/
│   ├── App.jsx           # Main application component
│   └── main.jsx          # React entry point
└── vite.config.js        # Vite config with API proxy
```

## Configuration

- `scholar_bridge/config/model_config.yaml` - LLM settings, ArXiv search parameters
- `scholar_bridge/config/prompt_templates.yaml` - Agent system prompts (modify to adjust agent behavior)

## Features

- 🔍 Automated ArXiv paper discovery
- 🎯 Brand voice extraction and matching
- 📄 Full PDF processing with RAG
- 🤖 Multi-agent workflow orchestration
- 💬 Interactive web interface
- ⚡ Fast and Deep processing modes
- 📝 Markdown blog post generation

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
