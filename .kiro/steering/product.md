# ScholarBridge - Product Overview

ScholarBridge is a "Research-to-Content" pipeline that automates converting academic research into engaging blog content tailored to a brand's niche and voice.

## Core Workflow

1. User provides a target website URL
2. System analyzes the website to extract brand niche and voice
3. Searches ArXiv for relevant academic papers
4. Filters papers for business relevance
5. Processes the best paper (Fast mode: abstract only, Deep mode: full PDF via RAG)
6. Generates a thought leadership blog post matching the brand voice

## Two Processing Modes

- **Fast Mode**: Quick analysis using paper abstracts only
- **Deep Mode**: Downloads full PDFs, ingests into ChromaDB, uses ReAct agent for deeper insights

## Key Value Proposition

Bridges the gap between academic research and business content by automatically finding relevant papers and translating complex findings into accessible, brand-aligned blog posts.
