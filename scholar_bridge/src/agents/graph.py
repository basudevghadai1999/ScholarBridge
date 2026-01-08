from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Import our existing agents
from .website_analysis import WebsiteAnalysisAgent
from .arxiv_search import ArxivSearchAgent
from .paper_filter import PaperFilterAgent
from .simplifier import SimplifierAgent
from .writer import WriterAgent
from .rag_engine import RagEngine  # [NEW]

# 1. Define State
class ScholarState(TypedDict):
    url: str
    mode: Optional[str] # [NEW] "fast" or "deep"
    niche: Optional[str]
    brand_voice: Optional[str]
    raw_papers: List[dict]
    best_paper: Optional[dict]
    insight: Optional[dict]
    rag_context: Optional[str] 
    final_blog: Optional[str]

from .react_researcher import ReactResearcherAgent

# 2. Define Nodes
async def analyze_website(state: ScholarState):
    agent = WebsiteAnalysisAgent()
    print(f"\n[Graph] Analyzing: {state['url']}")
    result = await agent.analyze(state['url'])
    
    if "error" in result:
        pass
        
    return {
        "niche": result.get("niche", "Technology"), 
        "brand_voice": result.get("brand_voice", "Professional")
    }

def search_arxiv(state: ScholarState):
    agent = ArxivSearchAgent()
    print(f"\n[Graph] Searching ArXiv for: {state['niche']}")
    papers = agent.search_papers(state["niche"])
    return {"raw_papers": papers}

async def filter_papers(state: ScholarState):
    agent = PaperFilterAgent()
    print(f"\n[Graph] Filtering {len(state['raw_papers'])} papers...")
    valid_papers = await agent.filter_papers(state["raw_papers"], state["niche"])
    
    best_paper = valid_papers[0] if valid_papers else None
    return {"best_paper": best_paper}

async def simplify_paper(state: ScholarState):
    agent = SimplifierAgent()
    print(f"\n[Graph] Simplifying Abstract: {state['best_paper']['title']}")
    insight = await agent.simplify(state["best_paper"], state["niche"])
    return {"insight": insight}

async def write_blog(state: ScholarState):
    agent = WriterAgent()
    print(f"\n[Graph] Writing Blog Post...")
    
    blog = await agent.write_blog(
        state["insight"], 
        state["best_paper"], 
        state["niche"], 
        state["brand_voice"],
        rag_context=state.get("rag_context", "") 
    )
    return {"final_blog": blog}

# [NEW] Vector RAG Node with ReAct
async def deep_dive(state: ScholarState):
    paper = state["best_paper"]
    print(f"\n[Graph] 🤿 Deep Deep into PDF: {paper['title']}")
    
    engine = RagEngine()
    
    # Download & Ingest
    pdf_url = paper['url']
    path = engine.download_pdf(pdf_url)
    if not path:
        return {"rag_context": "Could not download PDF."}
        
    collection_name = engine.ingest_paper(path)
    if not collection_name:
        return {"rag_context": "Could not ingest PDF."}
        
    # Run ReAct Loop
    researcher = ReactResearcherAgent(engine)
    context = await researcher.research(paper['title'], collection_name)
    
    return {"rag_context": context}

# 3. Define Conditional Logic
def route_after_filter(state: ScholarState):
    if not state.get("best_paper"):
        return "end"
    
    # Check mode
    if state.get("mode") == "deep":
        return "deep_dive"
        
    # Default to fast/simplify if not deep
    return "simplify"

# 4. Build Graph
workflow = StateGraph(ScholarState)

workflow.add_node("analyze_website", analyze_website)
workflow.add_node("search_arxiv", search_arxiv)
workflow.add_node("filter_papers", filter_papers)
workflow.add_node("deep_dive", deep_dive) 
workflow.add_node("simplify_paper", simplify_paper)
workflow.add_node("write_blog", write_blog)

workflow.set_entry_point("analyze_website")

workflow.add_edge("analyze_website", "search_arxiv")
workflow.add_edge("search_arxiv", "filter_papers")

# Branching logic
workflow.add_conditional_edges(
    "filter_papers",
    route_after_filter,
    {
        "deep_dive": "deep_dive",
        "simplify": "simplify_paper",
        "end": END
    }
)

# Deep dive re-joins the main track
workflow.add_edge("deep_dive", "simplify_paper") 
workflow.add_edge("simplify_paper", "write_blog")
workflow.add_edge("write_blog", END)

# Compile
app = workflow.compile()
