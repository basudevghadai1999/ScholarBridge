from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.agents.graph import app as graph_app
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
async def run_workflow(request: Request):
    try:
        raw_body = await request.body()
        print(f"DEBUG: Raw Body: {raw_body.decode('utf-8')}")
        data = await request.json()
    except Exception as e:
        print(f"ERROR Parsing JSON: {e}")
        return {"error": "Invalid JSON body", "details": str(e)}
        
    data = data or {} # Handle None
    url = data.get("url")
    mode = data.get("mode", "deep") # fast or deep
    
    if not url:
        return {"error": "URL is required"}
    
    print(f"Server received request for: {url} [Mode: {mode}]")
    
    # Run the LangGraph flow
    inputs = {
        "url": url,
        "mode": mode
    }
    # Invoke
    result = await graph_app.ainvoke(inputs)
    
    # Filter out non-serializable objects if any, though our state is pure JSON-compatible
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
