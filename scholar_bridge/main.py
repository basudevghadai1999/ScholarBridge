import asyncio
import argparse
import os
from termcolor import colored
from dotenv import load_dotenv
import asyncio
import argparse
import os
from termcolor import colored
from dotenv import load_dotenv
from scholar_bridge.src.agents.graph import app  # Import the compiled graph

# Load env vars
load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="ScholarBridge: Turn Research into Content (LangGraph)")
    parser.add_argument("--url", type=str, required=True, help="Target website URL to analyze brand")
    args = parser.parse_args()

    # Check API Key
    if not os.getenv("GEMINI_API_KEY"):
        print(colored("Error: GEMINI_API_KEY not found in environment variables.", "red"))
        return

    print(colored("Starting LangGraph Workflow...", "cyan"))
    
    # Run the Graph
    # Note: LangGraph execute is usually sync or async depending on configuration.
    # Since our nodes are async, we use ainvoke
    inputs = {"url": args.url}
    result = await app.ainvoke(inputs)
    
    print("\n\n" + "="*50)
    
    if result.get("final_blog"):
        print(colored("✨ FINAL BLOG POST ✨", "green"))
        print("="*50 + "\n")
        print(result["final_blog"])
        
        # Save to output
        os.makedirs("scholar_bridge/data/outputs", exist_ok=True)
        with open("scholar_bridge/data/outputs/latest_blog.md", "w") as f:
            f.write(str(result["final_blog"]))
        print(colored(f"\nSaved to scholar_bridge/data/outputs/latest_blog.md", "blue"))
    else:
        print(colored("❌ Workflow ended without producing a blog post (likely no relevant papers found).", "red"))


if __name__ == "__main__":
    asyncio.run(main())
