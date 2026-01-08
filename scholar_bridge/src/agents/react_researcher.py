import re
from ..llm.gemini_client import GeminiClient
from .rag_engine import RagEngine

class ReactResearcherAgent:
    def __init__(self, rag_engine: RagEngine):
        self.llm = GeminiClient()
        self.rag = rag_engine
        
    async def research(self, paper_title: str, collection_name: str) -> str:
        """
        Performs a deep dive research loop on the paper.
        """
        print(f"[ReAct] Starting autonomous research on: {paper_title}")
        
        system_prompt = (
            "You are a Senior Technical Researcher. Your goal is to extract specific, high-value technical details from a research paper to inform a business blog post.\n"
            "You have access to the full paper text via a tool.\n\n"
            "TOOLS:\n"
            "1. QUERY: search_term\n"
            "   (Searches the PDF for the term and returns relevant paragraphs)\n\n"
            "PROTOCOL:\n"
            "- Start by searching for the main methodology or algorithm.\n"
            "- Observe the results, then ask follow-up questions about specific metrics, parameters, or limitations.\n"
            "- Perform at least 3 queries to get different angles.\n"
            "- When you have enough information (Methodology, Key Results, Limitations), output 'FINAL REPORT:' followed by your summary.\n\n"
            "FORMAT:\n"
            "THOUGHT: <your reasoning>\n"
            "ACTION: QUERY: <search term>\n"
            "OBSERVATION: <tool output>\n"
            "... (repeat) ...\n"
            "FINAL REPORT: <summary>"
        )
        
        history = f"Target Paper: {paper_title}\n\n"
        max_steps = 6
        
        for step in range(max_steps):
            # 1. Think
            response = await self.llm.generate(history, system_instruction=system_prompt)
            clean_response = response.strip()
            
            # Print for debugging/log
            print(f"\n[ReAct Step {step+1}]")
            print(clean_response)
            
            history += f"\n{clean_response}\n"
            
            # 2. Act
            if "FINAL REPORT:" in clean_response:
                # We are done
                return clean_response.split("FINAL REPORT:", 1)[1].strip()
            
            if "QUERY:" in clean_response:
                # Parse query using Regex to find "QUERY: <term>"
                # Matches "ACTION: QUERY: term" or just "QUERY: term"
                match = re.search(r"QUERY:\s*(.*)", clean_response)
                if match:
                    query_term = match.group(1).split("\n")[0].strip() # Take just the line
                    
                    print(f"   >>> Tool Call: Searching PDF for '{query_term}'...")
                    tool_result = self.rag.query(collection_name, query_term, n_results=2)
                    
                    observation = f"OBSERVATION: {tool_result}"
                    history += f"\n{observation}\n"
                else:
                    history += "\nOBSERVATION: Could not parse QUERY command. Please use format 'QUERY: <term>'\n"
            else:
                # If the model didn't use a tool but didn't finish, nudge it
                history += "\nOBSERVATION: Please use a QUERY or provide FINAL REPORT.\n"
                
        # Fallback if loop ends
        return "Research incomplete. " + history[-500:]
