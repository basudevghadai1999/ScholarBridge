import yaml
import json
from ..llm.gemini_client import GeminiClient
from ..utils.json_parser import extract_json

class PaperFilterAgent:
    def __init__(self):
        self.llm = GeminiClient()
        with open("scholar_bridge/config/prompt_templates.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)["paper_filter_agent"]

    async def filter_papers(self, papers: list, niche: str) -> list:
        """
        Evaluates a list of papers and returns only the relevant ones.
        """
        valid_papers = []
        
        for paper in papers:
            system_prompt = self.prompts["system_prompt"].format(niche=niche)
            user_input = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            
            response = await self.llm.generate(user_input, system_instruction=system_prompt)
            
            try:
                evaluation = extract_json(response)
                if evaluation.get("is_relevant", False):
                    print(f"✅ Accepted: {paper['title']} (Score: {evaluation.get('business_impact_score')})")
                    paper["evaluation"] = evaluation
                    valid_papers.append(paper)
                else:
                    print(f"❌ Rejected: {paper['title']} (Reason: {evaluation.get('reason')})")
            except Exception as e:
                print(f"Error parsing filter response for '{paper['title']}': {e}")
                
        return valid_papers
