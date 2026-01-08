import yaml
import json
from ..llm.gemini_client import GeminiClient
from ..utils.json_parser import extract_json

class SimplifierAgent:
    def __init__(self):
        self.llm = GeminiClient()
        with open("scholar_bridge/config/prompt_templates.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)["simplifier_agent"]

    async def simplify(self, paper: dict, niche: str) -> dict:
        """
        Distills the paper abstract into simple actionable insights.
        """
        system_prompt = self.prompts["system_prompt"].format(
            paper_abstract=paper['abstract'],
            niche=niche
        )
        
        # Passing abstract in user prompt as well to ensure attention
        user_input = f"Simplify this abstract for a business audience in {niche}:\n\n{paper['abstract']}"
        
        response = await self.llm.generate(user_input, system_instruction=system_prompt)
        
        try:
            return extract_json(response)
        except Exception as e:
            print(f"Error simplifying paper '{paper['title']}': {e}")
            return {}
