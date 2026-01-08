import yaml
from ..llm.gemini_client import GeminiClient

class WriterAgent:
    def __init__(self):
        self.llm = GeminiClient()
        with open("scholar_bridge/config/prompt_templates.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)["writer_agent"]

    async def write_blog(self, insight: dict, paper: dict, niche: str, brand_voice: str, rag_context: str = "") -> str:
        """
        Writes a blog post based on the distilled insight and optional RAG context.
        """
        # System prompt sets the persona
        system_prompt = self.prompts["system_prompt"].format(
            niche=niche,
            brand_voice=brand_voice,
            blog_title="", 
            insight=""
        )
        
        # User input provides the specific task data
        user_input = (
            f"Write a blog post titled 'Why {paper['title']} Matters for {niche}'.\n\n"
            f"Research Insights (Abstract): \n"
            f"- Discovery: {insight.get('main_discovery')}\n"
            f"- Business Impact: {insight.get('business_implication')}\n"
            f"- Key Takeaway: {insight.get('key_takeaway')}\n\n"
        )
        
        if rag_context:
            user_input += (
                f"Deep Dive Facts (From Full PDF):\n"
                f"{rag_context}\n\n"
                f"INSTRUCTION: Incorporate specific metrics, methodologies, or quotes from the 'Deep Dive Facts' to make the post authoritative.\n\n"
            )
            
        user_input += "Follow the structure: Hook, Science, So What?, Action Plan."
        
        response = await self.llm.generate(user_input, system_instruction=system_prompt)
        return response
