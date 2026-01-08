import requests
from bs4 import BeautifulSoup
from ..llm.gemini_client import GeminiClient
from ..utils.json_parser import extract_json
import yaml
import json

class WebsiteAnalysisAgent:
    def __init__(self):
        self.llm = GeminiClient()
        with open("scholar_bridge/config/prompt_templates.yaml", "r") as f:
            self.prompts = yaml.safe_load(f)["website_analysis_agent"]
    
    # ... (scrape_text method remains unchanged) ...

    def scrape_text(self, url: str) -> str:
        """
        Scrapes visible text from a URL.
        """
        try:
            # Modern User Agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            # If 403/Forbidden, we can't scrape, but we shouldn't crash the whole app
            if response.status_code in [403, 401]:
                print(f"⚠️ Access denied (403) for {url}. Using fallback.")
                return ""
                
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return clean_text[:5000]
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    async def analyze(self, url: str) -> dict:
        """
        Analyzes the website and returns Brand/Niche info.
        """
        print(f"Analyzing website: {url}...")
        website_text = self.scrape_text(url)
        
        # Fallback if scraping failed
        if not website_text:
            print("⚠️ Website scraping failed. Defaulting to general Technology niche.")
            return {
                "niche": "Artificial Intelligence & Technology",
                "brand_voice": "Professional and Innovative",
                "value_props": ["Innovation", "Technology"]
            }

        system_prompt = self.prompts["system_prompt"]
        user_prompt = f"Analyze this website content:\n\n{website_text}"
        
        response_text = await self.llm.generate(user_prompt, system_instruction=system_prompt)
        
        try:
            return extract_json(response_text)
        except Exception:
            print("Error parsing JSON response")
            return {"raw_response": response_text}
