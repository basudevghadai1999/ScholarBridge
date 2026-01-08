import google.generativeai as genai
import os
from typing import Dict, Any, Optional
import yaml

class GeminiClient:
    def __init__(self, config_path: str = "scholar_bridge/config/model_config.yaml"):
        # Load Config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["model_settings"]
        
        # Setup API Key
        api_key = os.getenv(self.config["api_key_env_var"])
        if not api_key:
            raise ValueError(f"API Key not found in env var: {self.config['api_key_env_var']}")
        
        genai.configure(api_key=api_key)
        
        # Setup Model
        self.model = genai.GenerativeModel(
            model_name=self.config["model_name"],
            generation_config=genai.GenerationConfig(
                temperature=self.config["temperature"],
                max_output_tokens=self.config["max_output_tokens"]
            )
        )

    async def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generates content using the Gemini model.
        """
        # Note: In the python SDK, system_instruction is often set at model init or 
        # prepended. For simplicity with the standard SDK, we'll prepend if needed 
        # or rely on the prompt structure.
        
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"System Instruction: {system_instruction}\n\nUser Request: {prompt}"
            
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""
