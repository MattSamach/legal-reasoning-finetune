from google import genai
import os
from retrying import retry


class Gemini:
    def __init__(self, model_name, system_prompt, api_key=None):
        self.model_name = model_name
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        print(f"Using model: {model_name}")
        
    def __call__(self, user_prompt: str) -> str:
        """Makes call to Gemini model, returns just the string.

        Args:
            user_prompt (str): Contents of user prompt  

        Returns:
            str: The response text
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents = 
        )
        