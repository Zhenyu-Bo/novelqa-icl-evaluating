from abc import ABC, abstractmethod
from google import genai
from google.genai import types
import openai
import os

class LLM(ABC):
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class Gemini2Flash(LLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.system_prompt = ""
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,),
            contents=[prompt]
        )
        return response.text
    
class Deepseek(LLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.system_prompt = ""
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

def get_llm(llm_name: str, api_key: str | None = None) -> LLM:
    if llm_name == "gemini":
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY is not set")
        return Gemini2Flash(api_key)
    elif llm_name == "deepseek":
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key is None:
                raise ValueError("DEEPSEEK_API_KEY is not set")
        return Deepseek(api_key)
    else:
        raise ValueError(f"Unknown LLM: {llm_name}")
