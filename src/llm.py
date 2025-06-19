from abc import ABC, abstractmethod
from typing import Iterable
from google import genai
from google.genai import types
# import google.generativeai as genai
# from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold # 导入必要的类型
import openai
import os
import logging
import traceback
import httpx

class LLM(ABC):
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class Gemini(LLM):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.system_prompt = ""
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=1.0,
                top_p=0.95,
                max_output_tokens=8192,),
            contents=[prompt]
        )
        # print(response)
        return response.text


class Deepseek(LLM):
    def __init__(self, api_key: str, reason: bool = False):
        self.api_key = api_key
        self.system_prompt = ""
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.reason = reason

    def generate(self, prompt: str) -> str:
        if self.reason:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            reasoning_content = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content
            return f"Reasoning: {reasoning_content}\n\nAnswer: {content}"
            # return response.choices[0].message.content
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
    if llm_name == "gemini2.0":
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY is not set")
        return Gemini(api_key, model="gemini-2.0-flash")
    elif llm_name == "gemini2.5-flash":
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY is not set")
        return Gemini(api_key, model="gemini-2.5-flash-preview-05-20")
    elif llm_name == "gemini2.5-pro":
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError("GEMINI_API_KEY is not set")
        return Gemini(api_key, model="gemini-2.5-pro-preview-03-25")
    elif llm_name == "deepseek":
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key is None:
                raise ValueError("DEEPSEEK_API_KEY is not set")
        return Deepseek(api_key)
    elif llm_name == "deepseek-r1":
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key is None:
                raise ValueError("DEEPSEEK_API_KEY is not set")
        return Deepseek(api_key, reason=True)
    else:
        raise ValueError(f"Unknown LLM: {llm_name}")
