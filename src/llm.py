from abc import ABC, abstractmethod
from typing import Iterable
from google import genai
# import google.generativeai as genai
# from google.generativeai import types
# from google.generativeai.types import HarmCategory, HarmBlockThreshold, RequestOptions # 注意：GenerationConfig 不再从这里导入
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold # 导入必要的类型
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
        self.model_name = model
        self.system_prompt = ""
        self._init_client()

    def _init_client(self):
        """初始化或重新初始化模型客户端。"""
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        if self.system_prompt:
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt,
                safety_settings=safety_settings
            )
        else:
            self.client = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=safety_settings
            )

    def set_system_prompt(self, system_prompt: str):
        """设置系统提示并重新初始化客户端。"""
        self.system_prompt = system_prompt
        self._init_client()

    def generate(self, prompt: str, stream_internally_log: bool = False) -> str: # 返回类型是 str
        if not hasattr(self, 'client') or self.client is None:
            self._init_client()

        generation_config = GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,
        )

        full_response_parts = [] # 用于收集所有文本块

        try:
            request_options = {"timeout": 300.0}  # 设置超时时间为 5 分钟

            response_stream = self.client.generate_content(
                contents=[prompt],
                generation_config=generation_config,
                request_options=request_options,
                stream=True # 内部使用流式获取
            )

            if stream_internally_log:
                print("--- 内部流式接收开始 ---")

            for chunk in response_stream:
                chunk_text = ""
                if chunk.parts:
                    chunk_text = chunk.text # 尝试获取 .text 属性
                elif hasattr(chunk, 'text') and chunk.text is not None : # 兼容某些仅有 .text 的情况
                    chunk_text = chunk.text

                if chunk_text:
                    full_response_parts.append(chunk_text)
                    if stream_internally_log: # 如果需要，可以在内部打印每个块
                        print(chunk_text, end="", flush=True)
            
            if stream_internally_log:
                print("\n--- 内部流式接收结束 ---")

        except Exception as e:
            print(f"模型 {self.model_name} 在内部流式生成内容时出错: {e}")
            # 根据错误处理策略，您可以选择抛出错误，或者返回部分已接收的内容或错误信息
            # 例如，如果希望即使出错也返回已收集的部分：
            # if full_response_parts:
            #     return "".join(full_response_parts) + f"\n[错误: {e}]"
            raise # 重新抛出异常，让调用者知道发生了错误

        return "".join(full_response_parts) # 返回拼接后的完整字符串


class Deepseek(LLM):
    def __init__(self, api_key: str, reason: bool = False):
        self.api_key = api_key
        self.system_prompt = ""
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.reason = reason

    def generate(self, prompt: str, stream_internally_log: bool = False) -> str:
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
