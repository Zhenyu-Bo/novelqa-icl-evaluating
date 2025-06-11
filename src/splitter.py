import os
import re
import json
import logging
import tiktoken
from typing import List, Dict, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from .llm import LLM, get_llm # Assuming LLM and get_llm are in .llm
from .chapterizer import Chapterizer, LLMSplitter # Assuming Chapterizer is in .chapterizer
from .utils import save_json # Assuming save_json is in .utils

class HybridSplitter:
    """
    结合 Chapterizer 和 LLMSplitter 的功能，对书本内容进行分块。
    1. 使用 Chapterizer 基于正则表达式初步切分。
    2. 对超出 token 限制的块，使用 LLMSplitter 进行二次切分：
        a. 优先按章节标题切分 (LLMSplitter.generate_chunks_by_chapters)。
        b. 若失败，则按语义边界切分 (LLMSplitter.generate_chunks_by_boundaries)。
        c. 若仍失败，则使用 RecursiveCharacterTextSplitter 进行切分。
    3. 对 LLMSplitter/RecursiveCharacterTextSplitter 产生的仍然过大的子块，会再次使用 RecursiveCharacterTextSplitter 强制切分。
    """
    def __init__(self,
                 book_content: str,
                 llm: LLM,
                 book_title: str = "Unknown",
                 max_chunk_tokens: int = 50000,
                 llm_splitter_max_llm_tokens: int = 100000,
                 llm_splitter_chunk_overlap: int = 100,
                 llm_splitter_max_retries: int = 5,
                 llm_splitter_retry_delay: float = 1.0,
                 llm_splitter_max_chunk_tokens_for_merge: int = 20000,
                 llm_splitter_min_chunk_tokens_for_merge: int = 50,
                 char_overlap_fallback: int = 50,
                 chars_per_token_estimate: float = 3.5
                ):
        self.book_content = book_content
        self.book_title = book_title
        self.llm = llm
        self.max_chunk_tokens = max_chunk_tokens

        # Parameters for LLMSplitter when used for sub-splitting
        self.llm_splitter_max_llm_tokens = llm_splitter_max_llm_tokens
        self.llm_splitter_chunk_overlap = llm_splitter_chunk_overlap
        self.llm_splitter_max_retries = llm_splitter_max_retries
        self.llm_splitter_retry_delay = llm_splitter_retry_delay
        self.llm_splitter_max_chunk_tokens_for_merge = llm_splitter_max_chunk_tokens_for_merge
        self.llm_splitter_min_chunk_tokens_for_merge = llm_splitter_min_chunk_tokens_for_merge
        
        # Parameters for RecursiveCharacterTextSplitter fallback
        self.char_overlap_fallback = char_overlap_fallback
        self.chars_per_token_estimate = chars_per_token_estimate

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        try:
            self.token_counter = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"Failed to get tiktoken encoding cl100k_base: {e}. Token counting might be less accurate.")
            self.token_counter = None # Fallback if tiktoken is not available or fails

        self.total_tokens = self._count_tokens(book_content)
        self.final_chunks: List[str] = []

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数量。"""
        if not text:
            return 0
        if self.token_counter:
            try:
                return len(self.token_counter.encode(text))
            except Exception as e:
                self.logger.warning(f"Token counting with tiktoken failed: {e}. Estimating using len(text)/{self.chars_per_token_estimate}.")
                return int(len(text) / self.chars_per_token_estimate) if self.chars_per_token_estimate > 0 else len(text) // 4
        else:
            # Fallback if tiktoken is not available
            return int(len(text) / self.chars_per_token_estimate) if self.chars_per_token_estimate > 0 else len(text) // 4

    def split(self) -> List[str]:
        """执行混合分块过程。"""
        self.logger.info(f"Starting hybrid splitting for book: {self.book_title}")
        
        # 1. 使用 Chapterizer 进行初步切分
        initial_chunks_from_chapterizer = []
        try:
            chapterizer = Chapterizer(self.book_content, self.book_title)
            
            markdown = chapterizer.to_markdown()  # 生成章节标题的 Markdown 格式
            self.logger.info(f"Generated chapter structure for {self.book_title}:\n{markdown}")
            chapterizer.structure_from_markdown(markdown)  # 章节化

            chapter_dict, chapter_list = chapterizer.get_chapter_contents(level=0)
            
            # 将章节内容转换为列表
            initial_chunks_from_chapterizer = [chapter_dict[chapter] for chapter in chapter_list if chapter_dict[chapter].strip()]
            
            if not initial_chunks_from_chapterizer and self.book_content and self.book_content.strip():
                self.logger.warning("Chapterizer did not produce any segments from non-empty content. Using full book content as one initial chunk.")
                initial_chunks_from_chapterizer = [self.book_content]
            elif not initial_chunks_from_chapterizer:
                initial_chunks_from_chapterizer = []

            self.logger.info(f"Chapterizer produced {len(initial_chunks_from_chapterizer)} initial segments.")

        except Exception as e_chap:
            self.logger.error(f"Error during Chapterizer processing: {e_chap}. Treating entire book as one chunk.")
            if self.book_content and self.book_content.strip():
                initial_chunks_from_chapterizer = [self.book_content]
            else:
                initial_chunks_from_chapterizer = []

        # 2. 处理初步切分后的块，对过大的块进行二次切分
        processed_chunks_accumulator = []
        for i, chunk_text in enumerate(initial_chunks_from_chapterizer):
            if not chunk_text or not chunk_text.strip(): # 跳过空块
                continue

            num_tokens = self._count_tokens(chunk_text)
            self.logger.debug(f"Processing initial segment {i+1}/{len(initial_chunks_from_chapterizer)}, tokens: {num_tokens}, max_tokens: {self.max_chunk_tokens}")

            if num_tokens <= self.max_chunk_tokens:
                processed_chunks_accumulator.append(chunk_text)
            else:
                self.logger.info(f"Segment {i+1} ({num_tokens} tokens) is larger than max_chunk_tokens ({self.max_chunk_tokens}). Re-splitting.")
                
                # 为这个过大的块创建一个 LLMSplitter 实例
                # 注意：LLMSplitter 的 book_title 参数在这里用处不大，因为我们处理的是子块
                sub_splitter = LLMSplitter(
                    llm=self.llm,
                    book_content=chunk_text, # 当前过大的块
                    chunk_tokens=self.max_chunk_tokens, # LLM 切分时的分块最大 token 数
                    max_llm_tokens=self.llm_splitter_max_llm_tokens,
                    chunk_overlap=self.llm_splitter_chunk_overlap,
                    max_retries=self.llm_splitter_max_retries,
                    retry_delay=self.llm_splitter_retry_delay,
                    min_chunk_tokens_for_merge=self.llm_splitter_min_chunk_tokens_for_merge,
                    max_chunk_tokens_for_merge=self.llm_splitter_max_chunk_tokens_for_merge # LLMSplitter 内部合并小块时的最大 token 数
                )
                
                sub_split_chunks: Optional[List[str]] = None
                # 2a. 尝试使用 LLM 按章节标题切分
                try:
                    self.logger.debug(f"Attempting LLM-based chapter splitting for oversized segment {i+1}.")
                    sub_splitter.split_recursive(chunk_text, by_chapter=True) # 使用 LLM 切分章节
                    sub_split_chunks = sub_splitter.chunks
                    if not sub_split_chunks or not any(s.strip() for s in sub_split_chunks):
                        self.logger.warning(f"LLM chapter splitting yielded no valid sub-chunks for segment {i+1}.")
                        sub_split_chunks = None 
                    else:
                         self.logger.info(f"LLM chapter splitting for segment {i+1} yielded {len(sub_split_chunks)} sub-chunks.")
                except Exception as e1:
                    self.logger.warning(f"LLM chapter splitting failed for segment {i+1}: {e1}. Trying semantic boundaries.")
                    sub_split_chunks = None

                # 2b. 如果按章节标题切分失败，尝试按语义边界切分
                if sub_split_chunks is None:
                    try:
                        self.logger.debug(f"Attempting LLM-based boundary splitting for oversized segment {i+1}.")
                        sub_splitter.split_recursive(chunk_text, by_chapter=False) # 使用 LLM 按语义边界切分
                        sub_split_chunks = sub_splitter.chunks
                        if not sub_split_chunks or not any(s.strip() for s in sub_split_chunks):
                            self.logger.warning(f"LLM boundary splitting yielded no valid sub-chunks for segment {i+1}.")
                            sub_split_chunks = None 
                        else:
                            self.logger.info(f"LLM boundary splitting for segment {i+1} yielded {len(sub_split_chunks)} sub-chunks.")
                    except Exception as e1:
                        self.logger.warning(f"LLM boundary splitting failed for segment {i+1}: {e1}. Trying semantic boundaries.")
                        sub_split_chunks = None
                
                # 2c. 如果 LLM 辅助切分均失败，使用 RecursiveCharacterTextSplitter
                if sub_split_chunks is None:
                    self.logger.info(f"Falling back to RecursiveCharacterTextSplitter for oversized segment {i+1}.")
                    # char_chunk_size = int(self.max_chunk_tokens * self.chars_per_token_estimate)
                    # char_chunk_size = max(100, char_chunk_size) # 确保 chunk_size 为正且不太小
                    
                    r_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.max_chunk_tokens,
                        chunk_overlap=self.char_overlap_fallback,
                        length_function=self._count_tokens,
                        separators=["\n\n", "\n", ". ", " ", ""] # 标准分隔符
                    )
                    sub_split_chunks = r_splitter.split_text(chunk_text)
                    self.logger.info(f"RecursiveCharacterTextSplitter for segment {i+1} yielded {len(sub_split_chunks)} sub-chunks.")

                # 将子块添加到处理后的块累积器中
                final_sub_chunks_for_this_segment = []
                if sub_split_chunks:
                    for k, sub_chunk_text in enumerate(sub_split_chunks):
                        if not sub_chunk_text or not sub_chunk_text.strip():
                            continue
                        final_sub_chunks_for_this_segment.append(sub_chunk_text)
                processed_chunks_accumulator.extend(final_sub_chunks_for_this_segment)
        
        self.final_chunks = [chunk for chunk in processed_chunks_accumulator if chunk.strip()] # 清理空字符串
        self.logger.info(f"Hybrid splitting complete. Total final chunks: {len(self.final_chunks)}")
        return self.final_chunks

    def get_chunks(self) -> List[str]:
        """获取最终的分块列表。"""
        if not self.final_chunks:
            self.logger.warning("No chunks generated yet. Call split() method first.")
        return self.final_chunks

    def save_chunks_to_json(self, output_path: str) -> None:
        """将最终的分块列表保存为 JSON 文件。"""
        if not self.final_chunks:
            self.logger.error("No chunks to save. Run split() method first.")
            # Consider raising an error or just returning if no chunks
            # raise ValueError("No chunks to save. Run split() method first.")
            print("No chunks to save. Run split() method first.")
            return
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.logger.info(f"Created directory: {output_dir}")
            except OSError as e:
                self.logger.error(f"Failed to create directory {output_dir}: {e}")
                # Depending on desired behavior, either raise or try to save in current dir
                # For now, we'll let the open() call fail if dir creation fails.

        try:
            # Using the provided save_json utility if it handles path creation and logging
            # Otherwise, use direct json.dump
            if callable(globals().get('save_json')):
                 save_json(self.final_chunks, output_path) # Assumes save_json handles ensure_ascii and indent
                 self.logger.info(f"Successfully saved {len(self.final_chunks)} chunks to {output_path} using save_json.")
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.final_chunks, f, ensure_ascii=False, indent=4)
                self.logger.info(f"Successfully saved {len(self.final_chunks)} chunks to {output_path} using json.dump.")
        except IOError as e:
            self.logger.error(f"Failed to save chunks to {output_path}: {e}")
            raise # Re-raise the IOError
        except Exception as e: # Catch other potential errors from save_json
            self.logger.error(f"An unexpected error occurred while saving chunks to {output_path}: {e}")
            raise
        
        metadata = {
            "book_title": self.book_title,
            "total_tokens": self.total_tokens,
            "total_chunks": len(self.final_chunks),
            "max_chunk_tokens": self.max_chunk_tokens,
            "llm_splitter_max_llm_tokens": self.llm_splitter_max_llm_tokens,
            "llm_splitter_chunk_overlap": self.llm_splitter_chunk_overlap,
            "llm_splitter_max_retries": self.llm_splitter_max_retries,
            "llm_splitter_retry_delay": self.llm_splitter_retry_delay,
            "llm_splitter_max_chunk_tokens_for_merge": self.llm_splitter_max_chunk_tokens_for_merge,
            "llm_splitter_min_chunk_tokens_for_merge": self.llm_splitter_min_chunk_tokens_for_merge,
            "char_overlap_fallback": self.char_overlap_fallback,
            "chars_per_token_estimate": self.chars_per_token_estimate
        }
        return metadata
