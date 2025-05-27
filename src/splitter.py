import re
import json
import logging
import tiktoken
from typing import List, Dict, Optional, Set 
from tqdm import tqdm 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .llm import LLM 

class LLMSplitter():
    """使用 LLM 将书本内容按语义切分为 chunks """

    DEFAULT_MAX_LLM_TOKENS = 600000 
    DEFAULT_CHUNK_TOKENS = DEFAULT_MAX_LLM_TOKENS // 10 
    DEFAULT_CHUNK_OVERLAP = 2000
    DEFAULT_ENCODING_NAME = "cl100k_base"
    LLM_PROCESSING_SAFETY_MARGIN = 2000 
    MAX_RESPLIT_ITERATIONS = 3
    OVERSIZED_CHUNK_FACTOR = 2 # Factor for strict re-split
    MIN_CHUNK_TOKENS_FOR_MERGE = DEFAULT_CHUNK_TOKENS // 20 # Min tokens for a chunk to avoid merging

    def __init__(self, 
                 llm: LLM, 
                 book_content: str, 
                 book_title: Optional[str] = None,
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS, 
                 max_llm_tokens: int = DEFAULT_MAX_LLM_TOKENS, 
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 encoding_name: str = DEFAULT_ENCODING_NAME,
                 min_chunk_tokens_for_merge: int = MIN_CHUNK_TOKENS_FOR_MERGE):
        self.llm = llm
        self.book_title = book_title
        self.original_book_content = book_content 
        # This is the full content, normalized, used for final splitting by boundaries
        self.processed_book_content = re.sub(r'\s+', ' ', book_content).strip()
        
        self.chunk_tokens = chunk_tokens
        self.max_llm_tokens = max_llm_tokens
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens_for_merge = min_chunk_tokens_for_merge
        self.chunks: List[str] = []
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers: 
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        try:
            self.token_counter = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            self.logger.warning(f"Failed to load encoding '{encoding_name}', using default '{self.DEFAULT_ENCODING_NAME}'. Error: {e}")
            self.token_counter = tiktoken.get_encoding(self.DEFAULT_ENCODING_NAME)
            
        self.tokens_num_original = self._count_tokens(self.original_book_content)
        # Token count of the full processed content
        self.tokens_num_full_processed = self._count_tokens(self.processed_book_content)

        self.is_long_document: bool = self.tokens_num_full_processed > self.max_llm_tokens
        self.initial_doc_chunks_for_llm: Optional[List[str]] = None

        if self.is_long_document:
            self.logger.info(
                f"Book content tokens ({self.tokens_num_full_processed}) exceed max LLM tokens per call ({self.max_llm_tokens}). "
                f"Pre-splitting original content for LLM processing."
            )
            text_splitter = self._get_splitter_for_long_docs()
            # Split original content to better preserve potential paragraph structure for splitter
            raw_chunks = text_splitter.split_text(self.original_book_content) 
            self.initial_doc_chunks_for_llm = [
                re.sub(r'\s+', ' ', chunk).strip() for chunk in raw_chunks if chunk.strip()
            ]
            self.logger.info(f"Pre-split original content into {len(self.initial_doc_chunks_for_llm)} initial parts for LLM processing.")
            if not self.initial_doc_chunks_for_llm:
                 self.logger.warning("Pre-splitting resulted in no initial_doc_chunks_for_llm. This might indicate an issue or very short content despite being 'long'.")
        else:
            self.logger.info(
                f"Book content tokens ({self.tokens_num_full_processed}) are within max LLM tokens per call ({self.max_llm_tokens}). "
                f"No pre-splitting needed."
            )

        self.logger.info(f"LLMSplitter initialized for '{self.book_title if self.book_title else 'Untitled Book'}':")
        self.logger.info(f"  Target Semantic Chunk Tokens: {self.chunk_tokens}")
        self.logger.info(f"  Strict Re-split Threshold: {self.chunk_tokens * self.OVERSIZED_CHUNK_FACTOR:.0f} tokens")
        self.logger.info(f"  Min Chunk Tokens for Merging: {self.min_chunk_tokens_for_merge} tokens")
        self.logger.info(f"  Max LLM Call Tokens (for sub-problems): {self.max_llm_tokens}")
        self.logger.info(f"  Initial Split Overlap (tokens): {self.chunk_overlap}")
        self.logger.info(f"  Encoding: {encoding_name}")
        self.logger.info(f"  Original Book Tokens: {self.tokens_num_original}")
        self.logger.info(f"  Full Processed Book Tokens (for final splitting): {self.tokens_num_full_processed}")
        self.logger.info(f"  Is Long Document (requires pre-splitting for LLM calls): {self.is_long_document}")


    def _count_tokens(self, text: str) -> int:
        """Accurately count tokens in a text string."""
        if not text:
            return 0
        try:
            return len(self.token_counter.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed, estimating using len(text)//4. Error: {e}")
            return len(text) // 4 

    def set_prompt_directly(self, content: str) -> str:
        """设置让 LLM 直接输出分块内容的提示词"""
        prompt = f"""
        You are a professional text processing assistant. Please divide the following text into multiple chunks based on semantic coherence.

        Requirements:
        1. Each chunk should be semantically coherent, representing a complete part of the story or content.
        2. The token count of each chunk **MUST STRICTLY NOT EXCEED {self.chunk_tokens} tokens**. This is a **CRITICAL REQUIREMENT**. Adherence to this token limit is paramount.
        3. Prefer splitting at chapter boundaries or paragraph boundaries if possible, while respecting the token limit.
        4. Ensure each chunk is semantically complete and does not split in the middle of a sentence, unless absolutely necessary to meet the token limit.
        5. The length of each chunk should be relatively balanced, but semantic coherence and the strict token limit are most important.
        
        Please return the chunked results in the following JSON format:
        {{
            "chunks": [
                "Content of the first chunk (strictly <= {self.chunk_tokens} tokens)",
                "Content of the second chunk (strictly <= {self.chunk_tokens} tokens)",
                ...
            ]
        }}

        Here is the text to be divided:

        {content}
        """
        return prompt
    
    def set_prompt_boundaries(self, content: str) -> str:
        """设置让 LLM 输出分块边界的提示词"""
        prompt = f"""
        You are a professional text processing assistant. Your task is to identify semantic chunk boundaries in the following text.
        The provided text has ALREADY BEEN PROCESSED: all newline characters and other forms of whitespace have been converted into single spaces, and any leading/trailing whitespace has been removed.

        Requirements:
        1. Each chunk created by these boundaries should be semantically coherent.
        2. The token count of each chunk resulting from these boundaries **MUST STRICTLY NOT EXCEED {self.chunk_tokens} tokens**. This is a **HARD LIMIT**. Semantic coherence is important, but adherence to this token limit for the resulting chunks is paramount for subsequent processing steps.
        3. Prefer marking chunk boundaries at natural breaks in the text (e.g., end of paragraphs, topic shifts) as they appear in the processed text, as long as the resulting chunks respect the token limit.
        4. Chunk boundaries MUST be EXACT VERBATIM SUBSTRINGS of the provided 'Text to be divided' below.
        5. The length of each chunk should be relatively balanced, but semantic coherence and the strict token limit are most important.

        CRITICALLY IMPORTANT - Adhere to these rules for boundary text:
        - The boundary text you return MUST be an EXACT character-for-character match from the 'Text to be divided'.
        - Do NOT add, remove, or change ANY characters, including punctuation or spacing.
        - Do NOT attempt to reconstruct or infer original formatting (like newlines) that is NOT present in the input text.

        Please return only the chunk boundaries (the ending sentence or phrase of each chunk) in the following JSON format:
        {{
            "boundaries": [
                "Exact verbatim ending phrase of the first chunk (ensure chunk <= {self.chunk_tokens} tokens), copied precisely from the provided text",
                "Exact verbatim ending phrase of the second chunk (ensure chunk <= {self.chunk_tokens} tokens), copied precisely from the provided text",
                ...
            ]
        }}

        Here is the text to be divided:

        {content}
        
        **If you cannot identify any suitable boundaries that meet all requirements (especially the token limit for resulting chunks), return an empty JSON array: `{{ "boundaries": [] }}`**
        """
        return prompt

    def set_prompt_chapter_markers(self, content: str) -> str:
        """Sets the prompt for the LLM to identify chapter-ending markers."""
        prompt = f"""
        You are a professional text processing assistant. Your task is to identify chapter markers in the following text.
        A chapter marker is a short, distinctive phrase or sentence that occurs *at the very end of a chapter's content*, just before a new chapter would typically begin, OR a chapter title itself if it clearly delineates sections.
        The provided text has ALREADY BEEN PROCESSED: all newline characters and other forms of whitespace have been converted into single spaces, and any leading/trailing whitespace has been removed.

        Requirements:
        1. Identify phrases or chapter titles that serve as clear separators between chapters.
        2. These markers MUST be EXACT VERBATIM SUBSTRINGS of the provided 'Text to be analyzed' below.
        3. The goal is to use these markers to split the text into chapter-like segments.
        4. If a chapter title is used as a marker, it will signify the end of the *previous* segment and the start of a new one.
        5. The chapter markers you identify should not be too long or too short (e.g., not just a single word or number). Ideally, a chapter marker should be a complete sentence from the original text OR a chapter title.

        CRITICALLY IMPORTANT - Adhere to these rules for marker text:
        - The marker text you return MUST be an EXACT character-for-character match from the 'Text to be analyzed'.
        - Do NOT add, remove, or change ANY characters, including punctuation or spacing.
        - Do NOT attempt to reconstruct original formatting (like newlines) that is NOT present in the input text.

        Please return the identified chapter markers in the following JSON format:
        {{
            "chapter_markers": [
                "Exact verbatim marker for end of chapter 1 / start of chapter 2, copied precisely from the provided text",
                "Exact verbatim marker for end of chapter 2 / start of chapter 3, copied precisely from the provided text",
                ...
            ]
        }}

        Text to be analyzed:

        {content}
        
        **If you cannot identify any suitable chapter markers, return an empty JSON array: `{{ "chapter_markers": [] }}`**
        """
        return prompt

    def _parse_llm_response(self, response: str, expected_key: str) -> Optional[List[str]]:
        """Helper to parse JSON response from LLM and extract a list from a key."""
        if not response or not response.strip():
            self.logger.warning(f"LLM returned empty or whitespace-only response.")
            return None
        try:
            match = re.search(r'\{[\s\S]*?\}', response)
            if not match:
                match_array = re.search(r'\[[\s\S]*?\]', response)
                if match_array and expected_key == "chunks": 
                    self.logger.warning(f"Found JSON array but expected a JSON object with key '{expected_key}'. Response: {response[:200]}...")
                    return None 
                elif match_array and expected_key != "chunks": 
                    self.logger.warning(f"Found JSON array but expected a JSON object with key '{expected_key}'. Response: {response[:200]}...")
                    return None

                self.logger.warning(f"No valid JSON object found in LLM response. Response: {response[:200]}...")
                return None
            
            json_str = match.group(0)
            result = json.loads(json_str)
            
            if not isinstance(result, dict) or expected_key not in result:
                self.logger.warning(f"LLM response JSON format incorrect, '{expected_key}' field missing. Parsed: {result}")
                return None
            
            data_list = result[expected_key]
            if not isinstance(data_list, list):
                self.logger.warning(f"Field '{expected_key}' in LLM response is not a list. Found: {type(data_list)}")
                return None
            
            return [str(item) for item in data_list] # Ensure all items are strings
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response JSON. Error: {e}. Response: {response[:200]}...")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing LLM response: {e}. Response: {response[:200]}...")
            return None

    def _re_split_oversized_chunk(self, chunk_text: str, iteration: int = 1) -> List[str]:
        """
        Recursively re-splits a chunk if it's oversized, first by LLM, then by RecursiveCharacterTextSplitter as a fallback.
        """
        chunk_tokens = self._count_tokens(chunk_text)
        # When re-splitting, the target is self.chunk_tokens.
        # The strict_limit_tokens is the threshold that *triggers* this re-split function.
        # If a sub-chunk after splitting is still > self.chunk_tokens but <= strict_limit_tokens,
        # we might keep it if it's the result of a split, to avoid too granular splits unless necessary.
        # However, the primary goal is to get chunks <= self.chunk_tokens.
        # The OVERSIZED_CHUNK_FACTOR is used in _post_process_and_refine_chunks to decide *when* to call this.
        # Here, we are trying to get *below* self.chunk_tokens.

        # If the chunk is already small enough (or just slightly over but within a reasonable margin after a split attempt)
        # and this isn't the first call (meaning it's a result of a previous split attempt).
        if chunk_tokens <= self.chunk_tokens * self.OVERSIZED_CHUNK_FACTOR and iteration > 1:
            if chunk_tokens > self.chunk_tokens:
                 self.logger.info(f"Chunk of {chunk_tokens} tokens is above target {self.chunk_tokens} but below strict re-split factor {self.chunk_tokens * self.OVERSIZED_CHUNK_FACTOR:.0f} after iteration {iteration-1}. Keeping as is for now.")
            return [chunk_text]
        
        if iteration > self.MAX_RESPLIT_ITERATIONS:
            self.logger.warning(f"Max re-split iterations ({self.MAX_RESPLIT_ITERATIONS}) reached for a chunk of {chunk_tokens} tokens. Force splitting with RecursiveCharacterTextSplitter to target {self.chunk_tokens} tokens.")
            force_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_tokens, # Target the desired chunk size
                chunk_overlap=max(50, self.chunk_overlap // 4), 
                length_function=self._count_tokens,
                separators=["\n\n", "\n", ". ", " ", ""], # Use sensible separators
                keep_separator=False
            )
            return force_splitter.split_text(chunk_text)

        self.logger.info(f"Attempting to re-split an oversized chunk of {chunk_tokens} tokens (Iteration {iteration}). Target: <= {self.chunk_tokens}.")

        max_content_for_llm = self.max_llm_tokens - self.LLM_PROCESSING_SAFETY_MARGIN
        sub_chunks: List[str] = []
        llm_boundaries_found = False

        current_text_to_split_for_llm = chunk_text # The text we are trying to find boundaries in

        if self._count_tokens(current_text_to_split_for_llm) > max_content_for_llm:
            self.logger.info(f"Oversized chunk for re-splitting ({self._count_tokens(current_text_to_split_for_llm)} tokens) is too large for a single LLM call (max content: {max_content_for_llm} tokens). Using RecursiveCharacterTextSplitter to break it down for LLM boundary finding.")
            # This part is tricky: we are splitting a chunk that was *already* deemed oversized.
            # The LLM should find boundaries *within* this chunk.
            # If we pre-split it further here just for LLM calls, we need to ensure boundaries are applied correctly.
            # For simplicity in re-splitting, if it's too big for LLM, we might have to rely more on RecursiveCharacterTextSplitter.
            # Or, we can try to get boundaries from the first manageable part.
            # Let's try to use LLM on the first part if the whole chunk is too big.
            
            # Fallback: if even the chunk to be re-split is too big for LLM, go straight to text splitter for this chunk
            self.logger.warning(f"Chunk to re-split is too large for LLM boundary identification ({self._count_tokens(current_text_to_split_for_llm)} > {max_content_for_llm}). Using RecursiveCharacterTextSplitter directly for this chunk.")
            # No LLM boundaries will be found in this path
        else:
            self.logger.info(f"Oversized chunk for re-splitting ({self._count_tokens(current_text_to_split_for_llm)} tokens) is manageable for a single LLM call to find boundaries.")
            prompt = self.set_prompt_boundaries(current_text_to_split_for_llm) # Ask for boundaries within this specific chunk
            response = self.llm.generate(prompt)
            internal_boundaries = self._parse_llm_response(response, "boundaries")

            if internal_boundaries:
                self.logger.info(f"LLM found {len(internal_boundaries)} internal boundaries in the oversized chunk being re-split.")
                # Apply these boundaries to the current_text_to_split_for_llm
                sub_chunks = self._split_by_boundaries(current_text_to_split_for_llm, internal_boundaries)
                llm_boundaries_found = True
                if not sub_chunks and current_text_to_split_for_llm: # LLM gave boundaries but split resulted in nothing (e.g. boundary was whole text)
                    self.logger.warning("LLM boundaries for re-split resulted in no effective split. Will fall back to text splitter.")
                    llm_boundaries_found = False # Treat as no boundaries found
                    sub_chunks = [] # Clear sub_chunks
            else:
                self.logger.warning(f"LLM found no internal boundaries in the oversized chunk being re-split.")
        
        # If LLM failed or produced no usable split, use RecursiveCharacterTextSplitter
        if not llm_boundaries_found or not sub_chunks:
            self.logger.warning(f"LLM-based re-splitting failed or yielded no sub-chunks for the oversized chunk. Force splitting with RecursiveCharacterTextSplitter to target {self.chunk_tokens} tokens.")
            force_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_tokens, # Target the desired chunk size
                chunk_overlap=max(50, self.chunk_overlap // 4), 
                length_function=self._count_tokens,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=False
            )
            sub_chunks = force_splitter.split_text(chunk_text) # Split the original chunk_text passed to this function
            self.logger.info(f"Force-split oversized chunk into {len(sub_chunks)} sub-chunks using RecursiveCharacterTextSplitter.")
        
        final_refined_sub_chunks: List[str] = []
        for sc_idx, sc_text in enumerate(sub_chunks):
            sc_tokens = self._count_tokens(sc_text)
            if sc_tokens == 0: 
                continue
            # If a sub-chunk is still significantly oversized (e.g. > 1.5x target), recurse
            # Or if it's simply larger than the target, recurse. The goal is to get under self.chunk_tokens.
            if sc_tokens > self.chunk_tokens: # If any sub-chunk is still over the target
                self.logger.info(f"A sub-chunk ({sc_idx+1}/{len(sub_chunks)}) from re-splitting is still oversized ({sc_tokens} tokens > target {self.chunk_tokens}). Re-splitting it (Overall Iteration {iteration}, Next Level Iteration {iteration + 1}).")
                final_refined_sub_chunks.extend(self._re_split_oversized_chunk(sc_text, iteration + 1))
            else:
                final_refined_sub_chunks.append(sc_text)
        
        return final_refined_sub_chunks

    def _post_process_and_refine_chunks(self):
        self.logger.info("Starting post-processing to refine chunk sizes.")
        refined_chunks: List[str] = []
        initial_chunk_count = len(self.chunks)
        oversized_chunks_found = 0
        # This is the threshold that triggers a call to _re_split_oversized_chunk
        strict_token_limit_for_resplit_trigger = self.chunk_tokens * self.OVERSIZED_CHUNK_FACTOR

        for i, chunk_text in enumerate(tqdm(self.chunks, desc="Refining chunk sizes", unit="chunk")):
            chunk_token_count = self._count_tokens(chunk_text)

            if chunk_token_count > strict_token_limit_for_resplit_trigger:
                oversized_chunks_found += 1
                self.logger.warning(
                    f"Chunk {i+1}/{initial_chunk_count} (original index) has {chunk_token_count} tokens, "
                    f"exceeding strict re-split trigger limit of {strict_token_limit_for_resplit_trigger:.0f} (target: {self.chunk_tokens}). Attempting re-split."
                )
                newly_split_sub_chunks = self._re_split_oversized_chunk(chunk_text, iteration=1)
                refined_chunks.extend(newly_split_sub_chunks)
                self.logger.info(f"Original oversized chunk {i+1} was re-split into {len(newly_split_sub_chunks)} sub-chunks.")
            else:
                if chunk_token_count > self.chunk_tokens: # Over target, but not over strict trigger
                     self.logger.info(f"Chunk {i+1}/{initial_chunk_count} has {chunk_token_count} tokens (target {self.chunk_tokens}), but under strict re-split trigger {strict_token_limit_for_resplit_trigger:.0f}. Keeping as is for now.")
                refined_chunks.append(chunk_text)
        
        if oversized_chunks_found > 0:
            self.logger.info(f"Post-processing refinement complete. {oversized_chunks_found} oversized chunks were re-processed.")
            self.logger.info(f"Chunk count may have changed from {initial_chunk_count} to {len(refined_chunks)}.")
        else:
            self.logger.info("Post-processing refinement complete. No chunks exceeded the strict token limit for triggering re-splitting.")
        
        self.chunks = [c for c in refined_chunks if self._count_tokens(c) > 0] 
        if len(self.chunks) != len(refined_chunks):
            self.logger.info(f"Removed {len(refined_chunks) - len(self.chunks)} empty chunks after refinement.")

    def _merge_small_consecutive_chunks(self):
        if not self.chunks or len(self.chunks) < 2:
            self.logger.info("Not enough chunks to perform merging or no chunks present.")
            return

        self.logger.info(f"Starting post-merging of small chunks. Min tokens for a chunk to be considered small: < {self.min_chunk_tokens_for_merge}. Max combined tokens for merge: <= {self.chunk_tokens}")
        
        merged_chunks: List[str] = []
        i = 0
        num_merges = 0
        original_chunk_count = len(self.chunks)

        while i < len(self.chunks):
            # 取当前块作为起始点
            current_chunk = self.chunks[i]
            current_tokens = self._count_tokens(current_chunk)
            
            # 如果当前块足够大，直接添加并前进
            if current_tokens >= self.min_chunk_tokens_for_merge:
                merged_chunks.append(current_chunk)
                i += 1
                continue
            
            # 尝试与后续多个块合并
            combined_chunk = current_chunk
            combined_tokens = current_tokens
            next_idx = i + 1
            chunks_merged = 0  # 当前轮次合并的块数
            
            # 继续尝试与后续块合并，直到达到token上限或没有更多块
            while next_idx < len(self.chunks):
                next_chunk = self.chunks[next_idx]
                next_tokens = self._count_tokens(next_chunk)
                
                # 检查合并后是否超过上限
                if (combined_tokens + next_tokens) <= self.chunk_tokens:
                    # 合并块
                    combined_chunk = (combined_chunk + " " + next_chunk).strip()
                    combined_tokens += next_tokens  # 这是一个近似值，实际token数可能因空格而略有不同
                    chunks_merged += 1
                    next_idx += 1
                else:
                    # 如果合并下一个会超过限制，停止合并
                    break
            
            if chunks_merged > 0:
                # 实际计算最终合并块的token数，因为简单相加可能不精确
                actual_combined_tokens = self._count_tokens(combined_chunk)
                self.logger.info(f"Merged {chunks_merged+1} chunks starting at position {i+1} into a single chunk of {actual_combined_tokens} tokens.")
                merged_chunks.append(combined_chunk)
                num_merges += chunks_merged
                i = next_idx  # 跳过所有已合并的块
            else:
                # 无法合并，保留当前块
                self.logger.debug(f"Chunk {i+1} ({current_tokens} tk) is small but cannot be merged with any following chunks without exceeding limit.")
                merged_chunks.append(current_chunk)
                i += 1
        
        if num_merges > 0:
            self.logger.info(f"Performed {num_merges} merges of small chunks. Chunk count changed from {original_chunk_count} to {len(merged_chunks)}.")
            self.chunks = merged_chunks
        else:
            self.logger.info("No small consecutive chunks were merged.")


    def generate_chunks_directly(self) -> list[str]:
        method_name = "Direct LLM Chunking"
        self.logger.info(f"Starting: {method_name}")
        
        if self.is_long_document:
            self.chunks = self._process_long_document_directly()
        else:
            prompt = self.set_prompt_directly(self.processed_book_content)
            response = self.llm.generate(prompt)
            parsed_chunks = self._parse_llm_response(response, "chunks")
            if parsed_chunks is None: 
                self.logger.error(f"{method_name} failed to get valid chunks from LLM for the whole document.")
                self.chunks = [self.processed_book_content] if self.processed_book_content.strip() else [] # Fallback
            else:
                self.chunks = parsed_chunks
        
        self._post_process_and_refine_chunks() 
        self._merge_small_consecutive_chunks()
        self._log_chunk_statistics(method_name)
        if not self.chunks:
             self.logger.warning(f"{method_name} resulted in zero chunks.")
        return self.chunks
    
    def generate_chunks_by_boundaries(self) -> list[str]:
        method_name = "Boundary-based LLM Chunking"
        self.logger.info(f"Starting: {method_name}")
        
        if self.is_long_document:
            self.chunks = self._process_long_document_by_boundaries()
        else:
            prompt = self.set_prompt_boundaries(self.processed_book_content)
            max_retries = 3
            boundaries = None
            for attempt in range(max_retries):
                self.logger.info(f"Attempt {attempt + 1}/{max_retries} to get boundaries from LLM for the whole document.")
                response = self.llm.generate(prompt)
                boundaries = self._parse_llm_response(response, "boundaries")
                if boundaries is not None: 
                    break
                self.logger.warning(f"Attempt {attempt + 1} failed to get valid boundaries.")
            
            if boundaries is None:
                self.logger.error(f"{method_name} failed to get boundaries after {max_retries} retries. Treating as single chunk.")
                self.chunks = [self.processed_book_content] if self.processed_book_content.strip() else []
            elif not boundaries:
                self.logger.info("LLM returned no boundaries for the whole document. Treating entire content as a single chunk.")
                self.chunks = [self.processed_book_content] if self.processed_book_content.strip() else []
            else:
                self.chunks = self._split_by_boundaries(self.processed_book_content, boundaries)
        
        self._post_process_and_refine_chunks() 
        self._merge_small_consecutive_chunks()
        self._log_chunk_statistics(method_name)
        if not self.chunks:
             self.logger.warning(f"{method_name} resulted in zero chunks.")
        return self.chunks

    def generate_chunks_by_chapter_markers(self) -> list[str]:
        method_name = "Chapter-Marker-based LLM Chunking"
        self.logger.info(f"Starting: {method_name}")

        if self.is_long_document:
            self.chunks = self._process_long_document_by_chapter_markers()
        else:
            prompt = self.set_prompt_chapter_markers(self.processed_book_content)
            max_retries = 3
            chapter_markers = None
            for attempt in range(max_retries):
                self.logger.info(f"Attempt {attempt + 1}/{max_retries} to get chapter markers from LLM for the whole document.")
                response = self.llm.generate(prompt)
                chapter_markers = self._parse_llm_response(response, "chapter_markers")
                if chapter_markers is not None: 
                    break
                self.logger.warning(f"Attempt {attempt + 1} failed to get valid chapter markers.")
            
            if chapter_markers is None:
                self.logger.error(f"{method_name} failed to get chapter markers after {max_retries} retries. Treating as single chunk.")
                self.chunks = [self.processed_book_content] if self.processed_book_content.strip() else []
            elif not chapter_markers:
                self.logger.info("LLM returned no chapter markers for the whole document. Treating entire content as a single chunk.")
                self.chunks = [self.processed_book_content] if self.processed_book_content.strip() else []
            else:
                self.chunks = self._split_by_boundaries(self.processed_book_content, chapter_markers)
        
        self._post_process_and_refine_chunks() 
        self._merge_small_consecutive_chunks()
        self._log_chunk_statistics(method_name)
        if not self.chunks:
             self.logger.warning(f"{method_name} resulted in zero chunks.")
        return self.chunks
    
    def _split_by_boundaries(self, text_to_split: str, boundaries: list[str]) -> list[str]:
        """根据边界在原文中切分"""
        chunks: List[str] = []
        remaining_text = text_to_split
        self.logger.info(f"Attempting to split text of {self._count_tokens(text_to_split)} tokens using {len(boundaries)} boundaries/markers.")

        processed_boundaries = []
        for raw_boundary in boundaries:
            # Normalize boundary from LLM, same way original text was processed
            # Ensure boundary is a string
            boundary_str = str(raw_boundary) if raw_boundary is not None else ""
            boundary = re.sub(r'\s+', ' ', boundary_str).strip()
            if boundary:
                processed_boundaries.append(boundary)
            else:
                self.logger.warning(f"Skipping empty or invalid boundary/marker: '{raw_boundary}'")
        
        if not processed_boundaries:
            self.logger.warning("No valid boundaries/markers provided for splitting. Returning original text as single chunk.")
            return [text_to_split] if text_to_split.strip() else []

        for i, boundary in enumerate(processed_boundaries):
            self.logger.debug(f"Processing boundary/marker {i+1}/{len(processed_boundaries)}: '{boundary[:100]}...'")
            try:
                split_idx = remaining_text.index(boundary) 
                split_pos = split_idx + len(boundary) 
                
                chunk = remaining_text[:split_pos].strip()
                if chunk: 
                    chunks.append(chunk)
                remaining_text = remaining_text[split_pos:].strip() 
                self.logger.debug(f"Successfully split at boundary/marker {i+1}. New remaining text starts with: '{remaining_text[:100]}...'")
                if not remaining_text: # Stop if no text left
                    break
            except ValueError:
                self.logger.error(f"Boundary/marker '{boundary[:100]}...' NOT FOUND in remaining text. Skipping this one.")
                self.logger.debug(f"Remaining text sample (first 300 chars): '{remaining_text[:300]}'")
                continue 
        
        if remaining_text: 
            self.logger.debug("Adding final remaining text as the last chunk.")
            chunks.append(remaining_text)
        
        self.logger.info(f"Text split into {len(chunks)} chunks using boundaries/markers.")
        return chunks
    
    def _get_splitter_for_long_docs(self) -> RecursiveCharacterTextSplitter:
        """Configures RecursiveCharacterTextSplitter for initial long document breakdown."""
        initial_chunk_size_tokens = self.max_llm_tokens - self.LLM_PROCESSING_SAFETY_MARGIN
        if initial_chunk_size_tokens <= self.chunk_overlap: # Ensure chunk size is greater than overlap
            initial_chunk_size_tokens = self.max_llm_tokens // 2 
            self.logger.warning(
                f"Calculated initial_chunk_size_tokens ({self.max_llm_tokens - self.LLM_PROCESSING_SAFETY_MARGIN}) "
                f"was too small or less than/equal to overlap ({self.chunk_overlap}). "
                f"Using fallback initial_chunk_size_tokens: {initial_chunk_size_tokens}"
            )
        if initial_chunk_size_tokens <=0: # Further fallback
            initial_chunk_size_tokens = 1000 # Absolute minimum
            self.logger.error(f"Max LLM tokens too small. Using absolute minimum {initial_chunk_size_tokens} for initial split.")


        return RecursiveCharacterTextSplitter(
            chunk_size=initial_chunk_size_tokens,
            chunk_overlap=self.chunk_overlap, 
            length_function=self._count_tokens, 
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""], # Added more sentence enders
            keep_separator=True # Keep separators to help maintain context, will be normalized later
        )

    def _process_long_document_directly(self) -> list[str]:
        """处理超长文档 - 直接输出方式 (uses pre-split self.initial_doc_chunks_for_llm)"""
        self.logger.info("Processing long document for direct chunking using pre-split parts...")
        if not self.initial_doc_chunks_for_llm:
            self.logger.error("Long document processing called, but no initial_doc_chunks_for_llm were prepared.")
            return [self.processed_book_content] if self.processed_book_content.strip() else []

        all_final_chunks: List[str] = []
        for i, doc_chunk_content in enumerate(tqdm(self.initial_doc_chunks_for_llm, desc="LLM direct chunking on parts")):
            self.logger.info(f"Processing part {i+1}/{len(self.initial_doc_chunks_for_llm)} with LLM for direct sub-chunking (tokens: {self._count_tokens(doc_chunk_content)}).")
            prompt = self.set_prompt_directly(doc_chunk_content)
            response = self.llm.generate(prompt)
            sub_chunks = self._parse_llm_response(response, "chunks")
            
            if sub_chunks:
                self.logger.info(f"Extracted {len(sub_chunks)} sub-chunks from part {i+1}.")
                all_final_chunks.extend(sub_chunks)
            else:
                self.logger.warning(f"No sub-chunks extracted from part {i+1}. Using the part itself as a chunk.")
                all_final_chunks.append(doc_chunk_content) # Fallback: use the part itself
        
        self.logger.info(f"Long document direct processing resulted in {len(all_final_chunks)} final chunks before refinement.")
        return all_final_chunks
    
    def _process_long_document_by_boundaries(self) -> list[str]:
        """处理超长文档 - 边界切分方式 (uses pre-split self.initial_doc_chunks_for_llm)"""
        self.logger.info("Processing long document for boundary-based chunking using pre-split parts...")
        if not self.initial_doc_chunks_for_llm:
            self.logger.error("Long document processing called, but no initial_doc_chunks_for_llm were prepared.")
            return [self.processed_book_content] if self.processed_book_content.strip() else []
            
        collected_boundaries_raw: List[str] = []
        for i, doc_chunk_content in enumerate(tqdm(self.initial_doc_chunks_for_llm, desc="LLM boundary identification on parts")):
            self.logger.info(f"Processing part {i+1}/{len(self.initial_doc_chunks_for_llm)} with LLM for boundary identification (tokens: {self._count_tokens(doc_chunk_content)}).")
            prompt = self.set_prompt_boundaries(doc_chunk_content)
            response = self.llm.generate(prompt)
            chunk_boundaries = self._parse_llm_response(response, "boundaries")
            if chunk_boundaries: 
                self.logger.info(f"Extracted {len(chunk_boundaries)} boundaries from part {i+1}.")
                collected_boundaries_raw.extend(chunk_boundaries)
            else:
                 self.logger.info(f"No boundaries extracted from part {i+1}.")
        
        seen_normalized_boundaries: Set[str] = set()
        unique_ordered_boundaries: List[str] = []
        for raw_b in collected_boundaries_raw:
            normalized_b = re.sub(r'\s+', ' ', str(raw_b)).strip() # Ensure string and normalize
            if normalized_b and normalized_b not in seen_normalized_boundaries:
                unique_ordered_boundaries.append(str(raw_b)) # Use original raw (but as string) for splitting
                seen_normalized_boundaries.add(normalized_b)
        
        self.logger.info(f"Total {len(unique_ordered_boundaries)} unique, ordered boundaries collected from {len(collected_boundaries_raw)} raw boundaries for long document processing.")
        
        if not unique_ordered_boundaries:
            self.logger.warning("No unique boundaries found in long document. Treating entire content as a single chunk.")
            return [self.processed_book_content] if self.processed_book_content.strip() else []
            
        # IMPORTANT: Split the *full* processed_book_content using collected boundaries
        return self._split_by_boundaries(self.processed_book_content, unique_ordered_boundaries)
    
    def _process_long_document_by_chapter_markers(self) -> list[str]:
        """Processes a long document to find chapter markers and then splits (uses pre-split self.initial_doc_chunks_for_llm)."""
        self.logger.info("Processing long document for chapter marker identification using pre-split parts...")
        if not self.initial_doc_chunks_for_llm:
            self.logger.error("Long document processing called, but no initial_doc_chunks_for_llm were prepared.")
            return [self.processed_book_content] if self.processed_book_content.strip() else []

        collected_markers_raw: List[str] = []
        for i, doc_chunk_content in enumerate(tqdm(self.initial_doc_chunks_for_llm, desc="LLM chapter marker identification on parts")):
            self.logger.info(f"Processing part {i+1}/{len(self.initial_doc_chunks_for_llm)} with LLM for chapter marker identification (tokens: {self._count_tokens(doc_chunk_content)}).")
            prompt = self.set_prompt_chapter_markers(doc_chunk_content)
            response = self.llm.generate(prompt)
            chunk_markers = self._parse_llm_response(response, "chapter_markers")
            if chunk_markers: 
                self.logger.info(f"Extracted {len(chunk_markers)} chapter markers from part {i+1}.")
                collected_markers_raw.extend(chunk_markers)
            else:
                self.logger.info(f"No chapter markers extracted from part {i+1}.")
        
        seen_normalized_markers: Set[str] = set()
        unique_ordered_markers: List[str] = []
        for raw_m in collected_markers_raw:
            normalized_m = re.sub(r'\s+', ' ', str(raw_m)).strip() # Ensure string and normalize
            if normalized_m and normalized_m not in seen_normalized_markers:
                unique_ordered_markers.append(str(raw_m)) 
                seen_normalized_markers.add(normalized_m)

        self.logger.info(f"Total {len(unique_ordered_markers)} unique, ordered chapter markers collected from {len(collected_markers_raw)} raw markers for long document.")
        
        if not unique_ordered_markers:
            self.logger.warning("No unique chapter markers found in long document. Treating entire content as a single chunk.")
            return [self.processed_book_content] if self.processed_book_content.strip() else []

        # IMPORTANT: Split the *full* processed_book_content using collected markers
        return self._split_by_boundaries(self.processed_book_content, unique_ordered_markers)

    def _log_chunk_statistics(self, split_method_name: str) -> None:
        """Logs statistics about the generated chunks."""
        if not self.chunks:
            self.logger.warning(f"No chunks generated for method '{split_method_name}', cannot log statistics.")
            return

        num_chunks = len(self.chunks)
        chunk_token_counts = [self._count_tokens(chunk) for chunk in self.chunks]
        
        total_tokens_in_chunks = sum(chunk_token_counts)
        avg_tokens = total_tokens_in_chunks / num_chunks if num_chunks > 0 else 0
        min_tokens = min(chunk_token_counts) if chunk_token_counts else 0
        max_tokens = max(chunk_token_counts) if chunk_token_counts else 0

        self.logger.info(f"--- Chunk Statistics ({split_method_name}) ---")
        self.logger.info(f"  Number of chunks: {num_chunks}")
        self.logger.info(f"  Total tokens in all chunks: {total_tokens_in_chunks}")
        self.logger.info(f"  Average tokens per chunk: {avg_tokens:.2f}")
        self.logger.info(f"  Min tokens in a chunk: {min_tokens}")
        self.logger.info(f"  Max tokens in a chunk: {max_tokens}")
        self.logger.info(f"  Target semantic chunk tokens (config): {self.chunk_tokens}")
        self.logger.info(f"  Min chunk tokens for merging (config): {self.min_chunk_tokens_for_merge}")
        self.logger.info(f"--- End of Chunk Statistics ---")

    def save_chunks(self, output_dir: str, prefix: str = "chunk_", split_method_name: str = "unknown") -> None:
        """保存切分后的块到文件，并保存元数据"""
        import os 
        
        if not self.chunks:
            self.logger.error("No chunks available to save.")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create output directory {output_dir}. Error: {e}")
            return
        
        for i, chunk_content in enumerate(self.chunks):
            file_path = os.path.join(output_dir, f"{prefix}{i+1:04d}.txt") 
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(chunk_content)
            except IOError as e:
                self.logger.error(f"Failed to write chunk {i+1} to {file_path}. Error: {e}")
        
        self.logger.info(f"Successfully saved {len(self.chunks)} chunks to {output_dir}")

        chunk_token_counts = [self._count_tokens(c) for c in self.chunks]
        avg_tokens_val = sum(chunk_token_counts) / len(self.chunks) if self.chunks else 0
        
        metadata = {
            "source_book_title": self.book_title if self.book_title else "N/A",
            "total_original_tokens": self.tokens_num_original,
            "total_full_processed_tokens": self.tokens_num_full_processed,
            "splitting_method_used": split_method_name,
            "splitter_config": {
                "target_chunk_tokens_for_semantic_split": self.chunk_tokens,
                "max_llm_call_tokens": self.max_llm_tokens,
                "chunk_overlap_for_initial_split": self.chunk_overlap,
                "min_chunk_tokens_for_merge": self.min_chunk_tokens_for_merge,
                "oversized_chunk_factor": self.OVERSIZED_CHUNK_FACTOR,
                "max_resplit_iterations": self.MAX_RESPLIT_ITERATIONS,
            },
            "chunk_statistics": {
                "number_of_chunks": len(self.chunks),
                "token_counts_per_chunk": chunk_token_counts,
                "total_tokens_in_chunks": sum(chunk_token_counts),
                "avg_tokens_per_chunk": float(f"{avg_tokens_val:.2f}"),
                "min_tokens_in_chunk": min(chunk_token_counts) if chunk_token_counts else 0,
                "max_tokens_in_chunk": max(chunk_token_counts) if chunk_token_counts else 0,
            }
        }
        metadata_path = os.path.join(output_dir, f"{prefix}metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Metadata saved to {metadata_path}")
        except IOError as e:
            self.logger.error(f"Failed to write metadata to {metadata_path}. Error: {e}")
        except TypeError as e:
            self.logger.error(f"Failed to serialize metadata to JSON. Error: {e}. Metadata: {metadata}")

# Example Usage (outside the class, for testing):
# if __name__ == '__main__':
#     # This is a placeholder LLM. Replace with your actual LLM implementation.
#     class MockLLM(LLM):
#         def __init__(self, name="mock_llm"): self.name = name # Add dummy init
#         def generate(self, prompt: str) -> str:
#             print(f"\n--- MockLLM ({self.name}) received prompt for target_chunk_tokens={spliter.chunk_tokens if 'spliter' in globals() else 'N/A'}: ---")
#             print(prompt[:300] + "...")
#             print("--- End of MockLLM prompt ---\n")
#             if "directly" in prompt:
#                 # Simulate direct chunking, try to respect token limits somewhat
#                 if "This is a long repeating sentence" in prompt: # from long content test
#                     return json.dumps({"chunks": ["This is a long repeating sentence for testing long document processing. This is a long repeating sentence for testing long document processing.", "Marker One. This is a long repeating sentence for testing long document processing."]})
#                 return json.dumps({"chunks": ["This is the first direct chunk, quite short.", "This is the second direct chunk, also short."]})
#             elif "boundaries" in prompt:
#                 if "This is a long repeating sentence" in prompt: # from long content test
#                     if "Marker One" in prompt and "Marker Two" not in prompt:
#                         return json.dumps({"boundaries": ["testing long document processing. Marker One."]})
#                     elif "Marker Two" in prompt:
#                          return json.dumps({"boundaries": ["testing long document processing. Marker Two."]})
#                     else: # First part
#                         return json.dumps({"boundaries": ["sentence for testing long document processing."]}) # A short boundary
#                 elif "first chapter. End of Chapter 1." in prompt: # from boundary test
#                      return json.dumps({"boundaries": ["first chapter. End of Chapter 1."]})
#                 else: # Default for re-splitting oversized
#                     # Try to split the content in half if it's for re-splitting
#                     # This is a very crude simulation
#                     content_to_find_boundary = prompt.split("Here is the text to be divided:")[-1].strip()
#                     if len(content_to_find_boundary) > 50 : # if content is somewhat long
#                         mid_point = len(content_to_find_boundary) // 2
#                         # Find a space near midpoint
#                         space_idx = content_to_find_boundary.rfind(" ", 0, mid_point + 10)
#                         if space_idx == -1: space_idx = mid_point
#                         boundary_text = content_to_find_boundary[:space_idx+5] # take a bit more
#                         # ensure boundary is not too long
#                         boundary_text = boundary_text[:100] if len(boundary_text) > 100 else boundary_text
#                         # find last sentence ender
#                         match_dot = re.search(r'[.?!]', boundary_text[::-1])
#                         if match_dot:
#                             boundary_text = boundary_text[:len(boundary_text) - match_dot.start()]

#                         print(f"MockLLM (boundary re-split) found boundary: '{boundary_text}'")
#                         return json.dumps({"boundaries": [boundary_text.strip()]})
#                     return json.dumps({"boundaries": []}) # No boundary if too short
#             elif "chapter_markers" in prompt:
#                 if "This is a long repeating sentence" in prompt: # from long content test
#                     if "Marker One" in prompt and "Marker Two" not in prompt:
#                         return json.dumps({"chapter_markers": ["Marker One."]})
#                     elif "Marker Two" in prompt:
#                          return json.dumps({"chapter_markers": ["Marker Two."]})
#                     else: # First part
#                         return json.dumps({"chapter_markers": []}) 
#                 return json.dumps({"chapter_markers": ["first chapter. End of Chapter 1.", "adventure unfolds. End of Chapter 2."]})
#             return json.dumps({}) # Default empty JSON object

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     sample_book_content_chapters = """
#     Chapter 1: The Beginning.
#     This is the first sentence of the first chapter. It has several paragraphs. This is a very short sentence.
#     This is the second paragraph of the first chapter. End of Chapter 1.

#     Chapter 2: The Middle. Tiny.
#     Welcome to the second chapter. This chapter is also interesting.
#     It continues the story. The adventure unfolds. End of Chapter 2.

#     Chapter 3: The End. Also very small.
#     Finally, we reach the third chapter. The story concludes here.
#     All loose ends are tied up. The end.
#     """
#     mock_llm_instance = MockLLM(name="chapters_llm")
#     spliter = LLMSplitter(llm=mock_llm_instance, 
#                          book_content=sample_book_content_chapters, 
#                          book_title="My Sample Book Chapters",
#                          chunk_tokens=30, 
#                          max_llm_tokens=100, 
#                          min_chunk_tokens_for_merge=10 
#                          )
    
#     print("\n--- Testing Chapter Marker Splitting with Merge ---")
#     chapter_chunks = spliter.generate_chunks_by_chapter_markers()
#     spliter.save_chunks("./output_chapters_merged", prefix="chap_mrg_", split_method_name="Chapter-Marker (Merged)")
#     for i, chunk in enumerate(spliter.chunks):
#         print(f"Final Chunk {i+1} ({spliter._count_tokens(chunk)} tokens): {chunk[:150]}...")

#     print("\n--- Testing Direct Splitting with Merge ---")
#     # Content designed to produce small chunks initially
#     sample_direct_content = "First part. Second part, very short. Third part. Fourth, also tiny. Fifth part is longer and should be fine."
#     mock_llm_direct = MockLLM(name="direct_llm")
#     # Override generate for this specific test to control direct output better
#     def mock_direct_generate(prompt_str):
#         print(f"\n--- MockLLM (direct_llm) received prompt for direct: ---")
#         print(prompt_str[:300] + "...")
#         if "First part" in prompt_str:
#             return json.dumps({"chunks": ["First part.", "Second part, very short.", "Third part.", "Fourth, also tiny.", "Fifth part is longer and should be fine."]})
#         return json.dumps({"chunks":[]})
#     mock_llm_direct.generate = mock_direct_generate

#     spliter_direct = LLMSplitter(llm=mock_llm_direct,
#                                 book_content=sample_direct_content,
#                                 book_title="Direct Merge Test",
#                                 chunk_tokens=20, # Target
#                                 max_llm_tokens=100,
#                                 min_chunk_tokens_for_merge=8 # Merge if less than 8 tokens
#                                 )
#     direct_chunks = spliter_direct.generate_chunks_directly()
#     spliter_direct.save_chunks("./output_direct_merged", prefix="direct_mrg_", split_method_name="Direct LLM (Merged)")
#     for i, chunk in enumerate(spliter_direct.chunks):
#         print(f"Final Direct Chunk {i+1} ({spliter_direct._count_tokens(chunk)} tokens): {chunk[:150]}...")


#     print("\n--- Testing Boundary Splitting with Oversized and Merge ---")
#     # Content that might lead to an oversized chunk then re-split, then merge
#     # Oversized chunk factor is 1.5. Target is 20. So > 30 triggers re-split.
#     # Min merge is 8.
#     boundary_test_content = "This is the first section, quite long, maybe thirty five tokens or so to test re-splitting. This is a tiny follow up. Then another medium part. And a final tiny bit."
#     # Processed: "This is the first section, quite long, maybe thirty five tokens or so to test re-splitting. This is a tiny follow up. Then another medium part. And a final tiny bit."
#     # Tokens for "This is the first section, quite long, maybe thirty five tokens or so to test re-splitting." is 19.
#     # Tokens for "This is the first section, quite long, maybe thirty five tokens or so to test re-splitting. This is a tiny follow up." is 27.
#     # Tokens for "This is the first section, quite long, maybe thirty five tokens or so to test re-splitting. This is a tiny follow up. Then another medium part." is 36. (This should trigger re-split if it's one chunk)

#     mock_llm_boundary_complex = MockLLM(name="boundary_complex_llm")
#     def mock_boundary_complex_generate(prompt_str):
#         print(f"\n--- MockLLM (boundary_complex_llm) received prompt: ---")
#         print(prompt_str[:300] + "...")
#         content_for_boundaries = prompt_str.split("Here is the text to be divided:")[-1].strip()

#         if "thirty five tokens or so to test re-splitting. This is a tiny follow up. Then another medium part." in content_for_boundaries and "final tiny bit" not in content_for_boundaries : # Initial call for the whole text
#             # Return a boundary that creates one large chunk and one small one
#             return json.dumps({"boundaries": ["thirty five tokens or so to test re-splitting. This is a tiny follow up. Then another medium part."]}) # This chunk will be 36 tokens
#         elif "thirty five tokens or so to test re-splitting. This is a tiny follow up. Then another medium part." in content_for_boundaries: # This is the oversized chunk being re-split
#             # LLM should split this oversized chunk
#             return json.dumps({"boundaries": ["thirty five tokens or so to test re-splitting."]}) # Splits into 19 tokens and " This is a tiny follow up. Then another medium part." (16 tokens)
#         return json.dumps({"boundaries":[]})
#     mock_llm_boundary_complex.generate = mock_boundary_complex_generate

#     spliter_boundary = LLMSplitter(llm=mock_llm_boundary_complex,
#                                 book_content=boundary_test_content,
#                                 book_title="Boundary Complex Test",
#                                 chunk_tokens=20, # Target
#                                 max_llm_tokens=150, # LLM call limit
#                                 min_chunk_tokens_for_merge=8,
#                                 chunk_overlap=5
#                                 )
#     boundary_chunks = spliter_boundary.generate_chunks_by_boundaries()
#     spliter_boundary.save_chunks("./output_boundary_complex", prefix="bnd_cmplx_", split_method_name="Boundary Complex (Re-split & Merged)")
#     for i, chunk in enumerate(spliter_boundary.chunks):
#         print(f"Final Boundary Chunk {i+1} ({spliter_boundary._count_tokens(chunk)} tokens): {chunk[:150]}...")

#     print("\n--- Testing Long Document with Pre-splitting (Boundary method) ---")
#     long_text_part = "This is a long repeating sentence for testing long document pre-splitting. " # 12 tokens
#     # max_llm_tokens = 100. initial_chunk_size_tokens = 100 - LLM_PROCESSING_SAFETY_MARGIN (20) = 80 tokens.
#     # 80 tokens / 12 tokens_per_part approx 6-7 parts.
#     # original_book_content will be split into chunks of ~80 tokens for LLM calls.
#     # self.processed_book_content will be the full string.
#     very_long_content = (long_text_part * 10) + "Boundary Alpha. " + (long_text_part * 10) + "Boundary Beta. " + (long_text_part * 10)
#     # Total tokens: (12*30) + 3 + 3 = 366 tokens. is_long_document = True.
#     # initial_doc_chunks_for_llm will have ~366/80 = 4-5 chunks.

#     mock_llm_long_boundary = MockLLM(name="long_boundary_llm")
#     def mock_long_boundary_generate(prompt_str):
#         print(f"\n--- MockLLM (long_boundary_llm) received prompt for long doc part: ---")
#         print(prompt_str[:150] + "...")
#         # LLM sees parts of the long text.
#         if "Boundary Alpha" in prompt_str and "Boundary Beta" not in prompt_str:
#             return json.dumps({"boundaries": ["pre-splitting. Boundary Alpha."]})
#         elif "Boundary Beta" in prompt_str:
#             return json.dumps({"boundaries": ["pre-splitting. Boundary Beta."]})
#         else: # For other parts, maybe return a generic boundary or none
#             # find a sentence end in the first 50 chars
#             first_part = prompt_str[:80]
#             end_match = re.search(r'[.?!]', first_part)
#             if end_match:
#                 return json.dumps({"boundaries": [first_part[:end_match.end()]]})
#             return json.dumps({"boundaries": []})
#     mock_llm_long_boundary.generate = mock_long_boundary_generate
    
#     spliter_long_b = LLMSplitter(llm=mock_llm_long_boundary,
#                               book_content=very_long_content,
#                               book_title="Very Long Book For Boundary",
#                               chunk_tokens=60, # Target for final semantic chunks
#                               max_llm_tokens=100, # LLM call limit (tokens) -> initial_doc_chunks ~80 tokens
#                               chunk_overlap=10,
#                               min_chunk_tokens_for_merge=20
#                               )
#     long_b_chunks = spliter_long_b.generate_chunks_by_boundaries()
#     spliter_long_b.save_chunks("./output_long_boundaries", prefix="long_bnd_", split_method_name="Boundary (Long Doc Pre-split & Merged)")
#     for i, chunk in enumerate(spliter_long_b.chunks):
#         print(f"Final Long Boundary Chunk {i+1} ({spliter_long_b._count_tokens(chunk)} tokens): {chunk[:150]}...")
