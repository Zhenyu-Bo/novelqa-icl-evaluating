## 1. 项目介绍
本项目通过将长文本分割成可管理的片段，对每个片段单独分析，然后汇总结果并回答来提升大模型在长文本阅读任务上的表现。

## 2. 代码组织

```
novelqa-icl-evaluating/
├── reduce.py                  # 主执行脚本
├── src/                       # 源代码目录
│   ├── chapterizer.py         # 文本章节化和分割
│   ├── extractor.py           # 答案提取工具
│   ├── llm.py                 # LLM 接口
│   ├── loader.py              # 数据加载器
│   ├── path_builder.py        # 路径构建工具
│   ├── prompt.py              # 提示词构建
│   ├── splitter.py            # 文本分割器
│   └── utils.py               # 通用工具函数
├── .env                       # 环境变量配置（API密钥等）
└── requirements.txt           # 项目依赖
```

## 3. 核心模块介绍

### 3.1 reduce.py

主执行脚本，实现了整个处理流程：

1. 处理命令行参数
2. 加载书籍和问题数据
3. 调用文本分割器
4. 对分块应用LLM处理
5. 合并分块结果
6. 保存最终答案

### 3.2 splitter.py

包含 HybridSplitter 类，按下列步骤实现智能文本分割：

1. 使用 `Chapterizer` 基于正则表达式提取章节标题，然后根据章节标题进行切分，若出现错误则切分结果只有一个包含全部内容的分块。
2. 对于上一步中划分得到的过大的分块（分块阈值在初始化时确定），使用 `LLMSPlitter` 调用 LLM 进一步切分。若 LLM 切分失败，则使用 `RecursiveCharacterTextSplitter` 实行递归分割。

可用函数：
```python
class HybridSplitter:
    def __init__(self, book_content, llm, book_title, max_chunk_tokens=50000, 
                 llm_splitter_max_llm_tokens=100000, llm_splitter_chunk_overlap=100, 
                 llm_splitter_max_retries=5, llm_splitter_retry_delay=1.0, 
                 llm_splitter_max_chunk_tokens_for_merge=20000, 
                 llm_splitter_min_chunk_tokens_for_merge=50, 
                 char_overlap_fallback=50, chars_per_token_estimate=3.5): 
        # 初始化分割器，配置各种参数
        # 参数：
        #   book_content (str): 要分割的文本内容
        #   llm (LLM): 用于辅助分割的LLM实例
        #   book_title (str): 书籍标题
        #   max_chunk_tokens (int): 分块的最大token数
        #   llm_splitter_max_llm_tokens (int): LLM分割器调用 LLM 时的最大输入token数
        #   llm_splitter_chunk_overlap (int): LLM分割器的分块重叠token数
        #   llm_splitter_max_retries (int): LLM分割器的最大重试次数
        #   llm_splitter_retry_delay (float): LLM分割器的重试延迟时间
        #   llm_splitter_max_chunk_tokens_for_merge (int): LLM分割器合并分块的最大token数
        #   llm_splitter_min_chunk_tokens_for_merge (int): LLM分割器合并分块的最小token数
        #   char_overlap_fallback (int): 字符分割器的重叠字符数
        #   chars_per_token_estimate (float): 字符token估计值
    
    def split(self):
        # 执行分块操作，返回分块列表
        # 无需参数，返回分割后的文本块列表
    
    def save_chunks_to_json(self, output_path):
        # 保存分块结果到JSON文件，返回元数据
        # output_path: 保存JSON文件的路径
    
    def get_metadata(self):
        # 返回分块的相关信息
```

### 3.3 chapterizer.py

包含用于文本章节化的类：

Chapterizer: 基于正则表达式提取章节标题，然后根据章节标题按章节切分
LLMSplitter: 利用LLM识别章节结构和语义边界进行切分，支持按章节切分或按语义边界切分

### 3.4 llm.py

LLM接口封装，支持 Gemini 和 DeepSeek 两大类模型
可通过 `get_llm(llm_name, api_key)` 函数获得模型实例以进行对话，通过 `llm_name` 指定模型，`api_key` 需设为自己的 API Key, `llm_name` 支持的选项有：

* `gemini2.0`: gemini-2.0-flash
* `gemini2.5-flash`: gemini-2.5-flash-preview-05-20
* `gemini2.5-pro`: gemini-2.5-pro-preview-03-25
* `deepseek`: deepseek-v3
* `deepseek-r1`: deepseek-r1

获得模型实例后，可通过 `generate(prompt)` 方法进行对话


### 3.5 loader.py

数据加载组件：

* BookLoader: 加载小说文本
  可用函数：
  ```python
  class BookLoader:
    def __init__(self, path: str, book_id: str):
        # 初始化书籍加载器
        # 参数：
        #   path (str): 书籍文件路径
        #   book_id (str): 书籍ID，用于标识书籍
    
    def load(self) -> None:
        # 加载书籍内容
        # 从指定的 `book_path` 读取文本内容，并计算字数和字符数。
        # 无参数
    
    def get_content(self) -> str:
        # 获取书籍内容
        # 返回值：
        #   str: 书籍的文本内容
    
    def get_char_count(self) -> int:
        # 获取书籍的字符数
        # 返回值：
        #   int: 书籍的字符数
    
    def get_word_count(self) -> int:
        # 获取书籍的字数
        # 返回值：
        #   int: 书籍的字数
    
    def get_id(self) -> str:
        # 获取书籍ID
        # 返回值：
        #   str: 书籍ID
  ```
* QuestionLoader: 加载问题数据
  可用函数：
  ```python
  class QuestionLoader:
    def __init__(self, question_path: str, book_id: str):
        # 初始化问题加载器
        # 参数：
        #   question_path (str): 问题文件路径，JSON格式
        #   book_id (str): 书籍ID，用于关联问题和书籍
    
    def load(self) -> None:
        # 加载问题数据
        # 从指定的 `question_path` 读取JSON格式的问题数据，并存储在 `self.questions` 字典中。
        # 无参数
    
    def get_whole(self) -> Dict:
        # 获取所有问题数据
        # 返回值：
        #   Dict: 包含所有问题的字典，键为问题ID，值为问题内容
    
    def get_ith_question(self, i: int) -> QuestionModel:
        # 获取指定索引的问题
        # 参数：
        #   i (int): 问题索引
        # 返回值：
        #   QuestionModel: QuestionModel对象，包含问题的所有信息
    
    def get_next_question(self) -> QuestionModel:
        # 获取下一个问题
        # 按照加载顺序，返回下一个问题，并将指针移动到下一个问题。
        # 返回值：
        #   QuestionModel: QuestionModel对象，包含问题的所有信息
    
    def get_by_id(self, question_id: str) -> QuestionModel:
        # 通过问题ID获取问题
        # 参数：
        #   question_id (str): 要获取的问题ID
        # 返回值：
        #   QuestionModel: QuestionModel对象，包含问题的所有信息
    
    def __len__(self) -> int:
        # 获取问题总数
        # 返回值：
        #   int: 问题总数
    
    def __getitem__(self, index: int) -> QuestionModel:
        # 通过索引获取问题
        # 参数：
        #   index (int): 问题索引
        # 返回值：
        #   QuestionModel: QuestionModel对象，包含问题的所有信息
    
    def __setitem__(self, index: int, value: QuestionModel) -> None:
        # 设置指定索引的问题
        # 参数：
        #   index (int): 问题索引
        #   value (QuestionModel): 要设置的QuestionModel对象
    
    def __delitem__(self, index: int) -> None:
        # 删除指定索引的问题
        # 参数：
        #   index (int): 问题索引
    
    def __contains__(self, item: str) -> bool:
        # 检查是否包含指定问题ID
        # 参数：
        #   item (str): 问题ID
        # 返回值：
        #   bool: 如果包含指定问题ID，返回True，否则返回False
  ```
* BookMetaDataLoader: 加载书籍元数据
  可用函数：
  ```python
  class BookMetaDataLoader:
    def __init__(self, meta_data_path: str):
        # 初始化元数据加载器
        # 参数：
        #   meta_data_path (str): 元数据文件路径，JSON格式
    
    def load(self) -> None:
        # 加载元数据
        # 从指定的 `meta_data_path` 读取JSON格式的元数据，并存储在 `self.meta_data` 字典中。
        # 无参数
    
    def build_description(self, book_id: str) -> str:
        # 构建书籍描述
        # 根据元数据中的标题、作者、出版年份和时期，构建书籍的描述信息。
        # 参数：
        #   book_id (str): 书籍ID
        # 返回值：
        #   str: 书籍的描述信息
    
    def get_meta_data(self, book_id: str) -> dict:
        # 获取指定书籍的元数据
        # 参数：
        #   book_id (str): 书籍ID
        # 返回值：
        #   dict: 包含书籍元数据的字典
    
    def get_title(self, book_id: str) -> str:
        # 获取书籍标题
        # 参数：
        #   book_id (str): 书籍ID
        # 返回值：
        #   str: 书籍标题
    
    def get_author(self, book_id: str) -> str:
        # 获取书籍作者
        # 参数：
        #   book_id (str): 书籍ID
        # 返回值：
        #   str: 书籍作者
    
    def get_yearpub(self, book_id: str) -> int:
        # 获取书籍出版年份
        # 参数：
        #   book_id (str): 书籍ID
        # 返回值：
        #   int: 书籍出版年份
    
    def get_period(self, book_id: str) -> str:
        # 获取书籍所属时期
        # 参数：
        #   book_id (str): 书籍ID
        # 返回值：
        #   str: 书籍所属时期
  ```

### 3.6 path_builder.py

文件路径管理：

定义类 NovelQAPathBuilder用于构建 NovelQA 数据集的标准化文件路径。
可用函数：
```python
class NovelQAPathBuilder:
    def __init__(self, base_dir: str):
        # 初始化路径构建器
        # 参数：
        #   base_dir (str): NovelQA 数据集的根目录。所有文件路径都基于这个目录构建。
    
    def get_book_path(self, book_id: str) -> str:
        # 获取书籍文件路径
        # 根据书籍ID构建书籍文件的完整路径。
        # 参数：
        #   book_id (str): 书籍ID，例如 "B00"。
        # 返回值：
        #   str: 书籍文件的完整路径。
        # 异常：
        #   FileNotFoundError: 如果构建的文件路径不存在，则抛出此异常。
    
    def get_question_path(self, book_id: str) -> str:
        # 获取问题文件路径
        # 根据书籍ID构建问题文件的完整路径。
        # 参数：
        #   book_id (str): 书籍ID，例如 "B00"。
        # 返回值：
        #   str: 问题文件的完整路径。
        # 异常：
        #   FileNotFoundError: 如果构建的文件路径不存在，则抛出此异常。
    
    def get_meta_data_path(self) -> str:
        # 获取元数据文件路径
        # 构建元数据文件的完整路径。
        # 返回值：
        #   str: 元数据文件的完整路径。
        # 异常：
        #   FileNotFoundError: 如果构建的文件路径不存在，则抛出此异常。
```

### 3.7 prompt.py

提示词构建工具：构建不同场景的提示词模板

可用函数：
```python
def build_transform_question_prompt(question: str) -> str:
    # 构建转换问题的提示词，使得转换后的问题仍然保留原始问题的逻辑关系，并且可以基于单个文本块的内容进行回答

def build_prompt_icl(chapter_content: str, question: str) -> str:
    # 构建分块处理提示词，引导 LLM 仔细分析文本块内容，找出与问题相关的证据，并给出准确的回答。

def build_prompt_final(question: str) -> str:
    # 构建最终答案生成提示词，引导 LLM 仔细审查每个文本块的回答和证据，避免重复计算，并给出最终答案。

def build_transform_question_prompt2(question: str) -> str:
    # 构建转换问题的提示词（版本2）
    # 与 `build_transform_question_prompt` 类似，但对计数类问题有不同的处理方式。
    # 对于 "how many times" 等计数类问题，不要求 LLM 直接回答数量，而是列出所有相关证据。

def build_prompt_icl2(chapter_content: str, question: str) -> str:
    # 构建分块处理提示词（版本2）
    # 与 `build_prompt_icl` 类似，但要求 LLM 找出所有显然相关的证据，避免过度解读。

def build_prompt_icl_json(chapter_content: str, question_options: str) -> str:
    # 构建分块处理提示词（JSON格式）
    # 与 `build_prompt_icl2` 类似，但要求 LLM 以 JSON 格式输出结果。

def build_prompt_final_json(question: str) -> str:
    # 构建最终答案生成提示词（JSON格式）
    # 让 LLM 分析所有文本块的回答，并选择最佳答案，以 JSON 格式输出结果。
```

### 3.8 extractor.py

答案提取工具：从 LLM 的文本响应中提取答案选项、推理过程和证据。

可用函数：
```python
def extract_entries_no_evidence(response_str: str) -> list[dict]:
    # 解析模型返回的字符串，提取问题id、模型的分析、模型给出的答案

def extract_entries(response_str: str) -> list[dict]:
    # 解析模型返回的字符串，提取问题id、模型的分析、模型给出的答案、模型找到的证据

def extract_option(answer: str) -> str:
    # 提取 LLM 的给出的选择

def split_reasoning_answer(answer: str) -> tuple[str, str]:
    # 将推理和答案分开
```

### 3.9 utils.py

通用工具函数：提供 JSON 文件读写等常用功能。

可用函数：
```python
def load_json(file_path: str) -> dict:
    # 加载JSON文件

def save_json(data: any, file_path: str) -> None:
    # 保存数据到JSON文件
```

## 4. 执行

可以通过 `python reduce.py` 直接执行，也可以通过命令行设置一些参数，下面是支持的参数：

* `--book_id`：指定要处理的书籍 ID。如果设置为 `"all"`，则处理所有书籍。
* `--question_id`：指定要处理的问题 ID。如果设置为 `"all"`，则处理所有问题。
* `--max_workers`：指定用于并行处理的最大工作进程数。可以根据 CPU 核心数进行调整，以充分利用计算资源。
* `--use_cache`：指定是否使用缓存。`1` 表示使用缓存，`0` 表示不使用缓存。使用缓存可以避免重复计算，提高处理效率。
* `--cache_dir`：指定用于存储缓存文件的目录。
* `--output_dir`：指定用于存储输出文件的目录。
* `--skip_answered`：如果设置此标志，则跳过缓存中已回答的问题。默认情况下，此标志为 True。
* `--no_skip_answered`：如果设置此标志，则不跳过缓存中已回答的问题。
* `--max_chunk_tokens`：指定每个分块的最大 token 数。用于控制文本块的大小，以适应 LLM 的处理能力。
* `--max_llm_tokens`：指定使用 LLM 辅助切分调用 LLM 时的最大输入 token 数。用于限制 LLM 的输入长度，避免超出 LLM 的处理能力。
* `--chunk_overlap`：指定 LLM 分割器的分块重叠 token 数。用于在分块之间保留一定的上下文信息，提高 LLM 的处理效果。
* `--max_retries`：指定 LLM 分割器的最大重试次数。用于处理 LLM API 调用失败的情况。
* `--retry_delay`：指定 LLM 分割器的重试延迟时间（秒）。
* `--max_chunk_tokens_for_merge`：指定 LLM 分割器合并分块的最大 token 数。用于将过小的分块合并，减少分块数量。
* `--min_chunk_tokens_for_merge`：指定 LLM 分割器合并分块的最小 token 数。（因为处理过程中发现有时 LLM 可能将页标当作标题，以至于分块过小，此时需要合并）
* `--char_overlap_fallback`：指定字符分割器的重叠字符数。用于在 LLM 分割失败时，使用字符分割器进行备用分割。
* `--chars_per_token_estimate`：指定字符 token 估计值。用于估计文本的 token 数，以便控制分块大小（在使用 `tiktoken` 估计失败时才会起作用）

使用示例：

```python
# 处理所有书籍的所有问题，使用默认参数
python reduce.py

# 处理特定书籍
python reduce.py --book_id B05

# 处理特定问题
python reduce.py --book_id B05 --question_id Q1

# 自定义参数
python reduce.py --max_workers 4 --use_cache 0 --output_dir ./my_results
```

