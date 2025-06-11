# """章节切分的示例代码"""

# from src.chapterizer import Chapterizer
# from src.loader import BookLoader, BookMetaDataLoader
# from src.path_builder import NovelQAPathBuilder
# import os

# BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
# BOOK_IDS.remove("B06")
# BOOK_IDS.remove("B30")
# BOOK_IDS.remove("B45")
# BOOK_IDS.remove("B48") # 内容太长，予以舍弃
# BOOK_IDS = ["B00"]

# i = 0
# while i < len(BOOK_IDS):
#     book_id = BOOK_IDS[i]
#     path_builder = NovelQAPathBuilder('../NovelQA')
#     book_loader = BookLoader(path_builder.get_book_path(book_id), book_id)
#     book_loader.load()
#     book_content = book_loader.get_content()
#     meta_data_loader = BookMetaDataLoader(path_builder.get_meta_data_path())
#     meta_data_loader.load()
#     title = meta_data_loader.get_title(book_id)

#     # 章节化
#     chapterizer = Chapterizer(book_content, title)

#     # 生成章节标题
#     # markdown = chapterizer.to_markdown()
#     # print(f"生成章节标题 {book_id}:\n{markdown}")

#     # # 保存章节标题
#     # if not os.path.exists(f'structures/titles'):
#     #     os.makedirs(f'structures/titles')

#     # with open(f'structures/titles/{book_id}.txt', 'w', encoding='utf-8') as f:
#     #     f.write(markdown)

#     # # 中断以便用户检查生成之标题结构是否正确，如果不正确，用户可以修改生成之标题结构
#     # # _ = input("请检查生成的章节标题是否正确，按回车继续")

#     # # 加载用户修改之章节标题结构
#     # with open(f'structures/titles/{book_id}.txt', 'r', encoding='utf-8') as f:
#     #     markdown = f.read()

#     # # 章节化
#     # chapterizer.structure_from_markdown(markdown)

#     # # 保存章节结构
#     # chapterizer.save_structure(f'structures/{book_id}.json')

#     chapter_dict, chapter_list = chapterizer.get_chapter_contents()

#     if not os.path.exists(f'structures/chapters/{book_id}'):
#         os.makedirs(f'structures/chapters/{book_id}')

#     for idx, chapter in enumerate(chapter_list):
#         with open(f'structures/chapters/{book_id}/{idx}.txt', 'w', encoding='utf-8') as f:
#             f.write(chapter_dict[chapter])
    
#     print(f"章节化完成 {book_id}")
#     i += 1
#     command = input("按回车继续")
#     if command == 'q':
#         break
#     elif command != 'r':
#         i += 1

from src.path_builder import NovelQAPathBuilder
from src.llm import LLM, get_llm
from src.chapterizer import Chapterizer, LLMSplitter
from src.splitter import HybridSplitter
from src.loader import BookLoader
import os
import json
from dotenv import load_dotenv
import argparse
import google.generativeai as genai
import logging

logging.basicConfig(
    level=logging.INFO,  # 或 logging.INFO 查看更多信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('error.log')  # 同时保存到文件
    ]
)

parser = argparse.ArgumentParser(description='Chapterize a book using LLM.')
parser.add_argument('--book_id', type=str, default="all", help='ID of the book to chapterize')
parser.add_argument('--model', type=str, default='gemini2.0', help='LLM model to use for chapterization')
args = parser.parse_args()

load_dotenv()
NOVELQA_PATH = '../NovelQA'
path_builder = NovelQAPathBuilder(NOVELQA_PATH)
model_name = args.model
api_key = os.getenv('GEMINI_API_KEY') if 'gemini' in model_name.lower() else os.getenv('DEEPSEEK_API_KEY')

BOOK_IDS = ["B00", "B05", "B09", "B13", "B14", "B16", "B17", "B20", "B22", "B24",
            "B25", "B29", "B33", "B34", "B37", "B43", "B44", "B53", "B55", "B60"]
# BOOK_IDS.remove("B06")
# BOOK_IDS.remove("B30")
# BOOK_IDS.remove("B45")
if args.book_id != 'all':
    BOOK_IDS = [args.book_id]

i = 0
while i < len(BOOK_IDS):
    print(f"正在处理书籍: {BOOK_IDS[i]}")
    book_id = BOOK_IDS[i]
    book_path = path_builder.get_book_path(book_id)
    book_loader = BookLoader(book_id=book_id, book_path=book_path)
    book_loader.load()
    book_content = book_loader.get_content()
    llm: LLM = get_llm(model_name, api_key)
    # splitter = LLMSplitter(llm, book_content, max_llm_tokens=100000, chunk_tokens=50000)
    # splitter.generate_chunks_by_chapters()
    # chunks = splitter.split_recursive(book_content, by_chapter=True)
    # splitter.save_chunks(f'cache/llm/splitter/{book_id}')
    splitter = HybridSplitter(book_content=book_content,book_title="Unknown", llm=llm, llm_splitter_max_llm_tokens=100000, max_chunk_tokens=50000 ,llm_splitter_max_retries=5)
    splitter.split()
    metadata = splitter.save_chunks_to_json(f'cache/hybrid/{book_id}.json')
    print(f"已生成 {book_id} 的章节切分结果，元数据: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
    i += 1
    command = input(f"按回车继续处理下一个书籍，输入 'q' 退出: ")
    if command.lower() == 'q':
        break
    elif command == '\r':
        i += 1
