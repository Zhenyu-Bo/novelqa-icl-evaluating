"""章节切分的示例代码"""

from src.chapterizer import Chapterizer
from src.loader import BookLoader, BookMetaDataLoader
from src.path_builder import NovelQAPathBuilder
import re
import os

# 一些书籍的章节标题格式
chapter_patterns = {}
chapter_patterns["B00"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^PART\s+[IVXLCDM]+\.\s+.*')]
chapter_patterns["B02"] = [re.compile(r'^[A-Z\s]+$')]
chapter_patterns["B05"] = [re.compile(r'^[IVXLCDM]+\.$'), re.compile(r'^PART\s+[IVXLCDM]+\s*$')]
chapter_patterns["B07"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.$'), re.compile(r'^BOOK\s+[IVXLCDM]+\.\s*$')]
chapter_patterns["B10"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*\.$')]
chapter_patterns["B11"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
chapter_patterns["B13"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
chapter_patterns["B14"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^Book\sthe\s.*')]
chapter_patterns["B17"] = [re.compile(r'^Chapter\s+[A-Z][a-z]+\s*$'), re.compile(r'^BOOK\s+[A-Z]+\s*$')]
chapter_patterns["B18"] = [re.compile(r'^[IVXLCDM]+$')]

# 示例使用
BOOK_ID = 'B00'

# 加载书籍内容和标题
path_builder = NovelQAPathBuilder('./data/NovelQA')
book_loader = BookLoader(path_builder.get_book_path(BOOK_ID), BOOK_ID)
book_loader.load()
book_content = book_loader.get_content()
meta_data_loader = BookMetaDataLoader(path_builder.get_meta_data_path())
meta_data_loader.load()
title = meta_data_loader.get_title(BOOK_ID)

# 章节化
chapterizer = Chapterizer(book_content, title, chapter_patterns[BOOK_ID])

# 生成章节标题
markdown = chapterizer.to_markdwon()

# 保存章节标题
if not os.path.exists(f'structures/titles'):
    os.makedirs(f'structures/titles')

with open(f'structures/titles/{BOOK_ID}.txt', 'w') as f:
    f.write(markdown)

# 中断以便用户检查生成之标题结构是否正确，如果不正确，用户可以修改生成之标题结构
_ = input("请检查生成的章节标题是否正确，按回车继续")

# 加载用户修改之章节标题结构
with open(f'structures/titles/{BOOK_ID}.txt', 'r') as f:
    markdown = f.read()

# 章节化
chapterizer.structure_from_markdown(markdown)

# 保存章节结构
chapterizer.save_structure(f'structures/{BOOK_ID}.json')

chapter_dict, chapter_list = chapterizer.get_chapter_contents()

if not os.path.exists(f'structures/chapters/{BOOK_ID}'):
    os.makedirs(f'structures/chapters/{BOOK_ID}')

for chapter in chapter_list:
    with open(f'structures/chapters/{BOOK_ID}/{chapter}.txt', 'w') as f:
        f.write(chapter_dict[chapter])
