import re
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader
from src.utils import save_json

path_builder = NovelQAPathBuilder('./data/NovelQA')

chapter_patterns = {}
# chapter_patterns["B00"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^PART\s+[IVXLCDM]+\.\s+.*')]
# chapter_patterns["B02"] = [re.compile(r'^[A-Z\s]+$')]
# chapter_patterns["B05"] = [re.compile(r'^[IVXLCDM]+\.$'), re.compile(r'^PART\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B07"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.$'), re.compile(r'^BOOK\s+[IVXLCDM]+\.\s*$')]
# chapter_patterns["B10"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*\.$')]
# chapter_patterns["B11"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B13"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B14"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^Book\sthe\s.*')]
# chapter_patterns["B17"] = [re.compile(r'^Chapter\s+[A-Z][a-z]+\s*$'), re.compile(r'^BOOK\s+[A-Z]+\s*$')]
# chapter_patterns["B18"] = [re.compile(r'^[IVXLCDM]+$')]

def split_chapters(file_path, nested_pattern):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()

    chapters = []  # 存储所有章节内容
    current_chapter = None  # 当前正在处理的章节

    for line in lines:
        matched = False
        for pattern in nested_pattern:
            if pattern.match(line):
                matched = True
                break
        # 检查当前行是否为章节行
        if matched:
            # 如果当前已经有一个章节在收集，则保存它
            if current_chapter is not None:
                chapters.append(''.join(current_chapter))
            # 开始新的章节
            current_chapter = [line]
        else:
            # 如果当前处于章节中，则添加该行到当前章节
            if current_chapter is not None:
                current_chapter.append(line)
    
    # 处理最后一个章节
    if current_chapter is not None:
        chapters.append(''.join(current_chapter))
    
    return chapters

import unicodedata

def remove_invisible_chars(s):
    return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

for book_id in chapter_patterns:
    book_path = path_builder.get_book_path(book_id)
    chapters = split_chapters(book_path, chapter_patterns[book_id])  # 替换为你的文件路径
    book_loader = BookLoader(book_path, book_id)
    book_loader.load()
    book_content = book_loader.get_content()
    lines = book_content.split('\n')
    lines = [remove_invisible_chars(line) for line in lines]
    while len(lines) > 0 and not lines[0].strip():
        lines.pop(0)
    with open(f"./structures/responses/{book_id}.txt", 'w', encoding='utf-8') as f:
        f.write(lines[0] + '\n')
    with open(f"./structures/responses/{book_id}.txt", 'a', encoding='utf-8') as f:
        for chapter in chapters:
            f.write(chapter.split('\n')[0] + '\n')
            # print(f"章节 {idx} 的内容：")
            # print(chapter[:200])  # 打印前200字符以避免过长输出
            # print("-" * 50)
    # chapters = split_chapters("data/NovelQA/Books/PublicDomain/B00.txt")  # 替换为你的文件路径
    # for idx, chapter in enumerate(chapters, 1):
    #     print(f"章节 {idx} 的内容：")
    #     print(chapter[:200])  # 打印前200字符以避免过长输出
    #     print("-" * 50)


# from src.llm import get_llm

# def build_split_prompt(book_content):
#     return f"""Read the text below and output **only** the chapter and sub‑chapter headings exactly as they appear. Do **not** output any prose from the text itself. Pay attention to possible nesting.\nGiven text:\n{book_content}\nYou should output in the following format:\n# Novel title\n## Subsection 1 heading\n### Subsection 1 of subsection 1 heading\n### Subsection 1 of subsection 1 heading\n## Subsection 2 heading\n### Subsection 2 of subsection 1 heading\n### Subsection 2 of subsection 1 heading\nDon't output anything else, just output the result."""

# llm = get_llm('gemini')