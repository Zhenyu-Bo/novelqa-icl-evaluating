from src.llm import get_llm
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader
from src.utils import save_json

def build_split_prompt(book_content):
    return f"""Read the text below and output **only** the chapter and sub‑chapter headings exactly as they appear. Do **not** output any prose from the text itself. Pay attention to possible nesting.\nGiven text:\n{book_content}\nYou should output in the following format:\n# Novel title\n## Subsection 1 heading\n### Subsection 1 of subsection 1 heading\n### Subsection 1 of subsection 1 heading\n## Subsection 2 heading\n### Subsection 2 of subsection 1 heading\n### Subsection 2 of subsection 1 heading\nDon't output anything else, just output the result."""

llm = get_llm('gemini')

path_builder = NovelQAPathBuilder('./data/NovelQA')

BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
BOOK_IDS.remove("B06")
BOOK_IDS.remove("B30")
BOOK_IDS.remove("B45")
BOOK_IDS.remove("B48") # 内容太长，予以舍弃

def get_structure(lines, level=1) -> dict:

    def get_level(line: str) -> int:
        return line.count('#')

    structure = {}
    current_line = lines.pop(0)
    current_level = get_level(current_line)
    next_structures = []
    while len(lines) > 0 and get_level(lines[0]) > current_level:
        next_structures.append(get_structure(lines, current_level))
    structure['title'] = current_line.strip('#').strip()
    structure['structures'] = next_structures
    result = structure
    for _ in range(level - current_level - 1):
        result = {'title': '', 'structures': [result]}
    return structure

# with open("temp.txt", 'r', encoding='utf-8') as file:
#     content = file.read()
#     lines = content.split('\n')
#     new_lines = [line for line in lines if line.strip() != '' and line.startswith('#')]
#     structure = get_structure(new_lines)
#     save_json(structure, 'structure.json')
#     print(structure)

# import sys; sys.exit(0)

for book_id in BOOK_IDS:
    if not book_id == 'B03':
        continue
    book_path = path_builder.get_book_path(book_id)
    book_loader = BookLoader(book_path, book_id)
    book_loader.load()
    book_content = book_loader.get_content()
    prompt = build_split_prompt(book_content)
    with open(f"structures/prompts/{book_id}.txt", 'w', encoding='utf-8') as file:
        file.write(prompt)
    # continue
    response = llm.generate(prompt)
    with open(f"structures/responses/{book_id}.txt", 'w', encoding='utf-8') as file:
        file.write(response)
    # with open(f"structures/responses/{book_id}.txt", 'r', encoding='utf-8') as file:
    #     response = file.read()
    lines = response.split('\n')
    new_lines = [line for line in lines if line.startswith('#')]
    structure = get_structure(new_lines)
    save_json(structure, f'structures/{book_id}.json')

import sys; sys.exit(0)

import re

chapter_patterns = {}
chapter_patterns["B00"] = re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*')
# chapter_patterns["B02"] = re.compile(r'^[A-Z]+$')
# chapter_patterns["B05"] = re.compile(r'^[IVXLCDM]+\.$')
# chapter_patterns["B07"] = re.compile(r'^CHAPTER\s+[IVXLCDM]+\.$')
# chapter_patterns["B10"] = re.compile(r'^CHAPTER\s+[1-9][0-9]*\.$')
# chapter_patterns["B11"] = re.compile(r'^CHAPTER\s+[IVXLCDM]+$')
# chapter_patterns["B13"] = re.compile(r'^CHAPTER\s+[1-9][0-9]*$')
# chapter_patterns["B14"] = re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*')
# chapter_patterns["B17"] = re.compile(r'^Chapter\s+[A-Z][a-z]+\s*$')
# chapter_patterns["B18"] = re.compile(r'^[IVXLCDM]+$')

def split_chapters(file_path, nested_pattern):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()

    chapters = []  # 存储所有章节内容
    current_chapter = None  # 当前正在处理的章节

    for line in lines:
        # 检查当前行是否为章节行
        if nested_pattern.match(line):
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

# 示例使用
if __name__ == "__main__":
    for book_id in chapter_patterns:
        chapters = split_chapters(f"data/NovelQA/Books/PublicDomain/{book_id}.txt", chapter_patterns[book_id])  # 替换为你的文件路径
        for idx, chapter in enumerate(chapters, 1):
            print(f"章节 {idx} 的内容：")
            print(chapter[:200])  # 打印前200字符以避免过长输出
            print("-" * 50)
    # chapters = split_chapters("data/NovelQA/Books/PublicDomain/B00.txt")  # 替换为你的文件路径
    # for idx, chapter in enumerate(chapters, 1):
    #     print(f"章节 {idx} 的内容：")
    #     print(chapter[:200])  # 打印前200字符以避免过长输出
    #     print("-" * 50)
