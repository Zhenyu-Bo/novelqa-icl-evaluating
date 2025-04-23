from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader
from src.utils import save_json
import unicodedata

path_builder = NovelQAPathBuilder('./data/NovelQA')

def remove_invisible_chars(s):
    return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

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

def fill_content(structure: dict, book_content: str) -> dict:
    if len(structure['structures']) == 0:
        structure.pop('structures')
        structure['content'] = book_content
        return structure
    lines = book_content.split('\n')
    lines = [remove_invisible_chars(line) for line in lines]
    while len(lines) > 0 and not lines[0].strip():
        lines.pop(0)
    assert lines[0].startswith(structure['title']), f"{lines[0]}\n{structure['title']}"
    lines.pop(0)
    current_structure_idx = 0
    structures = structure['structures']
    structure_contents = [structure['title']]
    while len(lines) > 0:
        while len(lines) > 0 and (current_structure_idx == len(structure['structures']) or lines[0].strip().lower() != structures[current_structure_idx]['title'].strip().lower()):
            structure_contents[-1] += '\n' + lines.pop(0)
        if len(lines):
            structure_contents.append(lines.pop(0))
        current_structure_idx += 1
    assert current_structure_idx == len(structures) + 1, f"{current_structure_idx} {len(structures)}, {structures}"
    structure['content'] = structure_contents[0]
    for i in range(1, len(structure_contents)):
        structure['structures'][i - 1] = fill_content(structure['structures'][i - 1], structure_contents[i])
    return structure

import os

for book_id in BOOK_IDS:
    # break
    # if not book_id == 'B02':
    #     continue
    book_path = path_builder.get_book_path(book_id)
    book_loader = BookLoader(book_path, book_id)
    book_loader.load()
    book_content = book_loader.get_content()
    # prompt = build_split_prompt(book_content)
    # with open(f"structures/prompts/{book_id}.txt", 'w', encoding='utf-8') as file:
    #     file.write(prompt)
    # continue
    # response = llm.generate(prompt)
    # with open(f"structures/responses/{book_id}.txt", 'w', encoding='utf-8') as file:
    #     file.write(response)
    if not os.path.exists(f"structures/responses/{book_id}.txt"):
        continue
    with open(f"structures/responses/{book_id}.txt", 'r', encoding='utf-8') as file:
        response = file.read()
    lines = response.split('\n')
    new_lines = [line for line in lines if line.startswith('#')]
    structure = get_structure(new_lines)
    # structure = fill_content(structure, book_content.replace('\ufeff', ''))
    save_json(structure, f'structures/{book_id}.json')

import sys; sys.exit(0)
