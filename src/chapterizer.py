from functools import cached_property
import re
from .utils import save_json
import unicodedata
import os

class Chapterizer:
    def __init__(self, book_content: str, book_title: str, chapter_patterns: list[re.Pattern]):
        self.book_content = book_content
        self.chapter_patterns = chapter_patterns
        self.chapter_levels = {book_title: 1}
        self.chapter_titles = [book_title]
        self.structure = {'title': book_title, 'structures': [], 'content': ''}
        self._chapterize()

    @staticmethod
    def _remove_invisible_chars(s: str) -> str:
        return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

    def _chapterize(self):
        lines = self.book_content.split('\n')
        lines = [self._remove_invisible_chars(line) for line in lines]

        stk: list[int] = [1]
        structure_stk: list[dict] = [self.structure]

        pattern_level_dict = list(map(lambda _: None, self.chapter_patterns))

        for line in lines:
            for idx, pattern in enumerate(self.chapter_patterns):
                if pattern.match(line):
                    if pattern_level_dict[idx] is None:
                        pattern_level_dict[idx] = stk[-1] + 1
                    while pattern_level_dict[idx] <= stk[-1]:
                        pop_structure = structure_stk.pop()
                        structure_stk[-1]["structures"].append(pop_structure)
                        stk.pop()
                    stk.append(pattern_level_dict[idx])
                    structure_stk.append({'title': line.strip(), 'structures': [], 'content': ''})
                    self.chapter_levels[line] = stk[-1]
                    self.chapter_titles.append(line)
                    break
            structure_stk[-1]["content"] += line + '\n'
        while len(structure_stk) > 1:
            pop_structure = structure_stk.pop()
            structure_stk[-1]["structures"].append(pop_structure)

    def structure_from_markdown(self, markdown: str):
        lines = markdown.split('\n')
        self.structure = self._structrue_from_markdown_recursive(lines, 1)
        self.structure = self._fill_content_recursive(self.structure, self.book_content)
        self._chapter_from_structure(self.structure, 1)

    @staticmethod
    def _structrue_from_markdown_recursive(lines: str, level: int) -> dict:
        structure = {}
        current_line = lines.pop(0)
        current_level = current_line.count('#')
        next_structures = []
        while len(lines) > 0 and lines[0].count('#') > current_level:
            next_structures.append(Chapterizer._structrue_from_markdown_recursive(lines, current_level))
        structure['title'] = current_line.strip('#').strip()
        structure['structures'] = next_structures
        result = structure
        for _ in range(level - current_level - 1):
            result = {'title': '', 'structures': [result]}
        return structure
    
    @staticmethod
    def _fill_content_recursive(structure: dict, book_content: str) -> dict:
        if len(structure['structures']) == 0:
            # structure.pop('structures')
            structure['content'] = book_content
            return structure
        lines = book_content.split('\n')
        lines = [Chapterizer._remove_invisible_chars(line) for line in lines]
        while len(lines) > 0 and not lines[0].strip():
            lines.pop(0)
        assert lines[0].lower().startswith(structure['title'].lower()), f"{lines[0]}\n{structure['title']}"
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
            structure['structures'][i - 1] = Chapterizer._fill_content_recursive(structure['structures'][i - 1], structure_contents[i])
        return structure

    def _chapter_from_structure(self, structure: dict, level: int) -> dict:
        self.chapter_levels[structure['title']] = level
        self.chapter_titles.append(structure['title'])
        for structure in structure['structures']:
            self._chapter_from_structure(structure, level + 1)

    def get_structure(self):
        return self.structure
    
    def save_structure(self, path: str, content: bool = False):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if not content:
            new_structure = self.get_no_content(self.structure)
            save_json(new_structure, path)
            return
        save_json(self.structure, path)

    @staticmethod
    def get_no_content(structure: dict) -> dict:
        new_structure = dict(structure)
        new_structure.pop('content')
        for i in range(len(new_structure['structures'])):
            new_structure['structures'][i] = Chapterizer.get_no_content(new_structure['structures'][i])
        return new_structure

    def get_chapter_contents(self, level: int = 0):
        if level <= 0 and level > -self.structure_depth:
            level += self.structure_depth
        pass

    @cached_property
    def structure_depth(self):
        return max(self.chapter_levels.values())
    
    def to_markdwon(self):
        markdown = ""
        for title in self.chapter_titles:
            level = self.chapter_levels[title]
            markdown += "#" * level + " " + title + "\n"
        return markdown
