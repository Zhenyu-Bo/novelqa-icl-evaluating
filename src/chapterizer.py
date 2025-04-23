from functools import cached_property
import re
from utils import save_json
import unicodedata

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

        prev_idx = None

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

    def get_structure(self):
        return self.structure
    
    def save_structure(self, path: str):
        save_json(self.structure, path)

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
