import re
from path_builder import NovelQAPathBuilder
from loader import BookLoader
from utils import save_json
import unicodedata

class Chapterizer:
    def __init__(self, book_content: str, book_title: str, chapter_patterns: list[re.Pattern]):
        self.book_content = book_content
        self.chapter_patterns = chapter_patterns
        self.chapter_levels = {book_title: 1}
        self.chapter_titles = [book_title]
        self.structure = {}
        self._get_chapter_titles()

    @staticmethod
    def _remove_invisible_chars(s: str) -> str:
        return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

    def _get_chapter_titles(self):
        # 读取文件内容
        lines = self.book_content.split('\n')
        lines = [self._remove_invisible_chars(line) for line in lines]

        prev_idx = None

        stk: list[int] = [1]
        structure_stk: list[dict] = [self.structure]
        current_content = ""
        # FIXME 这里规定最大的章节级别为 10，但可能在个别的书目会导致问题
        pattern_level_dict = list(map(lambda _: None, self.chapter_patterns))

        # current_chapter = None  # 当前正在处理的章节

        for line in lines:
            matched = False
            for idx, pattern in enumerate(self.chapter_patterns):
                if pattern.match(line):
                    matched = True
                    prev_idx = stk[-1]
                    if pattern_level_dict[idx] is not None:
                        assert abs(pattern_level_dict[idx] - prev_idx) <= 1
                    else:
                        pattern_level_dict[idx] = stk[-1] + 1
                    while pattern_level_dict[idx] <= stk[-1]:
                        stk.pop()
                    stk.append(pattern_level_dict[idx])
                    self.chapter_levels[line] = stk[-1]
                    break
            if matched:
                self.chapter_levels[line] = stk[-1]
                self.chapter_titles.append(line)
            # # 检查当前行是否为章节行
            # if matched:
            #     # 如果当前已经有一个章节在收集，则保存它
            #     if current_chapter is not None:
            #         self.chapters.append(''.join(current_chapter))
            #     # 开始新的章节
            #     current_chapter = [line]
            # else:
            #     # 如果当前处于章节中，则添加该行到当前章节
            #     if current_chapter is not None:
            #         current_chapter.append(line)
        
        # 处理最后一个章节
        # if current_chapter is not None:
        #     self.chapters.append(''.join(current_chapter))
        
        # return chapters
    
    def to_markdwon(self):
        markdown = ""
        for title in self.chapter_titles:
            level = self.chapter_levels[title]
            markdown += "#" * level + " " + title + "\n"
        return markdown
    
from path_builder import NovelQAPathBuilder
from loader import BookLoader, BookMetaDataLoader
from utils import save_json
path_builder = NovelQAPathBuilder('../data/NovelQA')
book_loader = BookLoader(path_builder.get_book_path('B00'), 'B00')
book_loader.load()
book_content = book_loader.get_content()
book_meta_data_loader = BookMetaDataLoader('../data/NovelQA/bookmeta.json')
book_meta_data_loader.load()
book_title = book_meta_data_loader.get_title('B00')

chapterizer = Chapterizer(book_content, book_title, [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^PART\s+[IVXLCDM]+\.\s+.*')])
markdown = chapterizer.to_markdwon()
print(markdown)

# for book_id in chapter_patterns:
#     book_path = path_builder.get_book_path(book_id)
#     chapters = split_chapters(book_path, chapter_patterns[book_id])  # 替换为你的文件路径
#     book_loader = BookLoader(book_path, book_id)
#     book_loader.load()
#     book_content = book_loader.get_content()
#     lines = book_content.split('\n')
#     lines = [remove_invisible_chars(line) for line in lines]
#     while len(lines) > 0 and not lines[0].strip():
#         lines.pop(0)
#     with open(f"./structures/responses/{book_id}.txt", 'w', encoding='utf-8') as f:
#         f.write(lines[0] + '\n')
#     with open(f"./structures/responses/{book_id}.txt", 'a', encoding='utf-8') as f:
#         for chapter in chapters:
#             f.write(chapter.split('\n')[0] + '\n')
#             # print(f"章节 {idx} 的内容：")
#             # print(chapter[:200])  # 打印前200字符以避免过长输出
#             # print("-" * 50)
#     # chapters = split_chapters("data/NovelQA/Books/PublicDomain/B00.txt")  # 替换为你的文件路径
#     # for idx, chapter in enumerate(chapters, 1):
#     #     print(f"章节 {idx} 的内容：")
#     #     print(chapter[:200])  # 打印前200字符以避免过长输出
#     #     print("-" * 50)


# # from src.llm import get_llm

# # def build_split_prompt(book_content):
# #     return f"""Read the text below and output **only** the chapter and sub‑chapter headings exactly as they appear. Do **not** output any prose from the text itself. Pay attention to possible nesting.\nGiven text:\n{book_content}\nYou should output in the following format:\n# Novel title\n## Subsection 1 heading\n### Subsection 1 of subsection 1 heading\n### Subsection 1 of subsection 1 heading\n## Subsection 2 heading\n### Subsection 2 of subsection 1 heading\n### Subsection 2 of subsection 1 heading\nDon't output anything else, just output the result."""

# # llm = get_llm('gemini')
