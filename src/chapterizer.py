"""章节化器，将小说内容划分为章节"""

from functools import cached_property
import re
from .utils import save_json
import unicodedata
import os
import copy

class Chapterizer:
    """章节化器，将小说内容划分为章节"""

    CHAPTER_PATTERNS: list[re.Pattern] = [
        re.compile(r'^CHAPTER\s+\b([IVXLC]+|[1-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|' \
          r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|' \
          r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|' \
          r'twenty-one|thirty-one|forty-one|fifty-one|sixty-one|seventy-one|eighty-one|ninety-one|' \
          r'twenty-two|thirty-two|forty-two|fifty-two|sixty-two|seventy-two|eighty-two|ninety-two|' \
          r'twenty-three|thirty-three|forty-three|fifty-three|sixty-three|seventy-three|eighty-three|ninety-three|' \
          r'twenty-four|thirty-four|forty-four|fifty-four|sixty-four|seventy-four|eighty-four|ninety-four|' \
          r'twenty-five|thirty-five|forty-five|fifty-five|sixty-five|seventy-five|eighty-five|ninety-five|' \
          r'twenty-six|thirty-six|forty-six|fifty-six|sixty-six|seventy-six|eighty-six|ninety-six|' \
          r'twenty-seven|thirty-seven|forty-seven|fifty-seven|sixty-seven|seventy-seven|eighty-seven|ninety-seven|' \
          r'twenty-eight|thirty-eight|forty-eight|fifty-eight|sixty-eight|seventy-eight|eighty-eight|ninety-eight|' \
          r'twenty-nine|thirty-nine|forty-nine|fifty-nine|sixty-nine|seventy-nine|eighty-nine|ninety-nine|' \
          r'((the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|' \
          r'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|' \
          r'twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|' \
          r'twenty-first|thirty-first|forty-first|fifty-first|sixty-first|seventy-first|eighty-first|ninety-first|' \
          r'twenty-second|thirty-second|forty-second|fifty-second|sixty-second|seventy-second|eighty-second|ninety-second|' \
          r'twenty-third|thirty-third|forty-third|fifty-third|sixty-third|seventy-third|eighty-third|ninety-third|' \
          r'twenty-fourth|thirty-fourth|forty-fourth|fifty-fourth|sixty-fourth|seventy-fourth|eighty-fourth|ninety-fourth|' \
          r'twenty-fifth|thirty-fifth|forty-fifth|fifty-fifth|sixty-fifth|seventy-fifth|eighty-fifth|ninety-fifth|' \
          r'twenty-sixth|thirty-sixth|forty-sixth|fifty-sixth|sixty-sixth|seventy-sixth|eighty-sixth|ninety-sixth|' \
          r'twenty-seventh|thirty-seventh|forty-seventh|fifty-seventh|sixty-seventh|seventy-seventh|eighty-seventh|ninety-seventh|' \
          r'twenty-eighth|thirty-eighth|forty-eighth|fifty-eighth|sixty-eighth|seventy-eighth|eighty-eighth|ninety-eighth|' \
          r'twenty-ninth|thirty-ninth|forty-ninth|fifty-ninth|sixty-ninth|seventy-ninth|eighty-ninth|ninety-ninth))|' \
          r')\b(\.?((\s|—|-)+.{0,50})?)?$', re.IGNORECASE),
        re.compile(r'^PART\s+\b([IVXLC]+|[1-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|' \
          r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|' \
          r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|' \
          r'twenty-one|thirty-one|forty-one|fifty-one|sixty-one|seventy-one|eighty-one|ninety-one|' \
          r'twenty-two|thirty-two|forty-two|fifty-two|sixty-two|seventy-two|eighty-two|ninety-two|' \
          r'twenty-three|thirty-three|forty-three|fifty-three|sixty-three|seventy-three|eighty-three|ninety-three|' \
          r'twenty-four|thirty-four|forty-four|fifty-four|sixty-four|seventy-four|eighty-four|ninety-four|' \
          r'twenty-five|thirty-five|forty-five|fifty-five|sixty-five|seventy-five|eighty-five|ninety-five|' \
          r'twenty-six|thirty-six|forty-six|fifty-six|sixty-six|seventy-six|eighty-six|ninety-six|' \
          r'twenty-seven|thirty-seven|forty-seven|fifty-seven|sixty-seven|seventy-seven|eighty-seven|ninety-seven|' \
          r'twenty-eight|thirty-eight|forty-eight|fifty-eight|sixty-eight|seventy-eight|eighty-eight|ninety-eight|' \
          r'twenty-nine|thirty-nine|forty-nine|fifty-nine|sixty-nine|seventy-nine|eighty-nine|ninety-nine|' \
          r'((the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|' \
          r'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|' \
          r'twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|' \
          r'twenty-first|thirty-first|forty-first|fifty-first|sixty-first|seventy-first|eighty-first|ninety-first|' \
          r'twenty-second|thirty-second|forty-second|fifty-second|sixty-second|seventy-second|eighty-second|ninety-second|' \
          r'twenty-third|thirty-third|forty-third|fifty-third|sixty-third|seventy-third|eighty-third|ninety-third|' \
          r'twenty-fourth|thirty-fourth|forty-fourth|fifty-fourth|sixty-fourth|seventy-fourth|eighty-fourth|ninety-fourth|' \
          r'twenty-fifth|thirty-fifth|forty-fifth|fifty-fifth|sixty-fifth|seventy-fifth|eighty-fifth|ninety-fifth|' \
          r'twenty-sixth|thirty-sixth|forty-sixth|fifty-sixth|sixty-sixth|seventy-sixth|eighty-sixth|ninety-sixth|' \
          r'twenty-seventh|thirty-seventh|forty-seventh|fifty-seventh|sixty-seventh|seventy-seventh|eighty-seventh|ninety-seventh|' \
          r'twenty-eighth|thirty-eighth|forty-eighth|fifty-eighth|sixty-eighth|seventy-eighth|eighty-eighth|ninety-eighth|' \
          r'twenty-ninth|thirty-ninth|forty-ninth|fifty-ninth|sixty-ninth|seventy-ninth|eighty-ninth|ninety-ninth))|' \
          r')\b(\.?((\s|—|-)+.{0,50})?)?$', re.IGNORECASE),
        re.compile(r'^VOLUME\s+\b([IVXLC]+|[1-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|' \
          r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|' \
          r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|' \
          r'twenty-one|thirty-one|forty-one|fifty-one|sixty-one|seventy-one|eighty-one|ninety-one|' \
          r'twenty-two|thirty-two|forty-two|fifty-two|sixty-two|seventy-two|eighty-two|ninety-two|' \
          r'twenty-three|thirty-three|forty-three|fifty-three|sixty-three|seventy-three|eighty-three|ninety-three|' \
          r'twenty-four|thirty-four|forty-four|fifty-four|sixty-four|seventy-four|eighty-four|ninety-four|' \
          r'twenty-five|thirty-five|forty-five|fifty-five|sixty-five|seventy-five|eighty-five|ninety-five|' \
          r'twenty-six|thirty-six|forty-six|fifty-six|sixty-six|seventy-six|eighty-six|ninety-six|' \
          r'twenty-seven|thirty-seven|forty-seven|fifty-seven|sixty-seven|seventy-seven|eighty-seven|ninety-seven|' \
          r'twenty-eight|thirty-eight|forty-eight|fifty-eight|sixty-eight|seventy-eight|eighty-eight|ninety-eight|' \
          r'twenty-nine|thirty-nine|forty-nine|fifty-nine|sixty-nine|seventy-nine|eighty-nine|ninety-nine|' \
          r'((the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|' \
          r'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|' \
          r'twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|' \
          r'twenty-first|thirty-first|forty-first|fifty-first|sixty-first|seventy-first|eighty-first|ninety-first|' \
          r'twenty-second|thirty-second|forty-second|fifty-second|sixty-second|seventy-second|eighty-second|ninety-second|' \
          r'twenty-third|thirty-third|forty-third|fifty-third|sixty-third|seventy-third|eighty-third|ninety-third|' \
          r'twenty-fourth|thirty-fourth|forty-fourth|fifty-fourth|sixty-fourth|seventy-fourth|eighty-fourth|ninety-fourth|' \
          r'twenty-fifth|thirty-fifth|forty-fifth|fifty-fifth|sixty-fifth|seventy-fifth|eighty-fifth|ninety-fifth|' \
          r'twenty-sixth|thirty-sixth|forty-sixth|fifty-sixth|sixty-sixth|seventy-sixth|eighty-sixth|ninety-sixth|' \
          r'twenty-seventh|thirty-seventh|forty-seventh|fifty-seventh|sixty-seventh|seventy-seventh|eighty-seventh|ninety-seventh|' \
          r'twenty-eighth|thirty-eighth|forty-eighth|fifty-eighth|sixty-eighth|seventy-eighth|eighty-eighth|ninety-eighth|' \
          r'twenty-ninth|thirty-ninth|forty-ninth|fifty-ninth|sixty-ninth|seventy-ninth|eighty-ninth|ninety-ninth))|' \
          r')\b(\.?((\s|—|-)+.{0,50})?)?$', re.IGNORECASE),
        # re.compile(r'^SECTION\s+\b([IVXLC]+|[1-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|' \
        #   r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|' \
        #   r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|' \
        #   r'twenty-one|thirty-one|forty-one|fifty-one|sixty-one|seventy-one|eighty-one|ninety-one|' \
        #   r'twenty-two|thirty-two|forty-two|fifty-two|sixty-two|seventy-two|eighty-two|ninety-two|' \
        #   r'twenty-three|thirty-three|forty-three|fifty-three|sixty-three|seventy-three|eighty-three|ninety-three|' \
        #   r'twenty-four|thirty-four|forty-four|fifty-four|sixty-four|seventy-four|eighty-four|ninety-four|' \
        #   r'twenty-five|thirty-five|forty-five|fifty-five|sixty-five|seventy-five|eighty-five|ninety-five|' \
        #   r'twenty-six|thirty-six|forty-six|fifty-six|sixty-six|seventy-six|eighty-six|ninety-six|' \
        #   r'twenty-seven|thirty-seven|forty-seven|fifty-seven|sixty-seven|seventy-seven|eighty-seven|ninety-seven|' \
        #   r'twenty-eight|thirty-eight|forty-eight|fifty-eight|sixty-eight|seventy-eight|eighty-eight|ninety-eight|' \
        #   r'twenty-nine|thirty-nine|forty-nine|fifty-nine|sixty-nine|seventy-nine|eighty-nine|ninety-nine)\b(\.?(\s+.{0,80})?)?$', re.IGNORECASE),
        re.compile(r'^\b([IVXLC]+)\b(\.?(\.\s+.{0,50})?)?$'),
        re.compile(r'^BOOK\s+\b([IVXLC]+|[1-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|' \
          r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|' \
          r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|' \
          r'twenty-one|thirty-one|forty-one|fifty-one|sixty-one|seventy-one|eighty-one|ninety-one|' \
          r'twenty-two|thirty-two|forty-two|fifty-two|sixty-two|seventy-two|eighty-two|ninety-two|' \
          r'twenty-three|thirty-three|forty-three|fifty-three|sixty-three|seventy-three|eighty-three|ninety-three|' \
          r'twenty-four|thirty-four|forty-four|fifty-four|sixty-four|seventy-four|eighty-four|ninety-four|' \
          r'twenty-five|thirty-five|forty-five|fifty-five|sixty-five|seventy-five|eighty-five|ninety-five|' \
          r'twenty-six|thirty-six|forty-six|fifty-six|sixty-six|seventy-six|eighty-six|ninety-six|' \
          r'twenty-seven|thirty-seven|forty-seven|fifty-seven|sixty-seven|seventy-seven|eighty-seven|ninety-seven|' \
          r'twenty-eight|thirty-eight|forty-eight|fifty-eight|sixty-eight|seventy-eight|eighty-eight|ninety-eight|' \
          r'twenty-nine|thirty-nine|forty-nine|fifty-nine|sixty-nine|seventy-nine|eighty-nine|ninety-nine|' \
          r'((the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|' \
          r'tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|' \
          r'twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|' \
          r'twenty-first|thirty-first|forty-first|fifty-first|sixty-first|seventy-first|eighty-first|ninety-first|' \
          r'twenty-second|thirty-second|forty-second|fifty-second|sixty-second|seventy-second|eighty-second|ninety-second|' \
          r'twenty-third|thirty-third|forty-third|fifty-third|sixty-third|seventy-third|eighty-third|ninety-third|' \
          r'twenty-fourth|thirty-fourth|forty-fourth|fifty-fourth|sixty-fourth|seventy-fourth|eighty-fourth|ninety-fourth|' \
          r'twenty-fifth|thirty-fifth|forty-fifth|fifty-fifth|sixty-fifth|seventy-fifth|eighty-fifth|ninety-fifth|' \
          r'twenty-sixth|thirty-sixth|forty-sixth|fifty-sixth|sixty-sixth|seventy-sixth|eighty-sixth|ninety-sixth|' \
          r'twenty-seventh|thirty-seventh|forty-seventh|fifty-seventh|sixty-seventh|seventy-seventh|eighty-seventh|ninety-seventh|' \
          r'twenty-eighth|thirty-eighth|forty-eighth|fifty-eighth|sixty-eighth|seventy-eighth|eighty-eighth|ninety-eighth|' \
          r'twenty-ninth|thirty-ninth|forty-ninth|fifty-ninth|sixty-ninth|seventy-ninth|eighty-ninth|ninety-ninth))|' \
          r')\b(\.?((\s|—|-)+.{0,50})?)?$', re.IGNORECASE),
        re.compile(r'^_[ivxlc]+_$'),
        re.compile(r'^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)$')
    ]

    IGNORE_PATTERNS: list[re.Pattern] = [
        re.compile(r'^V\.\sSACKVILLE-WEST\.$'),
        re.compile(r'^L\.\sM\.\sMontgomery\.$'),
    ]


    def __init__(self, book_content: str, book_title: str = None):
        """构造方法
        Args:
            book_content (str): 小说内容
            book_title (str): 小说标题
        """
        # 书籍内容与标题
        self.book_content = Chapterizer._remove_invisible_chars(book_content)
        self.book_title = book_title    
        # 章节标题字典，键为章节标题，值为章节级别，小说标题级别为1，其后章节级别依次递增
        self.chapter_levels = {book_title: 1} if book_title is not None else {}
        # 章节标题列表，小说标题排在最前面，按出现顺序排列
        self.chapter_titles = [book_title] if book_title is not None else []
        # 章节结构字典，存储标题、内容、子章节，其中内容不包含子章节的内容
        self.structure = {'title': book_title, 'structures': [], 'content': ''} if book_title is not None else {'title': 'The Book', 'structures': [], 'content': ''}
        # 章节化
        self._chapterize()

    @staticmethod
    def _remove_invisible_chars(s: str) -> str:
        """移除不可见字符"""
        if not isinstance(s, str):
            print(f"Warning: {s} is not a string, type: {type(s)}")
            return None
        return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))
    
    @staticmethod
    def _remove_invalid_chars(s: str) -> str:
        """移除 windows 文件名非法字符以及其他不方便识别的字符"""
        if not isinstance(s, str):
            print(f"Warning: {s} is not a string, type: {type(s)}")
            return None
        s = re.sub(r'\s+', ' ', s)
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '.']
        for char in invalid_chars:
            s = s.replace(char, ' ')
        return s.strip()

    def _chapterize(self):
        """章节化"""
        # 分行并移除不可见字符
        lines = self.book_content.split('\n')
        lines = [self._remove_invisible_chars(line) for line in lines if line.strip()]
        lines = self._ignore_toc(lines)

        # 单调栈结构，存储既往章节的级别
        stk: list[int] = [1]
        # 栈结构，与stk对应，存储既往章节的结构
        structure_stk: list[dict] = [self.structure]

        # 正则表达式表达的章节的级别，我们假设不同的正则表达式表达的章节的级别是不同的，这需要我们设法去除目录
        pattern_level_dict = list(map(lambda _: None, self.CHAPTER_PATTERNS))

        # 逐行处理
        for line in lines:
            # 逐个正则表达式匹配
            for idx, pattern in enumerate(self.CHAPTER_PATTERNS):
                # 如果匹配成功
                if pattern.match(line) and not any(ignore_pattern.match(line) for ignore_pattern in self.IGNORE_PATTERNS):
                    # print(idx, prev_pattern_idx)
                    # 如果该正则表达式表达的章节的级别还未确定，设置为当前章节（栈顶）的级别+1
                    if pattern_level_dict[idx] is None:
                        pattern_level_dict[idx] = stk[-1] + 1
                    # 如果当前的章节级别小于栈顶的章节级别，弹出栈顶的章节，直到当前的章节级别大于栈顶的章节级别
                    # 举例：如果当前栈分别有
                    # 1, 2, 3
                    # 对应小说标题，第一部分和第五章（第一部分的第五章）
                    # 我们处理到了第二部分
                    # 那么我们应该弹出第五章、第一部分
                    while pattern_level_dict[idx] <= stk[-1]:
                        # 弹出栈顶的章节，并加入新的栈顶的章节的子章节
                        pop_structure = structure_stk.pop()
                        structure_stk[-1]["structures"].append(pop_structure)
                        stk.pop()
                    # 加入当前的章节
                    stk.append(pattern_level_dict[idx])
                    structure_stk.append({'title': line.strip(), 'structures': [], 'content': ''})
                    self.chapter_levels[line] = stk[-1]
                    self.chapter_titles.append(line)
                    break
            # 栈顶的章节即为正在处理的章节，将当前行加入栈顶的章节的内容
            structure_stk[-1]["content"] += line + '\n'
        # 处理到最后，栈中可能还有章节，不断弹出栈顶的章节，并加入新的栈顶的章节的子章节，直到栈中只剩下小说标题
        # 举例：
        # 最后可能是
        # 1, 2, 3
        # 对应小说标题，第三部分和第三部分第五章
        # 我们应该弹出第三部分第五章、第三部分
        while len(structure_stk) > 1:
            pop_structure = structure_stk.pop()
            structure_stk[-1]["structures"].append(pop_structure)

    @staticmethod
    def _ignore_toc(lines: list[str]) -> list[str]:
        """忽略目录，返回去除目录后的内容"""
        new_lines = []
        toc_pattern_idx = None
        is_toc = False
        prev_idx = None
        has_toc = False
        for line in lines:
            if line.strip() == '':
                # new_lines.append(line)
                continue
            matched = False
            for idx, pattern in enumerate(Chapterizer.CHAPTER_PATTERNS):
                if pattern.match(line):
                    matched = True
                    if idx == prev_idx and not is_toc:
                        has_toc = True
                        is_toc = True
                        toc_pattern_idx = idx
                        new_lines.pop()
                        prev_idx = None
                    elif is_toc and idx == toc_pattern_idx:
                        pass
                    else:
                        prev_idx = idx
                        new_lines.append(line)
            if not matched:
                is_toc = False
                toc_pattern_idx = None
                prev_idx = None
                new_lines.append(line)
        if not has_toc:
            return new_lines
        return Chapterizer._ignore_toc(new_lines)

    def structure_from_markdown(self, markdown: str):
        """从markdown中加载章节结构，主要是为了方便用户修改章节结构
        Args:
            markdown (str): markdown格式的章节结构
        Returns:
            dict: 章节结构字典
        """
        # 分行
        lines = markdown.split('\n')
        # 调用递归函数，从markdown中加载章节结构
        self.structure = self._structrue_from_markdown_recursive(lines, 1)
        # 调用递归函数，从章节结构中填充内容
        self.structure = self._fill_content_recursive(self.structure, self.book_content)
        # 调用递归函数，从章节结构中生成章节标题字典chapter_levels和章节标题列表chapter_titles
        self._chapter_from_structure(self.structure, 1)

    def structure_from_nocontent_structure(self, structure: dict):
        """从无内容的章节结构中加载章节结构，主要是为了方便用户修改章节结构
        Args:
            structure (dict): 无内容的章节结构
        Returns:
            dict: 章节结构字典
        """
        # 调用递归函数，从章节结构中填充内容
        self.structure = self._fill_content_recursive(structure, self.book_content)
        # 调用递归函数，从章节结构中生成章节标题字典chapter_levels和章节标题列表chapter_titles
        self._chapter_from_structure(self.structure, 1)

    @staticmethod
    def _structrue_from_markdown_recursive(lines: str, level: int) -> dict:
        """递归方法，从markdown中加载章节结构
        Args:
            lines (str): markdown格式的章节结构
            level (int): 当前章节的级别
        Returns:
            dict: 章节结构字典，不包含内容
        """
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
        """递归方法，从章节结构中填充内容
        Args:
            structure (dict): 章节结构字典，不包含内容
            book_content (str): 小说内容
        Returns:
            dict: 章节结构字典，包含内容
        """
        if len(structure['structures']) == 0:
            structure['content'] = book_content
            return structure
        lines = book_content.split('\n')
        lines = [Chapterizer._remove_invisible_chars(line) for line in lines]
        lines = Chapterizer._ignore_toc(lines)
        while len(lines) > 0 and not lines[0].strip():
            lines.pop(0)
        # assert lines[0].lower().startswith(structure['title'].lower()), f"{lines[0]}\n{structure['title']}"
        lines.pop(0)
        current_structure_idx = 0
        structures = structure['structures']
        structure_contents = [structure['title']]
        while len(lines) > 0:
            if current_structure_idx == len(structure['structures']):
                # 如果当前章节结构索引已经到达最后一个章节结构，则将剩余所有lines加到structure_contents[-1]中，以\n划分
                while len(lines) > 0:
                    structure_contents[-1] += '\n' + lines.pop(0)
                current_structure_idx += 1
                continue
            current_title = Chapterizer._remove_invalid_chars(structures[current_structure_idx]['title'])
            title_parts = current_title.split()
            matched_title = "" # 用于记录匹配到的标题部分
            idx = 0
            line = Chapterizer._remove_invalid_chars(lines[0].strip())
            # print(line)
            while idx < len(lines) and title_parts:
                if current_title.lower() in line.lower():
                    # 如果当前行包含完整的标题，则可以直接匹配
                    title_parts = []
                    matched_title = current_title
                    idx += 1
                    line = Chapterizer._remove_invalid_chars(lines[idx].strip()) if idx < len(lines) else ''
                    break
                if line.lower().startswith(title_parts[0].lower()):
                    # print(title_parts[0])
                    matched_title += title_parts[0] + ' '
                    line = line[len(title_parts[0]):].strip()
                    title_parts.pop(0)
                    if not line:
                        idx += 1
                        line = Chapterizer._remove_invalid_chars(lines[idx].strip()) if idx < len(lines) else ''
                else:
                    break
            if not title_parts:
                # print("Matched title:", matched_title.strip())
                # 如果标题部分全部匹配，则将当前章节标题加入结构内容
                structure_contents.append(matched_title.strip())
                lines[idx] = line
                lines = lines[idx:] if idx < len(lines) else []
                current_structure_idx += 1
            else:
                # 如果标题部分没有全部匹配，则将lines的第一行加入结构内容
                structure_contents[-1] += '\n' + lines.pop(0).strip()
        # print("Current structure index:", current_structure_idx)
        assert current_structure_idx == len(structures) + 1, f"{current_structure_idx} {len(structures)}, {structures}"
        structure['content'] = structure_contents[0]
        for i in range(1, len(structure_contents)):
            structure['structures'][i - 1] = Chapterizer._fill_content_recursive(structure['structures'][i - 1], structure_contents[i])
        return structure

    def _chapter_from_structure(self, structure: dict, level: int) -> dict:
        """递归方法，从章节结构中生成章节标题字典chapter_levels和章节标题列表chapter_titles
        Args:
            structure (dict): 章节结构字典，不包含内容
            level (int): 当前章节的级别
        Returns:
            dict: 章节结构字典，包含内容
        """
        self.chapter_levels[structure['title']] = level
        self.chapter_titles.append(structure['title'])
        for structure in structure['structures']:
            self._chapter_from_structure(structure, level + 1)

    def get_structure(self):
        """获取章节结构"""
        return self.structure
    
    def save_structure(self, path: str, content: bool = False):
        """保存章节结构
        Args:
            path (str): 保存路径
            content (bool): 是否保存内容，默认不保存
        """

        def get_no_content(structure: dict) -> dict:
            """得到不含内容的章节结构"""
            new_structure = copy.deepcopy(structure)
            new_structure.pop('content')
            for i in range(len(new_structure['structures'])):
                new_structure['structures'][i] = get_no_content(new_structure['structures'][i])
            return new_structure

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if not content:
            new_structure = get_no_content(self.structure)
            save_json(new_structure, path)
            return
        save_json(self.structure, path)

    def get_chapter_contents(self, level: int = 0) -> tuple[dict[str, str], list[str]]:
        """获取指定级别章节的内容
        Args:
            level (int): 指定级别，0表示获取所有章节的内容，1表示获取第一级章节的内容，2表示获取第二级章节的内容，以此类推
        Returns:
            tuple[dict[str, str], list[str]]: 章节内容字典，键为章节标题，多级标题以_分隔，值为章节内容；章节标题列表
        """

        if level <= 0 and level > -self._structure_depth:
            level += self._structure_depth
        assert level > 0 and level <= self._structure_depth
        result_dict = {}
        result_list = []
        def get_chapter_content(structure, current_key, current_level):
            """递归方法，获取指定级别章节的内容"""
            if current_level == level or len(structure['structures']) == 0:
                result_dict[current_key] = self._get_structure_content(structure)
                result_list.append(current_key)
                return
            for substructure in structure['structures']:
                get_chapter_content(substructure, current_key + '_' + substructure['title'], current_level + 1)
            return
        get_chapter_content(self.structure, self.chapter_titles[0], 1)
        chapter_dict = {}
        chapter_list = []
        for chapter in result_list:
            chapter_safe = Chapterizer._remove_invalid_chars(chapter)  # 清理非法字符
            chapter_dict[chapter_safe] = result_dict[chapter]  # 更新键值对
            chapter_list.append(chapter_safe)  # 更新章节标题列表
        # return result_dict, result_list
        return chapter_dict, result_list

    @staticmethod
    def _get_structure_content(structure: dict) -> str:
        """获取章节内容，包含子章节的内容"""
        result = structure['content']
        for substructure in structure['structures']:
            result += '\n' + substructure['content']
        return result

    @cached_property
    def _structure_depth(self):
        """获取章节结构的深度"""
        return max(self.chapter_levels.values())
    
    def to_markdwon(self):
        """将章节结构转换为markdown格式"""
        markdown = ""
        for title in self.chapter_titles:
            level = self.chapter_levels[title]
            markdown += "#" * level + " " + title + "\n"
        return markdown


import re
from .llm import LLM
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
class LLMChapterizer(Chapterizer):
    """章节化类, 使用LLM进行章节化"""
    
    def __init__(self, llm: LLM, book_content: str, book_title: str = None):
        self.llm = llm
        self.token_counter = tiktoken.get_encoding("cl100k_base")
        self.tokens_num = len(self.token_counter.encode(book_content))
        super().__init__(book_content, book_title)
        
    def _fill_content_recursive(self, structure: dict, book_content: str) -> dict:
        """
        从章节结构中填充内容的新实现
        Args:
            structure (dict): 章节结构字典，不包含内容
            book_content (str): 小说内容
        Returns:
            dict: 包含标题-内容映射的字典
        """
        # 提取所有章节标题和层级信息
        all_chapters = self._extract_all_chapters(structure)
        
        if not all_chapters:
            return {"Introduction": book_content}
        
        # 找到最低等级（数值最大）的章节
        max_level = min(chapter['level'] for chapter in all_chapters if chapter['level'] > 1)
        top_level_chapters = [ch for ch in all_chapters if ch['level'] == max_level]
        
        print(f"Found {len(top_level_chapters)} top-level chapters at level {max_level}")
        
        # 在原文中查找这些章节标题并切分内容
        chapter_contents = self._split_content_by_titles(book_content, top_level_chapters)
        
        return chapter_contents
    
    def _extract_all_chapters(self, structure: dict) -> list[dict]:
        """
        递归提取所有章节信息
        Args:
            structure (dict): 章节结构
        Returns:
            list[dict]: 包含标题和层级的章节列表
        """
        chapters = []
        
        def extract_recursive(struct, level):
            if struct.get('title'):
                chapters.append({
                    'title': struct['title'],
                    'level': level
                })
            
            for sub_struct in struct.get('structures', []):
                extract_recursive(sub_struct, level + 1)
        
        extract_recursive(structure, 1)
        return chapters
    
    def _split_content_by_titles(self, book_content: str, chapters: list[dict]) -> dict:
        """
        根据章节标题切分内容
        Args:
            book_content (str): 原始书籍内容
            chapters (list[dict]): 章节列表
        Returns:
            dict: 标题-内容映射字典
        """
        # 预处理文本
        lines = book_content.split('\n')
        lines = [self._remove_invisible_chars(line) for line in lines if line.strip()]
        lines = self._ignore_toc(lines)
        
        # 查找每个标题在文本中的位置
        title_positions = []
        
        for chapter in chapters:
            position = self._find_title_position(lines, chapter['title'])
            if position is not None:
                title_positions.append({
                    'title': chapter['title'],
                    'position': position,
                    'level': chapter['level']
                })
                print(f"Found title '{chapter['title']}' at line {position}")
            else:
                print(f"Warning: Title '{chapter['title']}' not found in text")
        
        # 按位置排序
        title_positions.sort(key=lambda x: x['position'])
        
        # 切分内容
        chapter_contents = {}
        
        # 处理第一个标题之前的内容
        if title_positions and title_positions[0]['position'] > 0:
            intro_content = '\n'.join(lines[:title_positions[0]['position']])
            if intro_content.strip():
                chapter_contents['Introduction'] = intro_content.strip()
        
        # 处理每个标题及其内容
        for i, title_info in enumerate(title_positions):
            title = title_info['title']
            start_pos = title_info['position']
            
            # 确定结束位置
            if i + 1 < len(title_positions):
                end_pos = title_positions[i + 1]['position']
            else:
                end_pos = len(lines)
            
            # 提取内容（包含标题行）
            content_lines = lines[start_pos:end_pos]
            content = '\n'.join(content_lines).strip()
            
            if content:
                chapter_contents[title] = content
                print(f"Filled content for '{title}': {len(content)} characters")
        
        # 如果没有找到任何标题，将所有内容归为Introduction
        if not chapter_contents:
            chapter_contents['Introduction'] = book_content.strip()
        
        return chapter_contents
    
    def _find_title_position(self, lines: list[str], target_title: str) -> int:
        """
        在文本行中查找标题的位置
        Args:
            lines (list[str]): 文本行列表
            target_title (str): 目标标题
        Returns:
            int: 标题所在行号，未找到返回None
        """
        # 清理目标标题用于匹配
        target_clean = self._remove_invalid_chars(target_title.strip())
        target_words = target_clean.lower().split()
        
        if not target_words:
            return None
        
        # 遍历每一行寻找匹配
        for line_idx, line in enumerate(lines):
            line_clean = self._remove_invalid_chars(line.strip())
            
            # 尝试完整匹配
            if target_clean.lower() in line_clean.lower():
                return line_idx
            
            # 尝试单词匹配
            if self._match_title_words(line_clean, target_words):
                return line_idx
        
        # 如果没有找到完整匹配，尝试模糊匹配
        return self._fuzzy_find_title(lines, target_title)
    
    def _match_title_words(self, line: str, target_words: list[str]) -> bool:
        """
        检查行是否包含目标标题的所有关键词
        Args:
            line (str): 待检查的行
            target_words (list[str]): 目标词列表
        Returns:
            bool: 是否匹配
        """
        line_lower = line.lower()
        matched_words = 0
        
        for word in target_words:
            if len(word) >= 2 and word in line_lower:  # 只检查长度>=2的词
                matched_words += 1
        
        # 如果匹配的重要词汇超过50%，认为找到了标题
        return matched_words >= len(target_words) * 0.6
    
    def _fuzzy_find_title(self, lines: list[str], target_title: str) -> int:
        """
        模糊查找标题位置
        Args:
            lines (list[str]): 文本行列表
            target_title (str): 目标标题
        Returns:
            int: 最佳匹配行号，未找到返回None
        """
        import difflib
        
        target_clean = self._remove_invalid_chars(target_title.strip()).lower()
        best_ratio = 0
        best_position = None
        
        for line_idx, line in enumerate(lines):
            line_clean = self._remove_invalid_chars(line.strip()).lower()
            
            # 计算相似度
            ratio = difflib.SequenceMatcher(None, target_clean, line_clean).ratio()
            
            if ratio > best_ratio and ratio > 0.6:  # 相似度阈值
                best_ratio = ratio
                best_position = line_idx
        
        if best_position is not None:
            print(f"Fuzzy matched '{target_title}' with ratio {best_ratio:.3f} at line {best_position}")
        
        return best_position
    
    def _build_structure_tree(self, chapter_list: list) -> dict:
        """
        从章节列表构建树形结构（修改版本，支持新的内容填充方式）
        """
        if not chapter_list:
            return {"title": "Unknown Book", "structures": [], "content": ""}
        
        # 找到书籍标题（level 1）
        book_title = "Unknown Book"
        for item in chapter_list:
            if item.get('level') == 1:
                book_title = self._clean_title(item.get('title', ''))
                break
        
        root = {"title": book_title, "structures": [], "content": ""}
        structure_stack = [root]
        level_stack = [1]
        
        for item in chapter_list:
            title = self._clean_title(item.get('title', ''))
            level = item.get('level', 2)
            
            if level <= 1:
                continue  # 跳过书籍标题
            
            # 调整栈，确保层级关系正确
            while len(level_stack) > 1 and level <= level_stack[-1]:
                popped = structure_stack.pop()
                level_stack.pop()
                if structure_stack:
                    structure_stack[-1]['structures'].append(popped)
            
            # 创建新的章节结构
            new_structure = {"title": title, "structures": [], "content": ""}
            structure_stack.append(new_structure)
            level_stack.append(level)
        
        # 处理剩余的栈
        while len(structure_stack) > 1:
            popped = structure_stack.pop()
            structure_stack[-1]['structures'].append(popped)
        
        return root
    
    def get_chapter_contents_from_structure(self, structure: dict = None) -> dict:
        """
        从结构中获取章节内容（新方法）
        Args:
            structure (dict): 章节结构，如果为None则使用self.structure
        Returns:
            dict: 标题-内容映射字典
        """
        if structure is None:
            structure = self.structure
        
        return self._fill_content_recursive(structure, self.book_content)
    
    def save_chapter_contents(self, output_dir: str, structure: dict = None) -> None:
        """
        保存章节内容到文件
        Args:
            output_dir (str): 输出目录
            structure (dict): 章节结构，如果为None则使用self.structure
        """
        import os
        import shutil
        
        chapter_contents = self.get_chapter_contents_from_structure(structure)
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (title, content) in enumerate(chapter_contents.items()):
            # 清理标题用作文件名
            safe_title = self._remove_invalid_chars(title)
            safe_title = safe_title[:50] if len(safe_title) > 50 else safe_title  # 限制文件名长度
            
            file_path = os.path.join(output_dir, f"{i+1:02d}_{safe_title}.txt")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{content}")
        
        print(f"已将{len(chapter_contents)}个章节保存到 {output_dir} 目录")

    # 还需要修改_chapterize方法以使用新的内容填充方式
    def _chapterize(self, chunk_size: int = 600000, chunk_overlap: int = 2000):
        """主要的章节化方法（修改版本）"""
        # 如果内容较短，直接处理
        if self.tokens_num <= chunk_size:
            structure = self.generate_chapter_structure()
        else:
            # 内容较长，分块处理
            structure = self.generate_chapter_contents_chunk_by_chunk(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            print("Chunks number:", len(structure))
        
        # 存储结构
        self.structure = structure
        
        print(json.dumps(structure, indent=2, ensure_ascii=False))
        
        # 使用新方法填充内容并生成章节信息
        chapter_contents = self.get_chapter_contents_from_structure(structure)
        
        # 重置并重新构建章节信息
        self.chapter_levels = {}
        self.chapter_titles = []
        
        # 递归遍历章节结构并生成 chapter_levels 和 chapter_titles
        def traverse_structure(structure, level):
            title = structure['title']
            self.chapter_levels[title] = level
            self.chapter_titles.append(title)
            for substructure in structure.get('structures', []):
                traverse_structure(substructure, level + 1)

        traverse_structure(structure, 1)
        
        # 为实际找到的章节内容也建立层级关系
        for i, title in enumerate(chapter_contents.keys()):
            if title not in self.chapter_levels:
                # 根据位置推断层级，Introduction为最高级，其他按顺序
                if title == "Introduction":
                    self.chapter_levels[title] = 2
                else:
                    self.chapter_levels[title] = 3  # 假设实际章节为第3级
                self.chapter_titles.append(title)
        
        print("章节标题列表:")
        print(self.chapter_titles)
        print(f"实际提取的章节内容: {list(chapter_contents.keys())}")
        
    def set_prompt(self, content: str) -> str:
        """设置让 LLM 进行章节切分的提示词"""
        prompt = f"""
        You are a professional text processing assistant. Your task is to identify the chapter structure of a novel and return it as a JSON array.

        CRITICAL REQUIREMENTS:
        1. Extract chapter titles EXACTLY as they appear in the original text
        2. Preserve ALL formatting: punctuation, capitalization, spacing, line breaks, and special characters
        3. Each title must be findable in the original text using exact string matching
        4. Do NOT modify, normalize, or clean up the titles in any way

        Requirements:
        1. Analyze the text and identify chapter titles and their hierarchical levels
        2. Level 1: Book title/main title
        3. Level 2: Part/Section titles (if any)  
        4. Level 3+: Chapter titles and sub-chapters
        5. Ignore table of contents - extract titles from the actual content
        6. If a title spans multiple lines, preserve the exact line breaks as they appear
        7. For very simple titles (single words/numbers like "I", "1", "One", etc.), connect the title with the first following sentence using "-" to make them more identifiable.
        For example, if the title is "I" and the first sentence is "The story begins", return it as "I-The story begins". If the title is "1" and the first sentence is "The sun was shining brightly", return it as "1-The sun was shining brightly". If the title is "One" and the first sentence is "The story begins", return it as "One-The story begins". This helps LLMs recognize them as titles.

        IMPORTANT: The returned titles must match the original text character-for-character, including:
        - All punctuation marks (., -, :, ;, !, ?, etc.)
        - All spacing and whitespace
        - All line breaks (\\n)
        - All capitalization
        - All special characters

        Text to analyze:
        {content}

        Return ONLY a JSON array in this format:
        [
            {{"title": "Exact title as it appears in original text", "level": 1}},
            {{"title": "Exact title as it appears in original text", "level": 2}},
            {{"title": "Exact title as it appears in original text", "level": 3}},
            ...
        ]

        Ensure the JSON is valid and properly formatted.
        """
        return prompt
        
    def set_merge_prompt(self, chunk_structures: list[list]) -> str:
        """设置让 LLM 合并章节的提示词"""
        prompt = f"""
        You are a professional text processing assistant. I will provide you with multiple chapter structure arrays extracted from different parts of a novel. Your task is to merge them into a single unified structure.

        CRITICAL REQUIREMENTS:
        1. Preserve ALL chapter titles EXACTLY as they appear in the input
        2. Do NOT modify titles in any way - keep original formatting, punctuation, spacing, line breaks
        3. Use exact string matching for duplicate detection

        Chapter structures from different parts:
        {json.dumps(chunk_structures, ensure_ascii=False, indent=2)}

        Requirements for merging:
        1. Remove duplicate chapters that appear in multiple parts (using exact title matching)
        2. Maintain the correct chronological order of chapters
        3. Ensure level consistency across the entire structure
        4. Handle overlapping content from chunk boundaries
        5. Preserve the original formatting of all titles EXACTLY
        6. The first structure usually contains the book title (level 1)

        Rules:
        - If the same chapter title appears in multiple parts, keep only the first instance
        - Maintain the sequence based on the order of appearance in the original text
        - Ensure parent-child relationships are preserved (e.g., chapters under the correct parts)
        - Remove any incomplete or truncated chapter entries from chunk boundaries
        - Do NOT change any character in the titles

        Return ONLY a JSON array in this format:
        [
            {{"title": "Exact title from input", "level": 1}},
            {{"title": "Exact title from input", "level": 2}},
            {{"title": "Exact title from input", "level": 3}},
            ...
        ]

        Ensure the final result is a valid JSON array with no duplicates and proper ordering.
        """
        return prompt
    
    def _parse_chapter_structure(self, response: str) -> dict:
        """解析 LLM 的输出"""
        try:
            response = self._remove_invisible_chars(response)
            # 提取 JSON 数组
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                raise ValueError("No valid JSON array found in LLM response")
            
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            if not isinstance(result, list) or not result:
                raise ValueError("Parsed result is not a valid list or is empty")
            
            # 构建章节结构树
            return self._build_structure_tree(result)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}")
        except Exception as e:
            raise ValueError(f"Structure parsing failed: {e}")
    
    # def _build_structure_tree(self, chapter_list: list) -> dict:
    #     """从章节列表构建树形结构"""
    #     if not chapter_list:
    #         return {"title": "Unknown Book", "structures": [], "content": ""}
        
    #     # 找到书籍标题（level 1）
    #     book_title = "Unknown Book"
    #     for item in chapter_list:
    #         if item.get('level') == 1:
    #             book_title = self._clean_title(item.get('title', ''))
    #             break
        
    #     root = {"title": book_title, "structures": [], "content": ""}
    #     structure_stack = [root]
    #     level_stack = [1]
        
    #     for item in chapter_list:
    #         title = self._clean_title(item.get('title', ''))
    #         level = item.get('level', 2)
            
    #         if level <= 1:
    #             continue  # 跳过书籍标题
            
    #         # 调整栈，确保层级关系正确
    #         while len(level_stack) > 1 and level <= level_stack[-1]:
    #             popped = structure_stack.pop()
    #             level_stack.pop()
    #             if structure_stack:
    #                 structure_stack[-1]['structures'].append(popped)
            
    #         # 创建新的章节结构
    #         new_structure = {"title": title, "structures": [], "content": ""}
    #         structure_stack.append(new_structure)
    #         level_stack.append(level)
        
    #     # 处理剩余的栈
    #     while len(structure_stack) > 1:
    #         popped = structure_stack.pop()
    #         structure_stack[-1]['structures'].append(popped)
        
    #     return root
    
    def _clean_title(self, title: str) -> str:
        """清理章节标题"""
        if not isinstance(title, str):
            return str(title)
        
        # 移除不可见字符但保留格式
        cleaned = self._remove_invalid_chars(title.strip())
        return cleaned if cleaned else "Untitled"
    
    def generate_chapter_structure(self, content: str = None) -> dict:
        """调用 LLM 生成章节结构"""
        target_content = content if content else self.book_content
        
        if not target_content or not target_content.strip():
            raise ValueError("Content is empty, cannot generate chapter structure")
        
        prompt = self.set_prompt(target_content)
        response = self.llm.generate(prompt)
        
        if not response or not response.strip():
            raise ValueError("LLM returned empty response")
        
        print("LLM response:")
        # print(response[:500] + "..." if len(response) > 500 else response)
        print(response)
        
        # 清理响应并解析
        cleaned_response = self._remove_invisible_chars(response)
        structure = self._parse_chapter_structure(cleaned_response)
        
        return structure

    def generate_chapter_contents_chunk_by_chunk(self, chunk_size: int = 800000, chunk_overlap: int = 500) -> dict:
        """使用 LLM 分块生成章节内容并合并"""
        
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 3,
            chunk_overlap=chunk_overlap * 3,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(self.book_content)
        print(f"Split content into {len(chunks)} chunks")
        
        chunk_structures = []
        
        # 处理每个块
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            if not chunk.strip():
                continue
                
            try:
                # 为每个块生成章节结构
                prompt = self.set_prompt(chunk)
                response = self.llm.generate(prompt)
                
                if not response or not response.strip():
                    print(f"Warning: Empty response for chunk {i+1}")
                    continue
                
                response = self._remove_invisible_chars(response)
                
                # 提取 JSON
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    chunk_structure = json.loads(json_match.group(0))
                    if chunk_structure:  # 只添加非空结构
                        chunk_structures.append(chunk_structure)
                        print(f"Extracted {len(chunk_structure)} chapters from chunk {i+1}")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                continue
        
        if not chunk_structures:
            raise ValueError("No valid chapter structures extracted from any chunk")
        
        # 合并所有块的结构
        merged_structure = self._merge_chunk_structures(chunk_structures)
        
        return merged_structure
    
    def _merge_chunk_structures(self, chunk_structures: list[list]) -> dict:
        """合并多个块的章节结构"""
        if not chunk_structures:
            return {"title": "Unknown Book", "structures": [], "content": ""}
        
        if len(chunk_structures) == 1:
            return self._build_structure_tree(chunk_structures[0])
        
        # 使用 LLM 进行智能合并
        try:
            merge_prompt = self.set_merge_prompt(chunk_structures)
            response = self.llm.generate(merge_prompt)
            
            if not response or not response.strip():
                print("Warning: LLM merge failed, using fallback method")
                return self._fallback_merge(chunk_structures)
            
            print("LLM merge response:")
            # print(response[:500] + "..." if len(response) > 500 else response)
            print(response)
            
            # 解析合并结果
            cleaned_response = self._remove_invisible_chars(response)
            return self._parse_chapter_structure(cleaned_response)
            
        except Exception as e:
            print(f"LLM merge failed: {e}, using fallback method")
            return self._fallback_merge(chunk_structures)
    
    def _fallback_merge(self, chunk_structures: list[list]) -> dict:
        """备用合并方法：简单去重和排序"""
        all_chapters = []
        seen_titles = set()
        
        # 收集所有唯一的章节
        for structure in chunk_structures:
            for chapter in structure:
                title = chapter.get('title', '').strip()
                level = chapter.get('level', 2)
                
                # 使用标题和级别的组合作为唯一标识
                identifier = f"{title}|{level}"
                
                if identifier not in seen_titles and title:
                    seen_titles.add(identifier)
                    all_chapters.append(chapter)
        
        # 按级别排序，然后按出现顺序
        all_chapters.sort(key=lambda x: (x.get('level', 999), chunk_structures[0].index(x) if x in chunk_structures[0] else 999))
        
        return self._build_structure_tree(all_chapters)
    

import tiktoken
class LLMSplitter():
    """使用 LLM 将书本内容按语义切分为 chunks """
    def __init__(self, llm: LLM, book_content: str, chunk_tokens=50000, max_llm_tokens=600000, chunk_overlap=0):
        self.llm = llm
        # 将所有连续的空白字符（包括\n, \r, \t, 空格等）替换为单个空格，并去除首尾空格
        self.book_content = LLMSplitter._normalize_text(book_content)
        self.chunk_tokens = chunk_tokens
        self.max_llm_tokens = max_llm_tokens
        self.chunk_overlap = chunk_overlap # 分块重叠部分的 token 数量
        self.chunks = []
        self.token_counter = tiktoken.get_encoding("cl100k_base")
        self.tokens_num = len(self.token_counter.encode(self.book_content)) # 使用处理后的 book_content 计算 tokens
        print(f"Book content token count (after normalization): {self.tokens_num}")
        self.initial_chunks = []
        if self.tokens_num + 1000 > max_llm_tokens:
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_llm_tokens * 4,  # 留出一些空间给提示词和回复
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
            self.initial_chunks = text_splitter.split_text(book_content)
            self.initial_chunks = [LLMSplitter._normalize_text(chunk) for chunk in self.initial_chunks if chunk.strip()]
            print(f"Initial chunks number: {len(self.initial_chunks)}")
            
    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        # 移除不可见字符
        text = ''.join(c for c in text if unicodedata.category(c) not in ('Cc', 'Cf') or c in [' ', '\t', '\n'])
        # 标准化 Unicode 字符
        text = unicodedata.normalize('NFKC', text)
        # 替换连续的空白字符为单个空格，并去除首尾空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def _remove_invisible_chars(s: str) -> str:
        """移除不可见字符"""
        if not isinstance(s, str):
            print(f"Warning: {s} is not a string, type: {type(s)}")
            return None
        return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))
    
    def set_prompt_directly(self, content: str) -> str:
        """设置让 LLM 直接输出分块内容的提示词"""
        prompt = f"""
        You are a professional text processing assistant. Please divide the following text into multiple chunks based on semantic coherence.

        Requirements:
        1. Each chunk should be semantically coherent, representing a complete part of the story or content.
        2. The token count of each chunk must not exceed {self.chunk_tokens} tokens. However, if necessary to maintain semantic coherence, the token count may slightly exceed this limit.
        3. Prefer splitting at chapter boundaries or paragraph boundaries.
        4. Ensure each chunk is semantically complete and does not split in the middle of a sentence.
        5. The length of each chunk should be relatively balanced, but semantic coherence is more important.
        
        Please return the chunked results in the following JSON format:
        {{
            "chunks": [
                "Content of the first chunk",
                "Content of the second chunk",
                ...
            ]
        }}

        Here is the text to be divided:

        {content}
        """
        return prompt
    
    def set_prompt_boundaries(self, content: str) -> str:
        """设置让 LLM 输出分块边界的提示词"""
        # 'content' 参数是 self.book_content 的一部分，已经经过了空白标准化处理
        prompt = f"""
        You are a professional text processing assistant. Your task is to identify semantic chunk boundaries in the following text.
        The provided text has ALREADY BEEN PROCESSED: all newline characters and other forms of whitespace have been converted into single spaces, and any leading/trailing whitespace has been removed.

        Requirements:
        1. Each chunk should be semantically coherent, representing a complete part of the story or content.
        2. The token count of each chunk (based on the provided, processed text) must not exceed {self.chunk_tokens} tokens. Slight deviations are acceptable if necessary to maintain semantic coherence.
        3. IMPORTANT: Mark chunk boundaries at the BEGINNING of new semantic sections, not at the end of previous sections.
        4. Prefer using chapter titles, section headings, or other natural starting points as boundaries.
        5. Chunk boundaries MUST be EXACT VERBATIM SUBSTRINGS of the provided 'Text to be divided' below.
        6. The length of each chunk should be relatively balanced, but semantic coherence is more important.

        CRITICALLY IMPORTANT - Adhere to these rules for boundary text:
        - The boundary text you return MUST be an EXACT character-for-character match from the 'Text to be divided'.
        - Do NOT add, remove, or change ANY characters, including punctuation or spacing, from how it appears in the provided text.
        - Do NOT attempt to reconstruct or infer original formatting (like newlines) that is NOT present in the input text.
        - Ensure the boundary string itself does not have any leading or trailing whitespace beyond what is part of the exact match in the provided text.
        - Always favor chapter titles, section headings, or paragraph beginnings as boundaries.

        Please return only the chunk boundaries (the BEGINNING sentence or phrase of each chunk) in the following JSON format:
        {{
            "boundaries": [
                "Exact verbatim BEGINNING phrase of the first chunk, copied precisely from the provided text",
                "Exact verbatim BEGINNING phrase of the second chunk, copied precisely from the provided text",
                ...
            ]
        }}

        Here is the text to be divided:

        {content}
        
        **If you cannot identify any suitable boundaries, return an empty JSON array: `{{ "boundaries": [] }}`**
        """
        return prompt
    
    def generate_chunks_directly(self) -> list[str]:
        """让 LLM 直接输出分块内容，并返回结果"""
        # 如果文本超过 LLM 处理上限，先进行分块
        if self.tokens_num > self.max_llm_tokens:
            return self._process_long_document_directly()
        
        # 文档长度在 LLM 处理范围内，直接处理
        prompt = self.set_prompt_directly(self.book_content)
        response = self.llm.generate(prompt)
        
        if not response:
            raise ValueError("LLM 返回空响应，请检查配置或输入内容")
        
        response = self._remove_invisible_chars(response)
        
        try:
            # 尝试从响应中提取 JSON
            import re
            import json
            
            # 查找 JSON 对象
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                raise ValueError("未能在LLM响应中找到有效的JSON对象")
            
            json_str = match.group(0)
            result = json.loads(json_str)
            
            if not isinstance(result, dict) or "chunks" not in result:
                raise ValueError("LLM响应中的JSON格式不正确，未找到'chunks'字段")
            
            self.chunks = result["chunks"]
            return self.chunks
            
        except json.JSONDecodeError as e:
            raise ValueError(f"解析LLM响应失败: {e}")
    
    def generate_chunks_by_boundaries(self) -> list[str]:
        """让 LLM 输出分块边界，然后在原文中切分，返回分块结果"""
        # 如果文本超过 LLM 处理上限，先进行分块
        if self.tokens_num > self.max_llm_tokens:
            return self._process_long_document_by_boundaries()
        
        # 文档长度在 LLM 处理范围内，直接处理
        prompt = self.set_prompt_boundaries(self.book_content)
        
        # 增加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(prompt)
                print("LLM response:")
                print(response)
                
                if not response:
                    print(f"LLM 返回空响应，第 {attempt + 1} 次重试")
                    continue  # 重试
                
                response = self._remove_invisible_chars(response)
                
                try:
                    # 尝试从响应中提取 JSON
                    # 查找 JSON 对象
                    match = re.search(r'\{[\s\S]*\}', response)
                    if not match:
                        raise ValueError("未能在LLM响应中找到有效的JSON对象")
                    
                    json_str = match.group(0)
                    result = json.loads(json_str)
                    
                    if not isinstance(result, dict) or "boundaries" not in result:
                        raise ValueError("LLM响应中的JSON格式不正确，未找到'boundaries'字段")
                    
                    # 根据边界在原文中切分
                    boundaries = result["boundaries"]
                    return self._split_by_boundaries(self.book_content, boundaries)
                    
                except json.JSONDecodeError as e:
                    raise ValueError(f"解析LLM响应失败: {e}")
            
            except Exception as e:
                print(f"发生错误: {e}")
        
        # 如果所有重试都失败，则抛出异常
        raise ValueError("LLM 多次返回空响应或发生错误，请检查配置或输入内容")
    
    def _split_by_boundaries(self, text: str, boundaries: list[str]) -> list[str]:
        """根据边界在原文中切分"""
        chunks = []
        # 'text' 参数通常是 self.book_content，它在 __init__ 中已经被标准化处理。
        # 为确保一致性，如果 text 可能来自其他地方，也应进行相同的处理。
        # 但在此上下文中，text 就是 self.book_content。
        remaining_text = text

        for raw_boundary in boundaries:
            # 对LLM返回的边界也进行同样的标准化处理，以应对LLM可能产生的微小空白差异
            # boundary = re.sub(r'\s+', ' ', raw_boundary).strip()
            boundary = LLMSplitter._normalize_text(raw_boundary)
            if len(boundary) > 100:
                boundary = boundary[:100]  # 只取前100个字符，避免过长的边界影响查找
            
            print(f"Processing boundary (raw from LLM): '{raw_boundary}'")
            print(f"Processing boundary (normalized): '{boundary}'")

            if not boundary: # 跳过空边界
                print("Skipping empty or whitespace-only boundary.")
                continue
                
            try:
                # 在标准化后的文本中查找标准化后的边界
                split_idx = remaining_text.index(boundary) 
                # split_pos = split_idx + len(boundary)
                
                # chunk = remaining_text[:split_pos]
                if split_idx > 0:
                    chunk = remaining_text[:split_idx]
                    chunks.append(chunk)
                    remaining_text = remaining_text[split_idx:]
                    print(f"Successfully split at boundary. Remaining text starts with: '{remaining_text[:100]}...'")
            except ValueError: # 如果 .index() 找不到子字符串，会引发 ValueError
                print(f"Boundary (normalized) '{boundary}' NOT FOUND in remaining text.")
                print(f"Remaining text sample (first 500 chars): '{remaining_text[:500]}'")
                # 此处可以添加更详细的日志记录或错误处理策略
                continue 
        
        if remaining_text.strip(): # 添加最后剩余的文本块
            chunks.append(remaining_text)
            
        print(f"Initial chunks number: {len(boundaries)}")
        print(f"Final chunks number: {len(chunks)}")
        
        self.chunks = chunks
        return chunks
    
    def _process_long_document_directly(self) -> list[str]:
        """处理超长文档 - 直接输出方式"""
        initial_chunks = self.initial_chunks
        
        all_chunks = []
        for chunk in initial_chunks:
            prompt = self.set_prompt_directly(chunk)
            response = self.llm.generate(prompt)
            
            if not response:
                continue
            
            response = self._remove_invisible_chars(response)
            
            try:
                match = re.search(r'\{[\s\S]*\}', response)
                if not match:
                    continue
                
                json_str = match.group(0)
                result = json.loads(json_str)
                
                if isinstance(result, dict) and "chunks" in result:
                    all_chunks.extend(result["chunks"])
            except:
                continue
        
        self.chunks = all_chunks
        return all_chunks
    
    def _process_long_document_by_boundaries(self) -> list[str]:
        """处理超长文档 - 边界切分方式"""
        # 使用简单的字符分割先将文档切成较小的块
        initial_chunks = self.initial_chunks
        
        all_boundaries = []
        for i, chunk in enumerate(initial_chunks):
            # 最后一块不需要找边界
            # if i == len(initial_chunks) - 1:
            #     continue
                
            prompt = self.set_prompt_boundaries(chunk)
            response = self.llm.generate(prompt)
            
            if not response:
                continue
            
            response = self._remove_invisible_chars(response)
            
            try:
                match = re.search(r'\{[\s\S]*\}', response)
                if not match:
                    continue
                
                json_str = match.group(0)
                result = json.loads(json_str)
                
                if isinstance(result, dict) and "boundaries" in result:
                    # 只取最后一个边界，作为当前块的结束
                    if result["boundaries"]:
                        all_boundaries.extend(result["boundaries"])  # 只取最后一个边界
            except:
                print(f"Failed to split chunk {i+1}, skipping...")
                continue
        
        # 根据所有收集到的边界切分原始文本
        return self._split_by_boundaries(self.book_content, all_boundaries)
    
    def save_chunks(self, output_dir: str, prefix: str = "chunk_") -> None:
        """保存切分后的块到文件"""
        import os
        
        if not self.chunks:
            raise ValueError("没有可保存的文本块，请先运行切分方法")
        
        import shutil
        if os.path.exists(output_dir):
            # 先删除目录下的所有文件
            shutil.rmtree(output_dir)
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每个块到单独的文件
        for i, chunk in enumerate(self.chunks):
            file_path = os.path.join(output_dir, f"{prefix}{i+1}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)
        
        print(f"已将{len(self.chunks)}个文本块保存到 {output_dir} 目录")
        
    def set_prompt_chapter_titles(self, content: str) -> str:
        """设置让 LLM 输出章节标题和级别的提示词"""
        prompt = f"""
        You are a professional text processing assistant. Your task is to identify chapter titles and their hierarchical levels in the following text for the purpose of text chunking.

        CRITICAL REQUIREMENTS:
        1. Extract chapter titles EXACTLY as they appear in the original text
        2. Preserve ALL formatting: punctuation, capitalization, spacing, line breaks, and special characters
        3. Each title must be findable in the original text using exact string matching
        4. Do NOT modify, normalize, or clean up the titles in any way
        5. Assign hierarchical levels to each title

        Requirements:
        1. Analyze the text and identify chapter titles and their hierarchical levels
        2. Level 1: Book title/main title (if present)
        3. Level 2: Part/Section titles (if any)  
        4. Level 3+: Chapter titles and sub-chapters
        5. Ignore table of contents - extract titles from the actual content
        6. If a title spans multiple lines, preserve the exact line breaks as they appear
        7. For very simple titles (single words/numbers like "I", "1", "One", etc.), connect the title with the first following sentence using "\n" to make them more identifiable
        For example, if the title is "I" and the first sentence is "The story begins", return it as "I\nThe story begins". If the title is "1" and the first sentence is "The sun was shining brightly", return it as "1\nThe sun was shining brightly". If the title is "One" and the first sentence is "The story begins", return it as "One\nThe story begins". This helps LLMs recognize them as titles.

        IMPORTANT: The returned titles must match the original text character-for-character, including:
        - All punctuation marks (., -, :, ;, !, ?, etc.)
        - All spacing and whitespace
        - All line breaks (\\n)
        - All capitalization
        - All special characters

        Text to analyze:
        {content}

        Return ONLY a JSON array in this format:
        {{
            "chapter_titles": [
                {{"title": "Exact title as it appears in original text", "level": 1}},
                {{"title": "Exact title as it appears in original text", "level": 2}},
                {{"title": "Exact title as it appears in original text", "level": 3}},
                ...
            ]
        }}

        If no suitable chapter titles are found, return: {{"chapter_titles": []}}
        """
        return prompt
    
    def generate_chunks_by_chapters(self) -> list[str]:
        """让 LLM 输出章节标题和级别，然后以最高级别章节为边界进行切分"""
        # 如果文本超过 LLM 处理上限，先进行分块处理
        if self.tokens_num > self.max_llm_tokens:
            return self._process_long_document_by_chapters()
        
        # 文档长度在 LLM 处理范围内，直接处理
        prompt = self.set_prompt_chapter_titles(self.book_content)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(prompt)
                print(f"\n=== LLM Chapter Title Response (Attempt {attempt + 1}) ===")
                print(response)
                
                if not response or not response.strip():
                    print(f"Empty response on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        print("All attempts failed, falling back to semantic boundaries")
                        return self.generate_chunks_by_boundaries()
                    continue
                
                # 解析响应
                chapter_data = self._parse_chapter_titles_with_levels(response)
                if chapter_data is not None:
                    if not chapter_data:
                        print("LLM returned empty chapter titles list")
                        print("Falling back to semantic boundaries")
                        return self.generate_chunks_by_boundaries()
                    else:
                        # 找到最高级别（数值最大）的章节标题
                        highest_level_titles = self._extract_highest_level_titles(chapter_data)
                        print(f"LLM returned {len(chapter_data)} total chapter titles")
                        print(f"Using {len(highest_level_titles)} highest-level titles as boundaries")
                        return self._split_by_boundaries(self.book_content, highest_level_titles)
                else:
                    print(f"Failed to parse response on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        print("All parsing attempts failed, falling back to semantic boundaries")
                        return self.generate_chunks_by_boundaries()
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    print("All attempts failed due to errors, falling back to semantic boundaries")
                    return self.generate_chunks_by_boundaries()
                continue
    
    def _parse_chapter_titles_with_levels(self, response: str) -> list[dict] | None:
        """解析 LLM 的章节标题和级别响应"""
        if not response or not response.strip():
            return None
        
        response = self._remove_invisible_chars(response)
        try:
            # 查找 JSON 对象
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                print("No JSON object found in response")
                return None
            
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            if not isinstance(result, dict) or "chapter_titles" not in result:
                print("Invalid JSON structure, missing 'chapter_titles' key")
                return None
            
            titles_list = result["chapter_titles"]
            if not isinstance(titles_list, list):
                print("'chapter_titles' is not a list")
                return None
            
            # 验证每个标题项的格式
            valid_titles = []
            for item in titles_list:
                if (isinstance(item, dict) and 
                    "title" in item and 
                    "level" in item and
                    item["title"].strip()):
                    valid_titles.append({
                        "title": str(item["title"]),
                        "level": int(item["level"])
                    })
            
            return valid_titles
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error parsing response: {e}")
            return None
    
    def _extract_highest_level_titles(self, chapter_data: list[dict]) -> list[str]:
        """提取最高级别（数值最大）的章节标题"""
        if not chapter_data:
            return []
        
        # 找到最高级别（数值最大）
        max_level = max(item["level"] for item in chapter_data)
        
        # 过滤出最高级别的标题，跳过书籍标题（level 1）
        if max_level > 1:
            highest_level_titles = [
                item["title"] for item in chapter_data 
                if item["level"] == max_level
            ]
        else:
            # 如果只有书籍标题，返回空列表
            highest_level_titles = []
        
        print(f"Found titles at levels: {sorted(set(item['level'] for item in chapter_data))}")
        print(f"Using level {max_level} titles as boundaries: {len(highest_level_titles)} titles")
        
        # 打印前几个标题作为示例
        if highest_level_titles:
            print("Sample highest-level titles:")
            for i, title in enumerate(highest_level_titles):
                print(f"  {i+1}. '{title}'")
        
        return highest_level_titles
    
    def set_prompt_merge_chapter_titles(self, chunk_chapter_data: list[list[dict]]) -> str:
        """设置让 LLM 合并和统一章节标题等级的提示词"""
        prompt = f"""
        You are a professional text processing assistant. I will provide you with multiple chapter title arrays extracted from different parts of a novel. Your task is to merge them into a single unified structure with consistent level assignments.

        CRITICAL REQUIREMENTS:
        1. Preserve ALL chapter titles EXACTLY as they appear in the input
        2. Do NOT modify titles in any way - keep original formatting, punctuation, spacing, line breaks
        3. Keep the original order of titles
        4. Standardize and unify the level assignments across all chunks
        5. Use exact string matching for duplicate detection

        Chapter title data from different chunks:
        {json.dumps(chunk_chapter_data, ensure_ascii=False, indent=2)}

        Requirements for merging and level standardization:
        1. Remove duplicate chapter titles that appear in multiple chunks (using exact title matching)
        2. Maintain the correct chronological order of chapters
        3. STANDARDIZE LEVEL ASSIGNMENTS: Ensure consistent hierarchy across the entire document
        - Level 1: Book title/main title (if any)
        - Level 2: Part/Section titles (e.g., "Part I", "Book One", "Volume I")
        - Level 3: Main chapter titles (e.g., "Chapter 1", "Chapter I")
        - Level 4+: Sub-chapters or sections within chapters
        4. Handle overlapping content from chunk boundaries
        5. Preserve the original formatting of all titles EXACTLY
        6. Resolve level inconsistencies by analyzing title patterns and content structure

        Level Standardization Rules:
        - If the same title appears with different levels in different chunks, choose the most appropriate level based on the overall document structure
        - Consider title patterns (e.g., "Chapter", "Part", "Book", "Volume") to determine appropriate levels
        - Ensure hierarchical consistency (parent levels should be lower numbers than child levels)
        - Maintain semantic relationships between titles

        Chapter title patterns to consider for level assignment:
        - Book/Volume titles -> Level 1-2
        - Part/Section titles -> Level 2-3  
        - Chapter titles -> Level 3-4
        - Sub-sections -> Level 4+

        Return ONLY a JSON object in this format:
        {{
            "chapter_titles": [
                {{"title": "Exact title from input", "level": standardized_level_number}},
                {{"title": "Exact title from input", "level": standardized_level_number}},
                {{"title": "Exact title from input", "level": standardized_level_number}},
                ...
            ]
        }}

        Ensure the final result has:
        - No duplicate titles
        - Consistent and logical level assignments
        - Proper chronological ordering
        - Preserved original title formatting
        
        Now process the provided chapter title data and return the unified structure.
        """
        return prompt

    
    def _process_long_document_by_chapters(self) -> list[str]:
        """处理超长文档 - 章节标题切分方式"""
        initial_chunks = self.initial_chunks
        
        all_chapter_data = []
        for i, chunk in enumerate(initial_chunks):
            print(f"Processing chunk {i+1}/{len(initial_chunks)} for chapter titles")
            
            prompt = self.set_prompt_chapter_titles(chunk)
            response = self.llm.generate(prompt)
            
            if not response:
                continue
            
            try:
                chapter_data = self._parse_chapter_titles_with_levels(response)
                if chapter_data:
                    # 打印提取到的章节标题
                    print(f"Chunk {i+1} chapter titles:")
                    for _, item in enumerate(chapter_data):
                        print(f"Level {item['level']}: '{item['title']}'")
                    all_chapter_data.extend(chapter_data)
                    print(f"Extracted {len(chapter_data)} chapter titles from chunk {i+1}")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                continue
        
        if not all_chapter_data:
            print("No chapter titles found in any chunk, falling back to semantic boundaries")
            return self.generate_chunks_by_boundaries()
        print("All chapter titles:")
        for _, item in enumerate(all_chapter_data):
            print(f"Level {item['level']}: '{item['title']}'")

        # 去重章节标题（保持顺序）
        unique_chapter_data = []
        seen_titles = set()
        for item in all_chapter_data:
            title = item["title"]
            if title not in seen_titles:
                unique_chapter_data.append(item)
                seen_titles.add(title)
        
        print(f"Found {len(unique_chapter_data)} unique chapter titles")
        
        # 提取最高级别的章节标题
        highest_level_titles = self._extract_highest_level_titles(unique_chapter_data)
        
        if not highest_level_titles:
            print("No suitable chapter titles found, falling back to semantic boundaries")
            return self.generate_chunks_by_boundaries()
        
        # 根据最高级别的章节标题切分原始文本
        return self._split_by_boundaries(self.book_content, highest_level_titles)
