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
        self.book_content = book_content
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
        return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

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
            while len(lines) > 0 and (current_structure_idx == len(structure['structures']) or not lines[0].lower().startswith(structures[current_structure_idx]['title'].lower())):
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
        return result_dict, result_list

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


from .llm import LLM
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
class LLMChapterizer(Chapterizer):
    """章节化类, 使用LLM进行章节化
    Args:
        Chapterizer (Chapterizer): 章节化类
    """
    def __init__(self, llm: LLM, book_content: str, book_title: str = None):
        self.llm = llm
        super().__init__(book_content, book_title)
        
    def _chapterize(self, chunk_size: int = 1000000, chunk_overlap: int = 10000):
        lines = self.book_content.split('\n')
        lines = [self._remove_invisible_chars(line) for line in lines if line.strip()]
        # lines = self._ignore_toc(lines)
        structure = self.generate_chapter_structure(lines)
        # structure = self.generate_chapter_contents_chunk_by_chunk(chunk_size = chunk_size, chunk_overlap = chunk_overlap)

        # 递归方法，用于遍历章节结构并生成 chapter_levels 和 chapter_titles
        def traverse_structure(structure, level):
            title = structure['title']
            self.chapter_levels[title] = level
            self.chapter_titles.append(title)
            for substructure in structure.get('structures', []):
                traverse_structure(substructure, level + 1)

        # 从根结构开始递归
        traverse_structure(structure, 1)
        
    def set_prompt(self, content: str) -> str:
        """设置让 LLM 进行章节切分的提示词"""
        prompt = f"""
        You are a helpful assistant. I will provide you with the content of a novel. Your task is to identify the chapters and their levels.
        Return the result as a structured JSON object where each chapter has a title and its level(1 for the main title, 2 for the first level, etc.).
        
        Note:
        - If the content includes a table of contents (TOC), **ignore the TOC and extract chapter titles directly from the main body of the text**.
        - Do not include "Contents" or similar non-chapter headings as part of the chapter structure.
        - The title of a chapter may not explicitly contain words like "Chapter", "Part", or "Section".
        - A chapter title may span multiple lines. For example, a title might consist of a main heading followed by a subtitle on the next line.
        - You need to infer the chapter boundaries and titles based on the context and semantics of the text.


        Here is the content of the novel: {content}

        Please provide the chapter structure in the following format:
        [
            {{"title": "[the title of the chapter1]", "level": [the level of the chapter1]}},
            {{"title": "[he title of the chapter2]", "level": [he title of the chapter2]}},
            ...
        ]
        For example, if the content is:
        "
            THE LAW AND THE LADY
            by Wilkie Collins
            
            Contents:
            Chapter 1. THE BRIDE’S MISTAKE
            Chapter 2. THE BRIDE’S THOUGHTS
            Chapter 3. THE STORY OF THE TRIAL. THE PRELIMINARIES
            Chapter 4. FIRST QUESTION--DID THE WOMAN DIE POISONED?
            Chapter 5. THE LAST OF THE STORY
            
            PART I. PARADISE LOST.
            CHAPTER I. THE BRIDE’S MISTAKE.
            “FOR after this manner in the old time the holy women also who trusted in God adorned themselves, ...”
            CHAPTER II. THE BRIDE’S THOUGHTS.
            WE had been traveling for a little more than an hour when a change passed insensibly over us both.
            PART II. PARADISE REGAINED.
            CHAPTER III. THE STORY OF THE TRIAL. THE PRELIMINARIES.
            LET me confess another weakness, on my part, before I begin the Story of the Trial ...
            CHAPTER IV. FIRST QUESTION--DID THE WOMAN DIE POISONED?
            THE proceedings began at ten o’clock. The prisoner was placed at the Bar, ...
            CHAPTER V.
            THE LAST OF THE STORY.
            In ten days more we returned to England, accompanied by Benjamin. ...
        "
        Your output should be:
        "
        [
            {{"title": "THE LAW AND THE LADY", "level": 1}},
            {{"title": "PART I. PARADISE LOST.", "level": 2}},
            {{"title": "CHAPTER I. THE BRIDE’S MISTAKE.", "level": 3}},
            {{"title": "CHAPTER II. THE BRIDE’S THOUGHTS.", "level": 3}},
            {{"title": "PART II. PARADISE REGAINED.", "level": 2}},
            {{"title": "CHAPTER III. THE STORY OF THE TRIAL. THE PRELIMINARIES.", "level": 3}},
            {{"title": "CHAPTER IV. FIRST QUESTION--DID THE WOMAN DIE POISONED?", "level": 3}},
            {{"title": "CHAPTER V. THE LAST OF THE STORY.", "level": 3}},
        ]"
        
        Now please provide the chapter structure for the given content.
        """
        
        return prompt
        
    
    def set_merge_prompt(self, structures: list[dict]) -> str:
        """设置让 LLM 合并章节的提示词"""
        prompt = f"""
        You are a helpful assistant. I will provide you with a list of chapter structures generated from different parts of a novel. 
        Your task is to merge these chapter structures into a single unified structure, ensuring there are no duplicate chapters across different parts.

        Here is the list of chapter structures: {json.dumps(structures, ensure_ascii=False)}

        Please provide the merged chapter structure in the following format:
        [
            {{"title": "[the title of the chapter1]", "level": [the level of the chapter1]}},
            {{"title": "[the title of the chapter2]", "level": [the level of the chapter2]}},
            ...
        ]

        Note:
        - The chapter structures are generated from different parts of the novel, and the chapter level standards may vary between parts. When merging, ensure that the chapter levels are unified and consistent across the entire structure.
        - If two or more chapters have the same title and level, they should be merged into a single chapter in the final structure.
        - The order of chapters should be preserved as much as possible based on their appearance in the input structures.
        - Ensure that the final structure is a valid JSON array and follows the specified format.
        
        Now please provide the merged chapter structure for the given list of chapter structures.
        """
        
        return prompt
    
    def _parse_chapter_structure(self, response: str) -> dict:
        """解析 LLM 的输出"""
        try:
            # 匹配 JSON 数组
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not match:
                raise ValueError("LLM response does not contain a valid JSON array")
            response = match.group(0)  # 提取匹配到的 JSON 数组

            # 解析 JSON
            result = json.loads(response)
            if not isinstance(result, list):
                raise ValueError("Parsed JSON is not a list")
            
            # 构造章节结构
            structure = {"title": self.book_title, "structures": [], "content": ""} if self.book_title else {"title": "The Book", "structures": [], "content": ""}
            structure_stack: list[dict] = [structure]  # 存储既往章节结构
            level_stack: list[int] = [1]  # 存储既往章节层级
            # self.chapter_levels = {structure['title']: 1}  # 初始化章节级别字典
            # self.chapter_titles = [structure['title']]  # 初始化章节标题列表

            for item in result:
                title = item['title']
                level = item['level']
                
                if level <= 1:
                    # 只有根节点（书籍标题）的层级为 1
                    continue

                # 如果当前章节的层级小于栈顶的层级，弹出栈顶
                while level <= level_stack[-1]:
                    pop_structure = structure_stack.pop()
                    structure_stack[-1]['structures'].append(pop_structure)  # 将弹出的章节添加到父章节的 structures 中
                    level_stack.pop()
                
                level_stack.append(level)
                structure_stack.append({"title": title, "structures": [], "content": ""})  # 将新章节压入栈中
                # self.chapter_levels[title] = level  # 更新章节级别
                # self.chapter_titles.append(title)  # 添加章节标题到列表中

            while len(structure_stack) > 1:
                pop_structure = structure_stack.pop()
                structure_stack[-1]['structures'].append(pop_structure)

            return structure
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    
    def generate_chapter_structure(self, content: str = None) -> dict:
        """调用 LLM 生成章节结构"""
        prompt = self.set_prompt(content) if content else self.set_prompt(self.book_content)
        response = self.llm.generate(prompt)
        response = self._remove_invisible_chars(response)
        print("LLM response:", response)
        # 解析 LLM 的输出
        structure = self._parse_chapter_structure(response)
        print(json.dumps(structure, indent=2, ensure_ascii=False))
        return structure

    def generate_chapter_contents_chunk_by_chunk(self, chunk_size: int = 1000000, chunk_overlap: int = 10000) -> list[dict]:
        """使用 LLM 生成章节内容，分块处理
        Args:
            chunk_size (int): 每个块的大小，默认 1000000 字符
            chunk_overlap (int): 块之间的重叠部分，默认 10000 字符
        Returns:
            list[str]: 章节内容列表
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(self.book_content)
        print(f"Total chunks: {len(chunks)}")
        structures = ""
        for chunk in chunks:
            structure = self.generate_chapter_structure(chunk)
            print(json.dumps(structure, indent=2, ensure_ascii=False))
            structures += json.dumps(structure, ensure_ascii=False) + "\n"
        structures = structures.strip().split('\n')
        # 合并章节结构
        prompt = self.set_merge_prompt(structures)
        response = self.llm.generate(prompt)
        response = self._remove_invisible_chars(response)
        print("LLM merge response:", response)
        # 解析 LLM 的输出
        merged_structure = self._parse_chapter_structure(response)
        print(json.dumps(merged_structure, indent=2, ensure_ascii=False))
        return merged_structure