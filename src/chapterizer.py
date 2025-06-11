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


    def __init__(self, book_content: str, book_title: str):
        """构造方法
        Args:
            book_content (str): 小说内容
            book_title (str): 小说标题
        """
        # 书籍内容
        self.book_content = book_content
        # 章节标题字典，键为章节标题，值为章节级别，小说标题级别为1，其后章节级别依次递增
        self.chapter_levels = {book_title: 1}
        # 章节标题列表，小说标题排在最前面，按出现顺序排列
        self.chapter_titles = [book_title]
        # 章节结构字典，存储标题、内容、子章节，其中内容不包含子章节的内容
        self.structure = {'title': book_title, 'structures': [], 'content': ''}
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
    
    def to_markdown(self):
        """将章节结构转换为markdown格式"""
        markdown = ""
        for title in self.chapter_titles:
            level = self.chapter_levels[title]
            markdown += "#" * level + " " + title + "\n"
        return markdown


import re
import json
import logging
import unicodedata
import tiktoken
import time
from typing import List, Dict, Optional, Set 
from tqdm import tqdm 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .llm import LLM 

class LLMSplitter():
    """使用 LLM 将书本内容按语义切分为 chunks """
    def __init__(self, llm: LLM, book_content: str, chunk_tokens=50000, max_llm_tokens=100000, chunk_overlap=0, max_retries=3, retry_delay=1.0, min_chunk_tokens_for_merge=100, max_chunk_tokens_for_merge=20000):
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # 将所有连续的空白字符（包括\n, \r, \t, 空格等）替换为单个空格，并去除首尾空格
        # self.book_content = LLMSplitter._normalize_text(book_content)
        self.book_content = book_content.strip()
        self.chunk_tokens = chunk_tokens
        self.max_llm_tokens = max_llm_tokens
        self.chunk_overlap = chunk_overlap # 分块重叠部分的 token 数量
        self.chunks = []
        self.token_counter = tiktoken.get_encoding("cl100k_base")
        self.tokens_num = len(self.token_counter.encode(self.book_content)) # 使用处理后的 book_content 计算 tokens
        self.logger.info(f"Book content token count (after normalization): {self.tokens_num}")
        
        # 重试配置
        self.max_retries = max_retries
        self.retry_delay = retry_delay  # 重试间隔时间（秒）
        
        # 小块合并配置
        self.min_chunk_tokens_for_merge = min_chunk_tokens_for_merge
        self.max_chunk_tokens_for_merge = max_chunk_tokens_for_merge if max_chunk_tokens_for_merge is not None else chunk_tokens
    
    def _count_tokens(self, text: str) -> int:
        """Accurately count tokens in a text string."""
        if not text:
            return 0
        try:
            return len(self.token_counter.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed, estimating using len(text)//4. Error: {e}")
            return len(text) // 4 
    
    def _call_llm_with_retry(self, prompt: str, operation_name: str = "LLM call") -> str | None:
        """带重试机制的 LLM 调用方法"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"[{operation_name}] Attempt {attempt + 1}/{self.max_retries}")
                response = self.llm.generate(prompt)
                
                if response and response.strip():
                    self.logger.info(f"[{operation_name}] Success on attempt {attempt + 1}")
                    return response
                else:
                    self.logger.info(f"[{operation_name}] Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.info(f"[{operation_name}] Error on attempt {attempt + 1}: {str(e)}")
                
            # 如果不是最后一次尝试，等待一段时间再重试
            if attempt < self.max_retries - 1:
                self.logger.info(f"[{operation_name}] Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
        
        self.logger.info(f"[{operation_name}] All {self.max_retries} attempts failed")
        return None
            
    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        # 移除不可见字符
        text = ''.join(c for c in text if unicodedata.category(c) not in ('Cc', 'Cf') or c in [' ', '\t', '\n'])
        # 移除连字符换行
        text = re.sub(r'-\s*\n\s*', '', text)
        # 标准化 Unicode 字符
        text = unicodedata.normalize('NFKC', text)
        # 替换连续的空白字符为单个空格，并去除首尾空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def _remove_invisible_chars(s: str) -> str:
        """移除不可见字符"""
        if not isinstance(s, str):
            self.logger.info(f"Warning: {s} is not a string, type: {type(s)}")
            return None
        return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))
    
    def _parser_json_from_response(self, response: str) -> dict:
        """从 LLM 响应中解析 JSON 字符串"""
        response = self._remove_invisible_chars(response)
        try:
            # 使用正则表达式提取 JSON 字符串
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                raise ValueError("未能在LLM响应中找到有效的JSON对象")
            json_str = match.group(0)
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            self.logger.error(f"解析LLM响应失败: {e}")
            raise ValueError(f"解析LLM响应失败: {e}")
    
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
        response = self._call_llm_with_retry(prompt, "Direct chunking")
        
        if not response:
            raise ValueError("LLM 多次调用失败，请检查配置或输入内容")
        
        response = self._remove_invisible_chars(response)
        
        try:
            # 尝试从响应中提取 JSON
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                raise ValueError("未能在LLM响应中找到有效的JSON对象")
            
            json_str = match.group(0)
            result = json.loads(json_str)
            
            if not isinstance(result, dict) or "chunks" not in result:
                raise ValueError("LLM响应中的JSON格式不正确，未找到'chunks'字段")
            
            self.chunks = result["chunks"]
            
            # 执行小块合并
            self._merge_small_consecutive_chunks()
            
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
        response = self._call_llm_with_retry(prompt, "Boundary detection")
        
        if not response:
            raise ValueError("LLM 多次调用失败，请检查配置或输入内容")
        
        self.logger.info("LLM response:")
        self.logger.info(response)
        
        response = self._remove_invisible_chars(response)
        
        try:
            # 尝试从响应中提取 JSON
            result = self._parser_json_from_response(response)
            
            # 根据边界在原文中切分
            boundaries = result["boundaries"]
            self.chunks = self._split_by_boundaries(self.book_content, boundaries)
            
            # 执行小块合并
            self._merge_small_consecutive_chunks()
            
            return self.chunks
            
        except json.JSONDecodeError as e:
            raise ValueError(f"解析LLM响应失败: {e}")
    
    def _split_by_boundaries(self, text: str, boundaries: list[str]) -> list[str]:
        """根据边界在原文中切分"""
        text = LLMSplitter._normalize_text(text)  # 确保文本经过标准化处理
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
                # 只取前 100 个字符
                boundary = boundary[:100]
            
            self.logger.info(f"Processing boundary (raw from LLM): '{raw_boundary}'")
            self.logger.info(f"Processing boundary (normalized): '{boundary}'")

            if not boundary: # 跳过空边界
                self.logger.info("Skipping empty or whitespace-only boundary.")
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
                    self.logger.info(f"Successfully split at boundary. Remaining text starts with: '{remaining_text[:100]}...'")
            except ValueError: # 如果 .index() 找不到子字符串，会引发 ValueError
                self.logger.info(f"Boundary (normalized) '{boundary}' NOT FOUND in remaining text.")
                self.logger.info(f"Remaining text sample (first 500 chars): '{remaining_text[:500]}'")
                # 此处可以添加更详细的日志记录或错误处理策略
                continue 
        
        if remaining_text.strip(): # 添加最后剩余的文本块
            chunks.append(remaining_text)
            
        self.logger.info(f"Initial chunks number: {len(boundaries)}")
        self.logger.info(f"Final chunks number: {len(chunks)}")
        
        # self.chunks = chunks
        return chunks
    
    def _process_long_document_directly(self) -> list[str]:
        """处理超长文档 - 直接输出方式"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_llm_tokens,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens
        )
        initial_chunks = text_splitter.split_text(self.book_content)
        self.logger.info(f"Initial chunks number: {len(initial_chunks)}")
        
        all_chunks = []
        for i, chunk in enumerate(initial_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(initial_chunks)} for direct chunking")
            prompt = self.set_prompt_directly(chunk)
            response = self._call_llm_with_retry(prompt, f"Direct chunking - chunk {i+1}")
            
            if not response:
                self.logger.info(f"Skipping chunk {i+1} due to LLM failure")
                continue
            
            response = self._remove_invisible_chars(response)
            
            try:
                match = re.search(r'\{[\s\S]*\}', response)
                if not match:
                    self.logger.info(f"No JSON found in chunk {i+1} response")
                    continue
                
                json_str = match.group(0)
                result = json.loads(json_str)
                
                if isinstance(result, dict) and "chunks" in result:
                    all_chunks.extend(result["chunks"])
                    self.logger.info(f"Successfully processed chunk {i+1}, got {len(result['chunks'])} sub-chunks")
            except Exception as e:
                self.logger.info(f"Failed to parse chunk {i+1} response: {e}")
                continue
        
        self.chunks = all_chunks
        
        # 执行小块合并
        self._merge_small_consecutive_chunks()
        
        return self.chunks
    
    def _process_long_document_by_boundaries(self) -> list[str]:
        """处理超长文档 - 边界切分方式"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_llm_tokens,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens
        )
        initial_chunks = text_splitter.split_text(self.book_content)
        self.logger.info(f"Initial chunks number: {len(initial_chunks)}")
        
        all_boundaries = []
        for i, chunk in enumerate(initial_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(initial_chunks)} for boundary detection")
            prompt = self.set_prompt_boundaries(chunk)
            response = self._call_llm_with_retry(prompt, f"Boundary detection - chunk {i+1}")
            
            if not response:
                self.logger.info(f"Skipping chunk {i+1} due to LLM failure")
                continue
            
            response = self._remove_invisible_chars(response)
            
            try:
                match = re.search(r'\{[\s\S]*\}', response)
                if not match:
                    self.logger.info(f"No JSON found in chunk {i+1} response")
                    continue
                
                json_str = match.group(0)
                result = json.loads(json_str)
                
                if isinstance(result, dict) and "boundaries" in result:
                    if result["boundaries"]:
                        all_boundaries.extend(result["boundaries"])
                        self.logger.info(f"Successfully processed chunk {i+1}, got {len(result['boundaries'])} boundaries")
            except Exception as e:
                self.logger.info(f"Failed to parse chunk {i+1} response: {e}")
                continue
        
        # 根据所有收集到的边界切分原始文本
        self.chunks = self._split_by_boundaries(self.book_content, all_boundaries)
        
        # 执行小块合并
        self._merge_small_consecutive_chunks()
        
        return self.chunks
    
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
        
        self.logger.info(f"已将{len(self.chunks)}个文本块保存到 {output_dir} 目录")
        
    def set_prompt_chapter_titles(self, content: str) -> str:
        """设置让 LLM 输出章节标题和级别的提示词"""
        prompt = f"""
        You are a professional text processing assistant. Your task is to identify chapter titles and their hierarchical levels in the following text for the purpose of text chunking.

        CRITICAL REQUIREMENTS:
        1. Extract chapter titles EXACTLY as they appear in the original text
        2. Preserve ALL formatting: punctuation, capitalization, spacing, line breaks, and special characters
        3. Each title must be findable in the original text using exact string matching
        4. Do NOT modify, normalize, or clean up the titles in any way. Do NOT add quotation marks, brackets, or any other characters not present in the original
        5. Assign hierarchical levels to each title
        
        VALIDATION REQUIREMENT:
        Before including any title in your response, mentally verify that the EXACT string (character-for-character) exists in the provided text. If you cannot find the exact match, do not include that title.

        Requirements:
        1. Analyze the text and identify chapter titles and their hierarchical levels
        2. Level 1: Book title/main title (if present)
        3. Level 2: Part/Section titles (if any)  
        4. Level 3+: Chapter titles and sub-chapters
        5. Ignore table of contents - extract titles from the actual content
        6. If a title spans multiple lines, preserve the exact line breaks as they appear
        7. For very simple titles (single words/numbers like "I", "1", "One", etc.), connect the title with the first following sentence using "\n" to make them more identifiable
        For example, if the title is "I" and the first sentence is "The story begins", return it as "I\nThe story begins". If the title is "1" and the first sentence is "The sun was shining brightly", return it as "1\nThe sun was shining brightly". If the title is "One" and the first sentence is "The story begins", return it as "One\nThe story begins". This helps LLMs recognize them as titles.

        IMPORTANT: The selected titles must match the original text character-for-character, including:
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
        response = self._call_llm_with_retry(prompt, "Chapter title extraction")
        
        if not response:
            self.logger.info("All chapter title extraction attempts failed, falling back to semantic boundaries")
            return self.generate_chunks_by_boundaries()
        
        self.logger.info(f"\n=== LLM Chapter Title Response ===")
        self.logger.info(response)
        
        # 解析响应
        chapter_data = self._parse_chapter_titles_with_levels(response)
        if chapter_data is not None:
            if not chapter_data:
                self.logger.info("LLM returned empty chapter titles list")
                self.logger.info("Falling back to semantic boundaries")
                return self.generate_chunks_by_boundaries()
            else:
                # 从章节标题中提取可以作为分块边界的标题
                boundaries = self._extract_boundaries_from_titles(chapter_data)
                self.logger.info(f"LLM returned {len(chapter_data)} total chapter titles")
                self.logger.info(f"Using {len(boundaries)} titles as boundaries")
                self.chunks = self._split_by_boundaries(self.book_content, boundaries)
                
                # 执行小块合并
                self._merge_small_consecutive_chunks()
                
                return self.chunks
        else:
            self.logger.info("Failed to parse chapter title response, falling back to semantic boundaries")
            return self.generate_chunks_by_boundaries()
    
    def _parse_chapter_titles_with_levels(self, response: str) -> list[dict] | None:
        """解析 LLM 的章节标题和级别响应"""
        if not response or not response.strip():
            return None
        
        response = self._remove_invisible_chars(response)
        try:
            # 查找 JSON 对象
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                self.logger.info("No JSON object found in response")
                return None
            
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            if not isinstance(result, dict) or "chapter_titles" not in result:
                self.logger.info("Invalid JSON structure, missing 'chapter_titles' key")
                return None
            
            titles_list = result["chapter_titles"]
            if not isinstance(titles_list, list):
                self.logger.info("'chapter_titles' is not a list")
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
            self.logger.info(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            self.logger.info(f"Unexpected error parsing response: {e}")
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
        
        self.logger.info(f"Found titles at levels: {sorted(set(item['level'] for item in chapter_data))}")
        self.logger.info(f"Using level {max_level} titles as boundaries: {len(highest_level_titles)} titles")
        
        # 打印前几个标题作为示例
        if highest_level_titles:
            self.logger.info("Sample highest-level titles:")
            for i, title in enumerate(highest_level_titles):
                self.logger.info(f"  {i+1}. '{title}'")
        
        return highest_level_titles
    
    def _extract_boundaries_from_titles(self, chapter_data: list[dict]) -> list[str]:
        """从章节标题数据中提取边界字符串"""
        if not chapter_data:
            return []
        
        boundaries = []
        for idx, item in enumerate(chapter_data):
            title = item["title"].strip()
            level = item["level"]
            if idx < len(chapter_data) - 1:
                next_title_level = chapter_data[idx + 1]["level"]
                if level >= next_title_level:
                    # 如果当前标题的级别不小于下一个标题的级别，使用当前标题作为边界
                    # 因为如果当前标题的级别小于下一个标题的级别，则下一个标题对应章节可能是当前标题对应章节的子章节
                    boundaries.append(title)
            else:
                # 最后一个标题直接添加
                boundaries.append(title)
        self.logger.info(f"Extracted {len(boundaries)} boundaries from chapter titles")
        return boundaries
    
    def set_prompt_merge_chapter_titles(self, chapter_titles_from_chunks: str) -> str:
        """设置让 LLM 合并和统一章节标题等级的提示词"""
        prompt = f"""
        You are a professional text processing assistant. I will provide you with multiple chapter title arrays extracted from different parts of a novel. Your task is to merge them into a single unified structure with consistent level assignments.

        CRITICAL REQUIREMENTS:
        1. Preserve ALL chapter titles EXACTLY as they appear in the input
        2. Do NOT modify titles in any way - keep original formatting, punctuation, spacing, line breaks
        3. Keep the original order of titles
        4. Standardize and unify the level assignments across all chunks
        5. Use exact string matching for duplicate detection

        Here is the chapter title data from different chunks:
        {chapter_titles_from_chunks}

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
                {{"title": "Exact title from input", "level": 1}},
                {{"title": "Exact title from input", "level": 2}},
                {{"title": "Exact title from input", "level": 3}},
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_llm_tokens,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens
        )
        initial_chunks = text_splitter.split_text(self.book_content)
        self.logger.info(f"Initial chunks number: {len(initial_chunks)}")
        
        all_chapter_data = []
        chapter_titles_from_chunks = ""
        
        for i, chunk in enumerate(initial_chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(initial_chunks)} for chapter titles")
            
            prompt = self.set_prompt_chapter_titles(chunk)
            response = self._call_llm_with_retry(prompt, f"Chapter title extraction - chunk {i+1}")
            
            if not response:
                self.logger.info(f"Skipping chunk {i+1} due to LLM failure")
                continue
            
            try:
                chapter_data = self._parse_chapter_titles_with_levels(response)
                if chapter_data:
                    # 打印提取到的章节标题
                    self.logger.info(f"Chunk {i+1} chapter titles:")
                    for _, item in enumerate(chapter_data):
                        self.logger.info(f"Level {item['level']}: '{item['title']}'")
                    all_chapter_data.extend(chapter_data)
                    self.logger.info(f"Extracted {len(chapter_data)} chapter titles from chunk {i+1}")
                    chapter_titles_from_chunks += f"Titles from chunk {i+1}:\n {json.dumps(chapter_data, ensure_ascii=False, indent=2)}\n"
                
            except Exception as e:
                self.logger.info(f"Error processing chunk {i+1}: {e}")
                continue
        
        if not all_chapter_data:
            self.logger.info("No chapter titles found in any chunk, falling back to semantic boundaries")
            return self.generate_chunks_by_boundaries()

        # 使用 LLM 合并章节标题
        merge_prompt = self.set_prompt_merge_chapter_titles(chapter_titles_from_chunks)
        self.logger.info("Merging chapter titles with LLM...")
        
        i = 0
        while i < self.max_retries:
            response = self._call_llm_with_retry(merge_prompt, "Chapter title merging")
    
            unique_chapter_data = []
            if response and response.strip():
                self.logger.info("LLM merge response:")
                self.logger.info(response)
                unique_chapter_data = self._parse_chapter_titles_with_levels(response)
                if unique_chapter_data is not None:
                    break  # 成功解析，退出重试循环
            self.logger.info(f"LLM merge response is empty or invalid on attempt {i+1}, retrying...")
            i += 1
        
        if not unique_chapter_data:
            self.logger.info("Failed to merge chapter titles, using fallback method")
            # 备用合并方法
            seen_titles = set()
            for item in all_chapter_data:
                title = item["title"]
                if title not in seen_titles:
                    unique_chapter_data.append(item)
                    seen_titles.add(title)
        
        self.logger.info(f"Found {len(unique_chapter_data)} unique chapter titles")
        self.logger.info("All chapter titles:")
        for _, item in enumerate(unique_chapter_data):
            self.logger.info(f"Level {item['level']}: '{item['title']}'")
        
        # 从章节标题中提取可以作为分块边界的标题
        boundaries = self._extract_boundaries_from_titles(unique_chapter_data)
        
        if not boundaries:
            self.logger.info("No suitable chapter titles found, falling back to semantic boundaries")
            return self.generate_chunks_by_boundaries()
        
        # 根据分块边界切分原始文本
        self.chunks = self._split_by_boundaries(self.book_content, boundaries)
        
        # 执行小块合并
        self._merge_small_consecutive_chunks()
        
        return self.chunks
    
    def _merge_small_consecutive_chunks(self):
        """合并连续的小块"""
        if not self.chunks or len(self.chunks) < 2:
            self.logger.info("Not enough chunks to perform merging or no chunks present.")
            return

        self.logger.info(f"Starting post-merging of small chunks. Min tokens for a chunk to be considered small: < {self.min_chunk_tokens_for_merge}. Max combined tokens for merge: <= {self.max_chunk_tokens_for_merge}")
        
        merged_chunks: List[str] = []
        i = 0
        num_merges = 0
        original_chunk_count = len(self.chunks)

        while i < len(self.chunks):
            # 取当前块作为起始点
            current_chunk = self.chunks[i]
            current_tokens = self._count_tokens(current_chunk)
            
            # 如果当前块足够大，直接添加并前进
            if current_tokens >= self.min_chunk_tokens_for_merge:
                merged_chunks.append(current_chunk)
                i += 1
                continue
            
            # 尝试与后续多个块合并
            combined_chunk = current_chunk
            combined_tokens = current_tokens
            next_idx = i + 1
            chunks_merged = 0  # 当前轮次合并的块数
            
            # 继续尝试与后续块合并，直到达到token上限或没有更多块
            while next_idx < len(self.chunks):
                next_chunk = self.chunks[next_idx]
                next_tokens = self._count_tokens(next_chunk)
                
                # 检查合并后是否超过上限
                if (combined_tokens + next_tokens) <= self.max_chunk_tokens_for_merge:
                    # 合并块
                    combined_chunk = (combined_chunk + " " + next_chunk).strip()
                    combined_tokens += next_tokens  # 这是一个近似值，实际token数可能因空格而略有不同
                    chunks_merged += 1
                    next_idx += 1
                else:
                    # 如果合并下一个会超过限制，停止合并
                    break
            
            if chunks_merged > 0:
                # 实际计算最终合并块的token数，因为简单相加可能不精确
                actual_combined_tokens = self._count_tokens(combined_chunk)
                self.logger.info(f"Merged {chunks_merged+1} chunks starting at position {i+1} into a single chunk of {actual_combined_tokens} tokens.")
                merged_chunks.append(combined_chunk)
                num_merges += chunks_merged
                i = next_idx  # 跳过所有已合并的块
            else:
                # 无法合并，保留当前块
                self.logger.debug(f"Chunk {i+1} ({current_tokens} tk) is small but cannot be merged with any following chunks without exceeding limit.")
                merged_chunks.append(current_chunk)
                i += 1
        
        if num_merges > 0:
            self.logger.info(f"Performed {num_merges} merges of small chunks. Chunk count changed from {original_chunk_count} to {len(merged_chunks)}.")
            self.chunks = merged_chunks
        else:
            self.logger.info("No small consecutive chunks were merged.")

    def split_recursive(self, text: str, by_chapter: bool = True):
        """递归分割文本"""
        if not text or not text.strip():
            return
        text = LLMSplitter._normalize_text(text)  # 确保文本经过标准化处理
        if self._count_tokens(text) <= self.max_llm_tokens:
            front_text = text
        else:        
            # 文本长度超过 LLM 处理范围，提取前 max_tokens_num 个 token 从中寻找标题
            front_text = text[:self.max_llm_tokens * 4]
        for i in range(self.max_retries):
            try:
                prompt = self.set_prompt_chapter_titles(front_text) if by_chapter else self.set_prompt_boundaries(front_text)
                response = self._call_llm_with_retry(prompt, "Recursive chunking")
                if not response:
                    raise ValueError("Failed to get a valid response from LLM after multiple retries")
                else:
                    if by_chapter:
                        chapter_titles = self._parse_chapter_titles_with_levels(response)
                        boundaries = self._extract_boundaries_from_titles(chapter_titles)
                    else:
                        boundaries = self._parser_json_from_response(response)['boundaries']
                    chunks = self._split_by_boundaries(text, boundaries)
                if chunks:
                    self.logger.info(f"Successfully split text into {len(chunks)} chunks using LLM response.")
                    self.chunks.extend(chunks)
                    last_chunk = chunks[-1]
                    if self._count_tokens(last_chunk) <= self.chunk_tokens:
                        # 如果最后一个大块小于等于 max_chunk_tokens_for_merge ，则已经处理完成
                        self._merge_small_consecutive_chunks()
                        return
                    elif len(chunks) == 1 and i >= self.max_retries:
                        # 如果只有一个大块，说明切分失败，直接使用 RecursiveCharacterTextSplitter 进行分割
                        self.chunks.pop()
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.chunk_tokens,
                            chunk_overlap=self.chunk_overlap,
                            length_function=self._count_tokens
                        )
                        chunks = text_splitter.split_text(text)
                        self.chunks.extend(chunks)
                        self._merge_small_consecutive_chunks()
                        return
                    else:
                        self.logger.info(f"Last chunk is too large ({self._count_tokens(last_chunk)} tokens), continuing recursive splitting.")
                        self.chunks.pop()  # 移除最后一个大块，继续递归分割
                        self.logger.info(f"Now there are {len(self.chunks)} chunks, continuing recursive splitting.")
                        self.split_recursive(last_chunk, by_chapter=by_chapter)  # 递归处理最后一个大块
                else:
                    raise ValueError("No chunks generated from LLM response")
                return
            except Exception as e:
                self.logger.info(f"Error during recursive splitting: {e}, retrying...")
                continue
        self.logger.info("Reached maximum retries for recursive splitting, no valid chunks generated, treat total text as a single chunk.")
        self.chunks.append(text)  # 如果所有尝试都失败，作为一个整体块加入到 self.chunks 中
        return
