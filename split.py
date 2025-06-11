"""章节切分的示例代码"""

from src.chapterizer import Chapterizer
from src.loader import BookLoader, BookMetaDataLoader
from src.path_builder import NovelQAPathBuilder
import os

# path_builder = NovelQAPathBuilder('./data/NovelQA')
# book_loader = BookLoader(path_builder.get_book_path("B29"), "B29")
# book_loader.load()
# book_content = book_loader.get_content()
# lines = book_content.split('\n')
# lines = Chapterizer._ignore_toc(lines)
# with open('tmp.txt', 'w') as f:
#     for line in lines:
#         f.write(line + '\n')

# import sys; sys.exit(0)

# 一些书籍的章节标题格式
# chapter_patterns = {}
# chapter_patterns["B00"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^PART\s+[IVXLCDM]+\.\s+.*')]
# chapter_patterns["B02"] = [re.compile('^(INTRODUCTION|THE TALES AND THE PERSONS)$'), re.compile(r'^PART\s+[A-Z]+$'), re.compile(r'^[A-Z\s]+$')]
# chapter_patterns["B03"] = [re.compile(r'^Part\s+[A-Z][a-z]+\s*$'), re.compile(r'^CHAPTER\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B05"] = [re.compile(r'^[IVXLCDM]+\.$'), re.compile(r'^PART\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B07"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.$'), re.compile(r'^BOOK\s+[IVXLCDM]+\.\s*$')]
# chapter_patterns["B10"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*\.$')]
# chapter_patterns["B11"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B13"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B14"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^Book\sthe\s.*')]
# chapter_patterns["B17"] = [re.compile(r'^Chapter\s+[A-Z][a-z]+\s*$'), re.compile(r'^BOOK\s+[A-Z]+\s*$')]
# chapter_patterns["B18"] = [re.compile(r'^[IVXLCDM]+$')]

BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
BOOK_IDS.remove("B06")
BOOK_IDS.remove("B30")
BOOK_IDS.remove("B45")
BOOK_IDS.remove("B48") # 内容太长，予以舍弃
BOOK_IDS = ["B00"]

i = 0
while i < len(BOOK_IDS):
    book_id = BOOK_IDS[i]
    path_builder = NovelQAPathBuilder('../NovelQA')
    book_loader = BookLoader(path_builder.get_book_path(book_id), book_id)
    book_loader.load()
    book_content = book_loader.get_content()
    meta_data_loader = BookMetaDataLoader(path_builder.get_meta_data_path())
    meta_data_loader.load()
    title = meta_data_loader.get_title(book_id)

    # 章节化
    chapterizer = Chapterizer(book_content, title)

    # 生成章节标题
    # markdown = chapterizer.to_markdown()
    # print(f"生成章节标题 {book_id}:\n{markdown}")

    # # 保存章节标题
    # if not os.path.exists(f'structures/titles'):
    #     os.makedirs(f'structures/titles')

    # with open(f'structures/titles/{book_id}.txt', 'w', encoding='utf-8') as f:
    #     f.write(markdown)

    # # 中断以便用户检查生成之标题结构是否正确，如果不正确，用户可以修改生成之标题结构
    # # _ = input("请检查生成的章节标题是否正确，按回车继续")

    # # 加载用户修改之章节标题结构
    # with open(f'structures/titles/{book_id}.txt', 'r', encoding='utf-8') as f:
    #     markdown = f.read()

    # # 章节化
    # chapterizer.structure_from_markdown(markdown)

    # # 保存章节结构
    # chapterizer.save_structure(f'structures/{book_id}.json')

    chapter_dict, chapter_list = chapterizer.get_chapter_contents()

    if not os.path.exists(f'structures/chapters/{book_id}'):
        os.makedirs(f'structures/chapters/{book_id}')

    for idx, chapter in enumerate(chapter_list):
        with open(f'structures/chapters/{book_id}/{idx}.txt', 'w', encoding='utf-8') as f:
            f.write(chapter_dict[chapter])
    
    print(f"章节化完成 {book_id}")
    i += 1
    command = input("按回车继续")
    if command == 'q':
        break
    elif command != 'r':
        i += 1