from src.chapterizer import Chapterizer
from src.loader import BookLoader, BookMetaDataLoader
from src.path_builder import NovelQAPathBuilder
from src.utils import load_json
import re
import os

chapter_patterns = {}
chapter_patterns["B00"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^PART\s+[IVXLCDM]+\.\s+.*')]
# chapter_patterns["B02"] = [re.compile(r'^[A-Z\s]+$')]
# chapter_patterns["B05"] = [re.compile(r'^[IVXLCDM]+\.$'), re.compile(r'^PART\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B07"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.$'), re.compile(r'^BOOK\s+[IVXLCDM]+\.\s*$')]
# chapter_patterns["B10"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*\.$')]
# chapter_patterns["B11"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B13"] = [re.compile(r'^CHAPTER\s+[1-9][0-9]*$'), re.compile(r'^BOOK\s+[IVXLCDM]+\s*$')]
# chapter_patterns["B14"] = [re.compile(r'^CHAPTER\s+[IVXLCDM]+\.\s+.*'), re.compile(r'^Book\sthe\s.*')]
# chapter_patterns["B17"] = [re.compile(r'^Chapter\s+[A-Z][a-z]+\s*$'), re.compile(r'^BOOK\s+[A-Z]+\s*$')]
# chapter_patterns["B18"] = [re.compile(r'^[IVXLCDM]+$')]

BOOK_ID = 'B00'

path_builder = NovelQAPathBuilder('./data/NovelQA')
book_loader = BookLoader(path_builder.get_book_path(BOOK_ID), BOOK_ID)
book_loader.load()
book_content = book_loader.get_content()
meta_data_loader = BookMetaDataLoader(path_builder.get_meta_data_path())
meta_data_loader.load()
description = meta_data_loader.get_title(BOOK_ID)
chapterizer = Chapterizer(book_content, description, chapter_patterns[BOOK_ID])
markdown = chapterizer.to_markdwon()
with open(f'{BOOK_ID}_temp.txt', 'w') as f:
    f.write(markdown)

_ = input()

with open(f'{BOOK_ID}_temp.txt', 'r') as f:
    markdown = f.read()

chapterizer.structure_from_markdown(markdown)

os.remove(f'{BOOK_ID}_temp.txt')

chapterizer.save_structure('structure.json')
