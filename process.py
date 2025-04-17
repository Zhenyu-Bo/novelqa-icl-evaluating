import json
import argparse
from src.extractor import extract_entries, extract_entries_no_evidence, merge
from src.loader import QuestionLoader
from src.path_builder import NovelQAPathBuilder

# 解析命令行参数
parser = argparse.ArgumentParser(description='Process a book ID.')
parser.add_argument('book_id', type=str, help='The book ID to process')
args = parser.parse_args()

# 示例文件路径，替换为实际路径
input_file = f'./results/responses/{args.book_id}.txt'

with open(input_file, 'r') as f:
    content = f.read()
    entries = extract_entries_no_evidence(content)

path_builder = NovelQAPathBuilder('./data/NovelQA')
book_id = args.book_id
question_path = path_builder.get_question_path(book_id)
question_loader = QuestionLoader(question_path, book_id)
question_loader.load()

result = dict(question_loader.get_whole())
print(result.keys())

result = merge(entries, result)

with open(f"results/{book_id}.json", 'w', encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
