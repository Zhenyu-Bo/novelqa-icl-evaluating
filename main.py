"""
使用示例：
python main.py --model gemini --base_dir ./data/NovelQA --output ./results --use_content True --title_only False
"""

from src.loader import BookLoader, QuestionLoader, BookMetaDataLoader
from src.llm import LLM, get_llm
from src.path_builder import NovelQAPathBuilder
from src.extractor import extract_entries, extract_entries_no_evidence, merge
from src.utils import save_json
import os
import argparse

parser = argparse.ArgumentParser(description="Parse command line arguments.")
parser.add_argument('--model', type=str, required=True, help="Specify the model name.")
parser.add_argument('--base_dir', type=str, required=True, help="Specify the base directory name.")
parser.add_argument('--output', type=str, required=True, help="Specify the output directory name.")
parser.add_argument('--use_content', type=str, required=False, default=False, help="Specify whether to use content or not.")
parser.add_argument('--title_only', type=str, required=False, default=True, help="Specify whether to use title only or not.")
parser.add_argument('--use_reduce', type=str, required=False, default=False, help="Specify whether to use reduce or not.")

args = parser.parse_args()
model_name = args.model
assert model_name in ['gemini', 'deepseek', 'deepseek-r1']
base_dir = args.base_dir
output_dir = args.output
use_content = args.use_content.lower() == 'true'
title_only = args.title_only.lower() == 'true'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(f'{output_dir}/prompts'):
    os.makedirs(f'{output_dir}/prompts')
if not os.path.exists(f'{output_dir}/responses'):
    os.makedirs(f'{output_dir}/responses')

def build_prompt_icl(book_content: str, question_options: str) -> str:
    """创建提示词
    基本的思路是，让模型分析问题，给出回答和对应的证据
    """
    return f"""You are a literature professor. I will provide you with the full text of a novel along with a series of questions. Please thoroughly analyze the novel's content to accurately respond to each of the following questions.\nBook Content:{book_content};\nBook ends. Questions start here:\n{question_options}\nQuestions end here. Try your best to answer the questions based on the given full text of the novel. The answer should be the analysis of text content around the question, and then the best option, and then the evidence from the novel. The analysis should be detailed and based on the document, and the evidence should be short and brief. Your output format should be <First Question ID>: \n<analysis of text content around the question>\nAnswer: <best option>\n<evidences>\n\n<Second Question ID>:\n<analysis of text content around the question>\nAnswer: <best option> ... <nth Question ID>: (, each answer in one line with all the supporting evidences. Each evidence should be a sentence exactly from the original text without any paraphrase. for example, the output may be like this:\nQ0001:\nBased on the document and the question,...\nAnswer: A\nThe first evidence\nThe second evidence\n...\nQ0002:\nBased on the document and the question,...\nAnswer: B\nThe first evidence\nThe second evidence\n...\nQ000n:\nBase on the document and the question,...\nAnswer: C\nThe first evidence\nThe second evidence\n...\n\n"""

def build_zero_shot_prompt(book_discription: str, question_options: str) -> str:
    return f"""You are a literature professor. I will provide you a series of questions along with four choices for each question. Please accurately select the correct choice to each of the following questions. All the questions are related to a book and can be answered by the book content. Here is the discription of the book:\n{book_discription}\nDescription ends here.\nQuestions start here:\n{question_options}\nQuestions end here.\nTry your best to answer the questions based on your own knowledge. You should first analyze the question and the book and then give your choise. Your should output the choice to each question with the format Your output format should be <First Question ID>: \n<analysis of text content around the question>\nAnswer: <best option>\n\n<Second Question ID>:\n<analysis of text content around the question>\nAnswer: <best option> ... <nth Question ID>:...\nFor example, the output may be like this:\nQ0001:\nBased on the document and the question,...\nAnswer: A\n\nQ0002:\nBased on the document and the question,...\nAnswer: B\n\nQ000n:\nBase on the document and the question,...\nAnswer: C\n\nOne more thing, don't use the markdown output format."""

def build_zero_shot_only_title(book_title: str, question_options: str) -> str:
    return f"""You are a literature professor. I will provide you a series of questions along with four choices for each question. Please accurately select the correct choice to each of the following questions. All the questions are related to a book and can be answered by the book content. The title of the book is: {book_title}\nQuestions start here:\n{question_options}\nQuestions end here.\nTry your best to answer the questions based on your own knowledge. You should first analyze the question and the book and then give your choise. Your should output the choice to each question with the format Your output format should be <First Question ID>: \n<analysis of text content around the question>\nAnswer: <best option>\n\n<Second Question ID>:\n<analysis of text content around the question>\nAnswer: <best option> ... <nth Question ID>:...\nFor example, the output may be like this:\nQ0001:\nBased on the document and the question,...\nAnswer: A\n\nQ0002:\nBased on the document and the question,...\nAnswer: B\n\nQ000n:\nBase on the document and the question,...\nAnswer: C\n\nOne more thing, don't use the markdown output format."""

# BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
BOOK_IDS = ["B44"]
test_question_id = "Q0368"
# BOOK_IDS.remove("B06")
# BOOK_IDS.remove("B30")
# BOOK_IDS.remove("B45")
# BOOK_IDS.remove("B48") # 内容太长，予以舍弃

def test(book_id: str, llm: LLM) -> dict:
    """对一本书进行测试"""
    # 构造路径
    path_builder = NovelQAPathBuilder(base_dir)
    book_path = path_builder.get_book_path(book_id)
    question_path = path_builder.get_question_path(book_id)
    # 加载数据
    question_loader = QuestionLoader(question_path, book_id)
    question_loader.load()
    if use_content:
        book_loader = BookLoader(book_path, book_id)
        book_loader.load()
        book_content = book_loader.get_content()
    else:
        meta_data_loader = BookMetaDataLoader(path_builder.get_meta_data_path())
        meta_data_loader.load()
        discription = meta_data_loader.build_description(book_id)
        title = meta_data_loader.get_title(book_id)
    question_dict = dict(question_loader.get_whole())
    question_iter = 0
    while question_iter < len(question_loader):
        # 如果要控制每次处理的问题数量，可以使用以下代码，特别的，如果要逐个问题调用的，取间隔为1即可
        # end_iter = min(question_iter + 20, len(question_loader))
        # 使用以下的代码，每次向模型传入所有的问题
        end_iter = len(question_loader)
        print(f"Processing {book_id} from {question_iter} to {end_iter}")
        if book_id not in BOOK_IDS:
            break
        question = question_loader.get_by_id(test_question_id)
        questions = question.get_question_options()
        print(questions)
        # questions = "".join([f"{question.get_question_options()}\n" for question in question_loader[question_iter:end_iter]])
        question_iter = end_iter
        if use_content:
            prompt = build_prompt_icl(book_content, questions)
        else:
            if title_only:
                prompt = build_zero_shot_only_title(title, questions)
            else:
                prompt = build_zero_shot_prompt(discription, questions)
        with open(f"{output_dir}/prompts/{book_id}.txt", 'w', encoding='utf-8') as file:
            file.write(prompt)
        # 调用模型
        response = llm.generate(prompt)
        print(response)
        with open(f"{output_dir}/responses/{book_id}.txt", 'w', encoding='utf-8') as file:
            file.write(response)
        # 解析输出
        if use_content:
            entries = extract_entries(response)
        else:
            entries = extract_entries_no_evidence(response)
        # 合并
        question_dict = merge(entries, question_dict)
    return question_dict

if __name__ == "__main__":
    llm: LLM = get_llm(model_name)
    for book_id in BOOK_IDS:
        result = test(book_id, llm)
        save_json(result, f"{output_dir}/{book_id}.json")
