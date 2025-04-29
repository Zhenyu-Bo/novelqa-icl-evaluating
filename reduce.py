from src.llm import LLM, get_llm
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader, QuestionLoader, BookMetaDataLoader
from src.chapterizer import Chapterizer
from src.extractor import extract_option
import unicodedata
import argparse
import os
import json
import logging
from dotenv import load_dotenv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--book_id", type=str, default="all")
parser.add_argument("--question_id", type=str, default="all")
args = parser.parse_args()

def question_transform(question: str, llm: LLM) -> str:
    prompt: str = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel. However, the novel is too long, so I will give the novel chapter by chapter, and you need to transform the question for each chapter. You should make sure the user will be able to get the answer of the original question with only the answers of the transformed quesions for each chapter. For example, if the question is 'How many times has Alice mentioned in the novel?', the transformed question may be 'Is Alice mentioned in this chapter? If so, how many times has Alice mentioned in this chapter?', if the question is 'Which chapter mentions Alice.', the transformed question may be 'Is Alice mentioned in this chapter?', if the question is 'When Jane Eyre met Mr. Lloyd for the first time, what's her feeling towards him?', the transformed question should be 'If Jane Eyre met Mr. Lloyd in this chapter? If so, what's her feeling towards him in the first meeting?'. You output should be the following format: {{"question": "the transformed question"}}.You should give only one best transformed question, and not output anything else.\nThe given question is {question}."""
    # print(question)
    response = llm.generate(prompt)
    # print(response)
    transformed_question = response.split('"question": "')[1].split('"')[0]
    # print(transformed_question)
    return transformed_question

def build_prompt_icl(chapter_content: str, question_options: str) -> str:
    """创建提示词
    基本的思路是，让模型分析问题，给出回答和对应的证据
    """
    return f"""You are a literature professor. I will provide you with the full text of a chapter from a novel along with a question. Please thoroughly analyze the chapter's content to accurately respond to the following question.\nChapter Content:{chapter_content};\nBook ends. Questions start here:\n{question_options}\nQuestions end here. Try your best to answer the question based on the given full text of the chapter. The answer should be the analysis of text content around the question with the evidence from the chapter, and the answer."""


NOVELQA_PATH = '../RAG/ReadAgent-RAG/NovelQA'
path_builder = NovelQAPathBuilder(NOVELQA_PATH)

def remove_invisible_chars(s):
    return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
BOOK_IDS.remove("B00")  # 已测试
BOOK_IDS.remove("B01")
BOOK_IDS.remove("B02")
BOOK_IDS.remove("B35")
BOOK_IDS.remove("B36")
BOOK_IDS.remove("B40")
BOOK_IDS.remove("B47")
BOOK_IDS.remove("B48")  # 内容太长，予以舍弃
BOOK_IDS.remove("B57")
BOOK_IDS.remove("B61")

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
llm: LLM = get_llm('gemini', api_key=api_key)
test_aspects = ['times', 'meaning', 'character']
test_complexity = ['dtl']


def get_chapter_contents_cached(chapterizer, book_id: str, cache_dir: str = "./cache") -> tuple:
    """
    检查缓存文件是否存在，若存在则直接读取 chapterizer 的结构数据，否则调用 chapterizer.get_chapter_contents() 并保存到文件中。

    Args:
        chapterizer: Chapterizer 对象
        book_id (str): 书籍ID
        cache_dir (str): 缓存目录，默认 "./cache"
    
    Returns:
        tuple: (structure_dict, titles)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{book_id}_chapters.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        structure_dict = data.get("structure_dict")
        titles = data.get("titles")
    else:
        structure_dict, titles = chapterizer.get_chapter_contents()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"structure_dict": structure_dict, "titles": titles}, f, ensure_ascii=False, indent=4)
    return structure_dict, titles


def reduce_test(book_id: str, test_question_id: str, llm: LLM):
    print(f"Processing book {book_id}")
    book_path = path_builder.get_book_path(book_id)
    book_loader = BookLoader(book_path, book_id)
    book_loader.load()
    book_content = book_loader.get_content()
    question_path = path_builder.get_question_path(book_id)
    question_loader = QuestionLoader(question_path, book_id)
    question_loader.load()
    meta_data_path = path_builder.get_meta_data_path()
    meta_data_loader = BookMetaDataLoader(meta_data_path)
    meta_data_loader.load()
    book_title = meta_data_loader.get_title(book_id)
    question_dict = question_loader.get_whole()
    for question_id, question in tqdm(question_dict.items(), desc=f"Processing questions for book {book_id}\n"):
        if test_question_id != 'all' and question_id!= test_question_id:
            continue
        if is_answered(book_id, question_id):
            print(f"Question {question_id} already answered, skipping.")
            continue
        question = question_loader.get_by_id(question_id)
        if question.get_aspect() not in test_aspects and question.get_complexity() not in test_complexity:
            print(f"Skipping question {question_id}, aspect: {question.get_aspect()}, complexity: {question.get_complexity()}")
            continue
        print(f"{question_id}: {question.get_question_str()}")
        transformed_question = question_transform(question.get_question_str(), llm)
        chapterizer = Chapterizer(book_content, book_title)
        structure = chapterizer.get_structure()
        prompt_final = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel, and a series of answers. The answers are to the question {transformed_question} for each chapter of the novel. You need to give the answer to the question based on the given answers. Here is the original question: {question.get_question_options()}, and the following are the answers to the transformed question for each chapter. """
        # structure_dict, titles = chapterizer.get_chapter_contents()
        structure_dict, titles = get_chapter_contents_cached(chapterizer, book_id)  # 使用缓存函数获取章节结构数据
        chapter_answers = []
        for title in tqdm(titles, desc="Processing chapters", leave=False):
            title_desc = title.split('_')[-1]
            for t in title.split('_')[:-1]:
                title_desc +=' of '+ t
            # print(title_desc)
            content = structure_dict[title]
            prompt_chapter = build_prompt_icl(content, transformed_question)
            answer_chapter = llm.generate(prompt_chapter)
            # print(answer_chapter)
            prompt_final += f"""The answer to the chapter {title_desc} is {answer_chapter}.\n"""
            chapter_answers.append(f"The answer to the chapter {title_desc} is {answer_chapter}. ")
        # print(prompt_final)
        prompt_final += f"""Now give your analysis and then the best choice of the original question. Note that you should also reexamize each answer and evidences to the transformed question rather than directly use them."""
        prompt_final += """At the last of your answer, you need to give your choice again in the format '<answer>my final answer: A, B, C, or D</answer>'."""
        prompt_final += """For example, if your answer is A, you should output <answer>my final answer: A</answer> at the last of your answer."""
        answer_final = llm.generate(prompt_final)
        print(answer_final)
        llm_option = extract_option(answer_final)
        is_correct = question.get_answer() == llm_option
        question_dict[question_id]["ModelAnswer"] = llm_option
        question_dict[question_id]["Correct"] = is_correct
        question_dict[question_id]["Analysis"] = answer_final
        question_dict[question_id]["ChapterAnswers"] = chapter_answers
        # question_dict[question_id]["prompt"] = prompt_final
        question_dict[question_id]["TransformedQuestion"] = transformed_question
        print(f"Question {question_id} - Model Answer: {llm_option}, Correct: {is_correct}")
        prompt_final = ""
        save_results_by_question(question_dict, book_id, question_id, output_dir="./outputs")

    return question_dict


def save_results(results: dict, book_id: str, output_dir: str = "./outputs"):
    # output_dir = f"{NOVELQA_PATH}/outputs/results"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{book_id}.json", 'w+', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_dir}/{book_id}.json")


def save_results_by_question(results: dict, book_id: str, question_id: str, output_dir: str = "./outputs"):
    """
    保存结果到文件中。
    如果文件不存在，则直接写入 results
    如果文件存在，则从文件中加载已有数据，对每个问题的结果进行更新，然后写回文件
    """
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{book_id}.json")
    
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as file:
            existing_results = json.load(file)
        # 直接覆盖已有结果
        existing_results[question_id] = results[question_id]
        updated_results = existing_results
    else:
        # 如果文件不存在，则直接使用新的结果
        updated_results = results
        
    with open(outfile, 'w', encoding='utf-8') as file:
        json.dump(updated_results, file, ensure_ascii=False, indent=4)
    print(f"Results saved to {outfile}")
    
    
def is_answered(book_id: str, question_id: str, output_dir: str = './outputs') -> bool:
    outfile = os.path.join(output_dir, f"{book_id}.json")
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as file:
            questions = json.load(file)
        if question_id in questions and 'ModelAnswer' in questions[question_id]:
            return True
    return False


if __name__ == "__main__":
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    # BOOK_IDS = [f"B{i:02}" for i in range(0, 1)]
    for book_id in tqdm(BOOK_IDS, desc="Processing books"):
        if args.book_id != 'all' and book_id != args.book_id:
            continue
        results = reduce_test(book_id, args.question_id, llm)
        # save_results(results, book_id, output_dir=output_dir)
    
