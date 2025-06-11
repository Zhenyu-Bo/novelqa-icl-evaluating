import time
from src.llm import LLM, get_llm
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader, QuestionLoader, BookMetaDataLoader
from src.chapterizer import Chapterizer
from src.extractor import extract_option
from src.prompt import build_transform_question_prompt, build_prompt_icl, build_prompt_final, build_prompt_icl2, build_prompt_icl_json, build_prompt_final_json
import unicodedata
import argparse
import os
import json
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import inflect
import re

os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

parser = argparse.ArgumentParser()
parser.add_argument("--book_id", type=str, default="all")
parser.add_argument("--question_id", type=str, default="all")
parser.add_argument("--max_workers", type=int, default=8, help="Number of worker processes to use")
parser.add_argument("--use_cache", type=bool, default=True, help="Whether to use cache for chapter contents")
parser.add_argument("--cache_dir", type=str, default="./cache/chapters", help="Directory to store cache files")
parser.add_argument("--output_dir", type=str, default="./outputs/reduce/selected/prompt8", help="Directory to store output files")
parser.add_argument("--skip_answered", action="store_true", help="Skip already answered questions")
parser.add_argument("--no_skip_answered", action="store_false", dest="skip_answered", help="Do not skip already answered questions")
parser.set_defaults(skip_answered=True)  # 默认值为 True
args = parser.parse_args()
skip_answered = args.skip_answered


NOVELQA_PATH = '../NovelQA'
path_builder = NovelQAPathBuilder(NOVELQA_PATH)

# BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
BOOK_IDS = ["B00", "B05", "B09", "B13", "B14", "B16", "B17", "B20", "B22", "B24",
            "B25", "B29", "B33", "B34", "B37", "B43", "B44", "B53", "B55", "B60"]
book_ids_to_remove = [
    "B01",  # 未实现章节切分
    "B02",  # 未实现章节切分
    "B06",  # Books/PublicDomain 中没有 B06.txt
    "B30",  # Books/PublicDomain 中没有 B30.txt
    "B35",  # 未实现章节切分
    "B36",  # 未实现章节切分
    "B40",  # 未实现章节切分
    "B45",  # Books/PublicDomain 中没有 B45.txt
    "B47",  # 未实现章节切分
    "B48",  # 内容太长，予以舍弃
    "B57",  # 未实现章节切分
    "B61",  # 未实现章节切分
]
question_ids_to_remove = []
for book_id in book_ids_to_remove:
    if book_id in BOOK_IDS:
        BOOK_IDS.remove(book_id)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
llm: LLM = get_llm('gemini2.0', api_key=api_key)
test_aspects = ['all']
test_complexity = []

# 配置日志输出
logging.basicConfig(
    level=logging.WARNING,  # 或 logging.INFO 查看更多信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('error.log')  # 同时保存到文件
    ]
)


def question_transform(question: str, llm: LLM) -> str:
    # prompt: str = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel. However, the novel is too long, so I will give the novel chapter by chapter, and you need to transform the question for each chapter. You should make sure the user will be able to get the answer of the original question with only the answers of the transformed quesions for each chapter. Your output should only include the transformed question, and the transformed question should be wrapped in two special tokens: <answer>, </answer>. For example, if the question is '<answer>How many times has Alice mentioned in the novel?', the transformed question may be 'Is Alice mentioned in this chapter? If so, how many times has Alice mentioned in this chapter?</answer>', if the question is 'Which chapter mentions Alice.', the transformed question may be '<answer>Is Alice mentioned in this chapter?</answer>', if the question is 'When Jane Eyre met Mr. Lloyd for the first time, what's her feeling towards him?', the transformed question should be '<answer>If Jane Eyre met Mr. Lloyd in this chapter? If so, what's her feeling towards him in the first meeting?</answer>'. You should give only one best transformed question, and not output anything else.\nThe given question is {question}."""
    prompt = build_transform_question_prompt(question)
    # print("Original question:", question)
    response = llm.generate(prompt)
    # print("Response:", response)
    match = re.search(r'<answer>(.*?)</answer>', response)
    if match:
        transformed_question = match.group(1)
    else:
        print(f"Warning: No <answer> tags found in response for question: {question}")
        transformed_question = response.replace("<answer>", "").replace("</answer>", "")
    # transformed_question = response.replace("<answer>", "").replace("</answer>", "")
    print("Transformed question:", transformed_question)
    return transformed_question


def remove_invisible_chars(s):
    return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))


def get_chapter_contents_cached(chapterizer, book_id: str, use_cache: bool = True, cache_dir: str = './cache') -> tuple:
    """
    检查缓存文件是否存在，若存在则直接读取 chapterizer 的结构数据，否则调用 chapterizer.get_chapter_contents() 并保存到文件中
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{book_id}_chapters.json")
    if os.path.exists(cache_file) and use_cache:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        structure_dict = data.get("structure_dict")
        titles = data.get("titles")
    else:
        structure_dict, titles = chapterizer.get_chapter_contents()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"structure_dict": structure_dict, "titles": titles}, f, ensure_ascii=False, indent=4)
    return structure_dict, titles


def is_answered(book_id: str, question_id: str, output_dir: str = './outputs/reduce/selected') -> bool:
    outfile = os.path.join(output_dir, f"{book_id}.json")
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as file:
            questions = json.load(file)
        if question_id in questions and 'ModelAnswer' in questions[question_id] and questions[question_id]['TransformedQuestion'] != "":
            return True
    return False


def reduce_test(book_id: str, test_question_id: str, llm: LLM, output_dir:str = './outputs/reduce', use_cache: bool = True, cache_dir: str = './cache') -> dict:
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
        if test_question_id != 'all' and question_id != test_question_id:
            continue
        if question_id in question_ids_to_remove:
            print(f"Question {question_id} removed, skipping.\n")
            continue
        if skip_answered is True and is_answered(book_id, question_id, output_dir=output_dir):
            print(f"Question {question_id} already answered, skipping.\n")
            continue
        question = question_loader.get_by_id(question_id)
        if 'all' not in test_aspects and 'all' not in test_complexity:
            if question.get_aspect() not in test_aspects and question.get_complexity() not in test_complexity:
                print(f"Skipping question {question_id}, aspect: {question.get_aspect()}, complexity: {question.get_complexity()}\n")
                continue
        print(f"{question_id}: {question.get_question_options()}")
        transformed_question = question_transform(question.get_question_str(), llm)
        chapterizer = Chapterizer(book_content, book_title)
        prompt_final = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel, and a series of answers. The answers are to the question {transformed_question} for each chapter of the novel. You need to give the answer to the question based on the given answers. The following are the answers to the transformed question for each chapter. """
        structure_dict, titles = get_chapter_contents_cached(chapterizer, book_id, use_cache=use_cache, cache_dir=cache_dir)  # 使用缓存函数获取章节结构数据
        chapter_answers = []
        i = 1
        for title in tqdm(titles, desc=f"Processing chapters for {book_id}", leave=False):
            title_desc = title.split('_')[-1]
            for t in title.split('_')[:-1]:
                title_desc +=' of '+ t
            # print(title_desc)
            content = structure_dict[title]
            prompt_chapter = build_prompt_icl(content, transformed_question)
            answer_chapter = llm.generate(prompt_chapter)
            # print(answer_chapter)
            # prompt_final += f"""The answer to chapter {title_desc} is: {answer_chapter}.\n"""
            # chapter_answers.append(f"The answer to the chapter {title_desc} is {answer_chapter}. ")
            prompt_final += f"""The answer to chunk {i} is: {answer_chapter}.\n"""
            chapter_answers.append(f"The answer to the chunk {i} is {answer_chapter}. ")
            i += 1
        
        # print(prompt_final)
        prompt_final += build_prompt_final(question.get_question_options())
        answer_final = llm.generate(prompt_final)
        # print(answer_final)
        llm_option = extract_option(answer_final)
        is_correct = question.get_answer() == llm_option
        
        question_dict[question_id]["ModelAnswer"] = llm_option
        question_dict[question_id]["Correct"] = is_correct
        question_dict[question_id]["Analysis"] = answer_final
        question_dict[question_id]["ChapterAnswers"] = chapter_answers
        question_dict[question_id]["TransformedQuestion"] = transformed_question
        print(f"Question {question_id} - Correct Answer: {question.get_answer()} - Model Answer: {llm_option}, Correct: {is_correct}")
        prompt_final = ""
        save_results_by_question(question_dict, book_id, question_id, output_dir=output_dir)

    return question_dict


def save_results(results: dict, book_id: str, output_dir: str = "./outputs/reduce"):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{book_id}.json", 'w+', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_dir}/{book_id}.json")


def save_results_by_question(results: dict, book_id: str, question_id: str, output_dir: str = "./outputs/reduce"):
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


def process_book(book_id: str, question_id: str, model_name: str = 'gemini', output_dir: str = './outputs/reduce', use_cache: bool = True, cache_dir: str = './cache'):
    try:
        # 在子进程中重新初始化 llm
        api_key = os.environ.get("GEMINI_API_KEY")
        llm = get_llm(model_name, api_key=api_key)
        reduce_test(book_id, question_id, llm, output_dir=output_dir, use_cache=use_cache, cache_dir=cache_dir)
        return f"Book {book_id} processed successfully."
    except KeyboardInterrupt:
        # 捕获键盘中断异常，退出进程
        raise
    except Exception as e:
        logging.error(f"Error processing {question_id} of {book_id}: {e}", exc_info=True)
        return f"Error processing book {book_id}: {e}"


if __name__ == "__main__":
    # output_dir = "./outputs/reduce"
    output_dir = args.output_dir
    use_cache = args.use_cache
    cache_dir = args.cache_dir
    os.makedirs(output_dir, exist_ok=True)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
    # # BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
    # for book_id in tqdm(BOOK_IDS, desc="Processing books"):
    #     if args.book_id != 'all' and book_id != args.book_id:
    #         continue
    #     results = reduce_test(book_id, args.question_id, llm, output_dir=output_dir, use_cache=use_cache, cache_dir=cache_dir)
        # save_results(results, book_id, output_dir=output_dir)
    # 设置多进程池
    max_workers = min(len(BOOK_IDS), args.max_workers)  # 使用 CPU 核心数或书本数中的较小值
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交每本书的任务到进程池
        futures = {
            executor.submit(process_book, book_id, args.question_id, 'gemini2.0', output_dir, use_cache, cache_dir): book_id
            for book_id in BOOK_IDS
            if args.book_id == 'all' or book_id == args.book_id
        }

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing books"):
            book_id = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                logging.error(f"Error processing book {book_id}: {e}", exc_info=True)
                print(f"Error processing book {book_id}: {e}")