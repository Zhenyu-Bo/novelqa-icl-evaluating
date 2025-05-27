from src.llm import LLM, get_llm
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader, QuestionLoader, BookMetaDataLoader
from src.splitter import LLMSplitter  # Changed from Chapterizer
from src.extractor import extract_option, split_reasoning_answer
from src.prompt import build_transform_question_prompt, build_prompt_icl, build_prompt_final, build_prompt_icl2, build_prompt_icl_json, build_prompt_final_json
import argparse
import os
import json
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--book_id", type=str, default="all")
parser.add_argument("--question_id", type=str, default="all")
parser.add_argument("--model", type=str, default="gemini2.0")
parser.add_argument("--max_workers", type=int, default=4, help="Number of worker processes to use")
parser.add_argument("--use_cache", action="store_true", help="Whether to use cache for chapter contents")
parser.add_argument("--no_use_cache", action="store_false", dest="use_cache", help="Whether to use cache for chapter contents")
parser.set_defaults(use_cache=True)
parser.add_argument("--cache_dir", type=str, default="./cache/llm/splitter", help="Directory to store cache files for old chapterizer (if any part still uses it)")
parser.add_argument("--splitter_cache_dir", type=str, default="./cache/llm/splitter", help="Directory to store cache files for LLMSplitter chunks")
parser.add_argument("--output_dir", type=str, default="./outputs/reduce/selected/splitter/gemini", help="Directory to store output files")
parser.add_argument("--skip_answered", action="store_true", help="Skip already answered questions")
parser.add_argument("--no_skip_answered", action="store_false", dest="skip_answered", help="Do not skip already answered questions")
parser.set_defaults(skip_answered=True)

# LLMSplitter specific arguments
parser.add_argument("--splitter_chunk_tokens", type=int, default=30000, help="Target token size for LLMSplitter output chunks.")
parser.add_argument("--splitter_max_llm_tokens", type=int, default=600000, help="Max tokens for LLMSplitter's internal LLM calls (if its method uses LLM for splitting).")
parser.add_argument("--splitter_chunk_overlap", type=int, default=200, help="Chunk overlap for LLMSplitter's internal text splitting.")
parser.add_argument("--splitter_min_merge_tokens", type=int, default=500, help="Min token size for a chunk to avoid merging in LLMSplitter.")
parser.add_argument("--splitter_split_method", type=str, default="generate_chunks_by_boundaries",
                    choices=["generate_chunks_by_boundaries", "generate_chunks_directly", "generate_chunks_by_chapter_markers"],
                    help="Method LLMSplitter uses to generate chunks.")

args = parser.parse_args()
skip_answered = args.skip_answered
model_name = args.model
use_cache = args.use_cache

NOVELQA_PATH = '../NovelQA'
path_builder = NovelQAPathBuilder(NOVELQA_PATH)

# BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]  # 从 B00 到 B62 的书籍
BOOK_IDS = ["B00", "B05", "B09", "B13", "B14", "B16", "B17", "B20", "B22", "B24",
            "B25", "B29", "B33", "B34", "B37", "B43", "B44", "B53", "B55", "B60"]
book_ids_to_remove = [
    "B06",  # Books/PublicDomain 中没有 B06.txt
    "B30",  # Books/PublicDomain 中没有 B30.txt
    "B45",  # Books/PublicDomain 中没有 B45.txt
]

question_ids_to_remove = []
for book_id in book_ids_to_remove:
    if book_id in BOOK_IDS:
        BOOK_IDS.remove(book_id)

load_dotenv()

test_aspects = ['all']
test_complexity = []

# 设置日志记录
logging.basicConfig(
    filename="error.log",  # 日志文件路径
    level=logging.ERROR,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
)


def question_transform(question: str, llm: LLM) -> str:
    prompt = build_transform_question_prompt(question)
    response = llm.generate(prompt)
    transformed_question = split_reasoning_answer(response)[1]
    transformed_question = transformed_question.replace("<answer>", "").replace("</answer>", "")
    print("Transformed question:", transformed_question)
    return transformed_question


def get_llm_splitter_chunks(
    llm_splitter: LLMSplitter,
    book_id: str,
    use_cache: bool = True,
    cache_dir: str = './cache/splitter',
    split_method_name: str = "generate_chunks_by_boundaries"
) -> list:
    """
    Checks if cached chunks from LLMSplitter exist, otherwise generates and caches them.
    Cache filename includes book_id, model_name (of splitter's LLM), target_chunk_tokens, and split_method.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Construct a unique cache filename based on relevant parameters
    cache_file = os.path.join(cache_dir, f"{book_id}.json")

    if os.path.exists(cache_file) and use_cache:
        print(f"Loading LLMSplitter chunks from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print(f"Generating LLMSplitter chunks for {book_id} using method '{split_method_name}' (Cache file: {cache_file})")
        if split_method_name == "generate_chunks_by_boundaries":
            chunks = llm_splitter.generate_chunks_by_boundaries()
        elif split_method_name == "generate_chunks_directly":
            chunks = llm_splitter.generate_chunks_directly()
        elif split_method_name == "generate_chunks_by_chapter_markers":
            chunks = llm_splitter.generate_chunks_by_chapter_markers()
        else:
            raise ValueError(f"Unknown LLMSplitter split_method_name: {split_method_name}")

        print(f"Saving LLMSplitter chunks to cache: {cache_file}")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
        llm_splitter.save_chunks(output_dir=f"{cache_dir}/{book_id}", split_method_name=split_method_name)
    print(f"LLMSplitter chunks for {book_id} loaded/generated, total chunks: {len(chunks)}")
    return chunks


def is_answered(book_id: str, question_id: str, output_dir: str = './outputs/reduce/selected') -> bool:
    outfile = os.path.join(output_dir, f"{book_id}.json")
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as file:
            questions = json.load(file)
        if question_id in questions and 'ModelAnswer' in questions[question_id] and questions[question_id]['TransformedQuestion'] != "":
            return True
    return False


def reduce_test(
    book_id: str,
    test_question_id: str,
    llm: LLM,
    output_dir: str = './outputs/reduce',
    use_cache: bool = True,
    llm_splitter_cache_dir: str = './cache/llm_splitter_chunks',
    splitter_chunk_tokens: int = 60000,
    splitter_max_llm_tokens: int = 600000,
    splitter_chunk_overlap: int = 2000,
    splitter_min_merge_tokens: int = 2000,
    splitter_split_method: str = "generate_chunks_by_boundaries"
) -> dict:
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

        # Instantiate LLMSplitter
        llm_splitter = LLMSplitter(
            llm=llm,  # The same LLM instance can be used for splitting
            book_content=book_content,
            book_title=book_title,
            chunk_tokens=splitter_chunk_tokens,
            max_llm_tokens=splitter_max_llm_tokens,
            chunk_overlap=splitter_chunk_overlap,
            min_chunk_tokens_for_merge=splitter_min_merge_tokens
        )

        prompt_final = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel, and a series of answers. The answers are to the question "{transformed_question}" for each chunk of the novel. You need to give the answer to the original question "{question.get_question_str()}" based on the given answers. The following are the answers to the transformed question for each chunk. """

        # Get chunks using LLMSplitter with caching
        chunks = get_llm_splitter_chunks(
            llm_splitter,
            book_id,
            use_cache=use_cache,
            cache_dir=llm_splitter_cache_dir,
            split_method_name=splitter_split_method
        )

        if not chunks:
            logging.error(f"Book {book_id}, Question {question_id}: LLMSplitter returned no chunks. Skipping question.")
            print(f"LLMSplitter returned no chunks for book {book_id}. Skipping question {question_id}.\n")
            question_dict[question_id]["TransformedQuestion"] = transformed_question
            question_dict[question_id]["ModelAnswer"] = "ERROR: No chunks to process"
            question_dict[question_id]["Correct"] = False
            save_results_by_question(question_dict, book_id, question_id, output_dir=output_dir)
            return question_dict

        chapter_answers = []
        chapter_reasoning = []

        for i, chunk_content in enumerate(tqdm(chunks, desc=f"Processing chunks for {book_id} Q:{question_id}", leave=False)):
            chunk_desc = f"Chunk {i+1}"
            prompt_chapter = build_prompt_icl(chunk_content, transformed_question)
            answer_chapter_raw = llm.generate(prompt_chapter)

            reasoning_chapter, answer_chapter = split_reasoning_answer(answer_chapter_raw)
            prompt_final += f"""The answer for {chunk_desc} is: {answer_chapter}.\n"""
            chapter_answers.append(f"{chunk_desc}: {answer_chapter}")
            if reasoning_chapter is not None:
                chapter_reasoning.append(f"{chunk_desc}: {reasoning_chapter}")

        prompt_final += build_prompt_final(question.get_question_options())
        llm_answer_raw = llm.generate(prompt_final)
        reasoning, answer_final = split_reasoning_answer(llm_answer_raw)
        llm_option = extract_option(answer_final)
        is_correct = question.get_answer() == llm_option

        question_dict[question_id]["TransformedQuestion"] = transformed_question
        question_dict[question_id]["ModelAnswer"] = llm_option
        question_dict[question_id]["Correct"] = is_correct
        if reasoning is not None:
            question_dict[question_id]["Reasoning"] = reasoning
        question_dict[question_id]["Analysis"] = answer_final
        question_dict[question_id]["ChapterAnswers"] = chapter_answers
        if chapter_reasoning is not None:
            question_dict[question_id]["ChapterReasoning"] = chapter_reasoning
        print(f"Book: {book_id} - Question {question_id} - Correct Answer: {question.get_answer()} - Model Answer: {llm_option}, Correct: {is_correct}")
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


def process_book(
    book_id: str,
    question_id_arg: str,
    model_name_arg: str,
    api_key_arg: str,
    output_dir_arg: str,
    use_cache_arg: bool,
    llm_splitter_cache_dir_arg: str,
    splitter_chunk_tokens_arg: int,
    splitter_max_llm_tokens_arg: int,
    splitter_chunk_overlap_arg: int,
    splitter_min_merge_tokens_arg: int,
    splitter_split_method_arg: str
) -> str:
    try:
        # 在子进程中重新初始化 llm
        llm = get_llm(model_name_arg, api_key=api_key_arg)
        reduce_test(
            book_id,
            question_id_arg,
            llm,
            output_dir=output_dir_arg,
            use_cache=use_cache_arg,
            llm_splitter_cache_dir=llm_splitter_cache_dir_arg,
            splitter_chunk_tokens=splitter_chunk_tokens_arg,
            splitter_max_llm_tokens=splitter_max_llm_tokens_arg,
            splitter_chunk_overlap=splitter_chunk_overlap_arg,
            splitter_min_merge_tokens=splitter_min_merge_tokens_arg,
            splitter_split_method=splitter_split_method_arg
        )
        return f"Book {book_id} processed successfully."
    except KeyboardInterrupt:
        # 捕获键盘中断异常，退出进程
        raise
    except Exception as e:
        logging.error(f"Error processing {question_id_arg} of {book_id}: {e}", exc_info=True)
        return f"Error processing book {book_id}: {e}"


if __name__ == "__main__":
    output_dir = args.output_dir
    use_cache = args.use_cache
    llm_splitter_cache_dir = args.splitter_cache_dir

    if "gemini" in model_name.lower():
        api_key = os.getenv("GEMINI_API_KEY")
    elif "deepseek" in model_name.lower():
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        api_key = None

    os.makedirs(output_dir, exist_ok=True)
    if use_cache:
        os.makedirs(llm_splitter_cache_dir, exist_ok=True)
    print("Model name:", model_name)
    print(f"Using LLMSplitter with method: {args.splitter_split_method}, target chunk tokens: {args.splitter_chunk_tokens}")

    max_workers = min(len(BOOK_IDS), args.max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_book,
                book_id,
                args.question_id,
                model_name,
                api_key,
                output_dir,
                use_cache,
                llm_splitter_cache_dir,
                args.splitter_chunk_tokens,
                args.splitter_max_llm_tokens,
                args.splitter_chunk_overlap,
                args.splitter_min_merge_tokens,
                args.splitter_split_method
            ): book_id
            for book_id in BOOK_IDS
            if args.book_id == 'all' or book_id == args.book_id
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing books"):
            book_id = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                logging.error(f"Error processing book {book_id}: {e}", exc_info=True)
                print(f"Error processing book {book_id}: {e}")