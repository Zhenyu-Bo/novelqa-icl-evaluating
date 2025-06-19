import os
import json
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from src.llm import LLM, get_llm
from src.extractor import extract_option
from src.prompt import build_prompt_final
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 设置代理（如果需要）
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('summarize_results.log')
    ]
)

logger = logging.getLogger(__name__)

# 书籍列表和相关配置（从 reduce.py 复制）
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

# 从 BOOK_IDS 中移除要移除的书籍
for book_id in book_ids_to_remove:
    if book_id in BOOK_IDS:
        BOOK_IDS.remove(book_id)

def load_question_data(input_file: str) -> dict:
    """
    从保存的结果文件中加载问题数据
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading file {input_file}: {e}")
        return {}

def extract_question_info(question_data: dict, question_id: str) -> dict:
    """
    从问题数据中提取必要信息
    """
    if question_id not in question_data:
        return None
    
    question_info = question_data[question_id]
    
    # 提取基本信息
    extracted = {
        'question_id': question_id,
        'question': question_info.get('Question', ''),
        'options': question_info.get('Options', {}),
        'correct_answer': question_info.get('Answer', ''),
        'transformed_question': question_info.get('TransformedQuestion', ''),
        'chapter_answers': question_info.get('ChapterAnswers', []),
        'analysis': question_info.get('Analysis', ''),
        'model_answer': question_info.get('ModelAnswer', ''),
        'correct': question_info.get('Correct', False)
    }
    
    return extracted

def format_question_options(options: dict) -> str:
    """
    格式化选项为字符串
    """
    if not options:
        return ""
    
    option_str = ""
    for key, value in sorted(options.items()):
        option_str += f"{key}. {value}\n"
    
    return option_str.strip()

def build_summarization_prompt(question_info: dict) -> str:
    """
    构建用于总结章节回答的提示词
    """
    question = question_info['question']
    transformed_question = question_info['transformed_question']
    chapter_answers = question_info['chapter_answers']
    options = question_info['options']
    
    # 格式化选项
    options_str = format_question_options(options)
    
    # 构建提示词
    prompt = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel, and a series of answers. The answers are to the question "{transformed_question}" for each chapter of the novel. You need to give the answer to the original question "{question}" based on the given answers.

The following are the answers to the transformed question for each chapter:

"""
    
    # 添加章节回答
    for chapter_answer in chapter_answers:
        prompt += f"{chapter_answer}\n"
    
    # 使用 build_prompt_final 函数
    prompt += build_prompt_final(question + "\n" + options_str)
    
    return prompt

def is_answered(book_id: str, question_id: str, output_dir: str) -> bool:
    """
    检查问题是否已经被回答
    """
    outfile = os.path.join(output_dir, f"{book_id}.json")
    if os.path.exists(outfile):
        with open(outfile, 'r', encoding='utf-8') as file:
            questions = json.load(file)
        if question_id in questions and 'ModelAnswer' in questions[question_id] and questions[question_id]['ModelAnswer'] != "" and questions[question_id]['ModelAnswer'] != 'ERROR':
            return True
    return False

def save_results_by_question(results: dict, book_id: str, question_id: str, output_dir: str):
    """
    保存单个问题的结果到文件中。
    如果文件不存在，则直接写入 results
    如果文件存在，则从文件中加载已有数据，对该问题的结果进行更新，然后写回文件
    """
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{book_id}.json")
    
    try:
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
        logger.info(f"Results for question {question_id} saved to {outfile}")
    except Exception as e:
        logger.error(f"Error saving results for question {question_id} to {outfile}: {e}")

def summarize_answers(llm: LLM, question_info: dict) -> dict:
    """
    使用LLM总结章节回答并得出最终答案
    """
    try:
        # 构建提示词
        prompt = build_summarization_prompt(question_info)
        
        # 调用LLM
        max_retries = 1
        retries = 0
        response = None
        while retries < max_retries and response is None:
            logger.info(f"Trying LLM generation ({retries + 1}/{max_retries})...")
            try:
                response = llm.generate(prompt).strip()
                if response:  # 确保响应不为空
                    break
            except Exception as e:
                logger.error(f"Try {retries + 1} failed: {e}")
                retries += 1
                if retries >= max_retries:
                    raise e
        
        if not response:
            raise Exception("Failed to get response from LLM after all retries")
            
        logger.info(f"LLM response: {response[:200]}...")  # 仅记录前200个字符
        
        # 提取选项
        model_answer = extract_option(response)
        
        # 判断是否正确
        is_correct = question_info['correct_answer'] == model_answer
        
        return {
            'prompt': prompt,
            'full_response': response,
            'model_answer': model_answer,
            'correct': is_correct
        }
        
    except Exception as e:
        logger.error(f"Error in summarize_answers for question {question_info['question_id']}: {e}")
        return {
            'prompt': '',
            'full_response': f"ERROR: {str(e)}",
            'model_answer': 'ERROR',
            'correct': False
        }

def process_book_results(book_id: str, input_dir: str, output_dir: str, model_name: str, api_key: str, skip_answered: bool = True) -> dict:
    """
    处理单本书的结果，逐个问题处理并立即保存
    """
    try:
        # 在子进程中重新初始化 llm
        llm = get_llm(model_name, api_key=api_key)
        
        input_file = os.path.join(input_dir, f"{book_id}.json")
        
        if not os.path.exists(input_file):
            logger.warning(f"Input file not found: {input_file}")
            return {}
        
        logger.info(f"Processing book {book_id}")
        
        # 加载原始数据
        question_data = load_question_data(input_file)
        if not question_data:
            return {}
        
        # 统计信息
        processed_count = 0
        correct_count = 0
        
        for question_id in tqdm(question_data.keys(), desc=f"Processing questions for {book_id}"):
            # 检查是否已回答（如果启用了跳过已回答的选项）
            if skip_answered and is_answered(book_id, question_id, output_dir):
                logger.info(f"Question {question_id} already answered, skipping.")
                continue
            
            # 提取问题信息
            question_info = extract_question_info(question_data, question_id)
            if not question_info:
                logger.warning(f"Could not extract info for question {question_id}")
                continue
            
            # 检查是否有章节回答
            if not question_info['chapter_answers']:
                logger.warning(f"No chapter answers found for question {question_id}")
                # 复制原始结果但不处理
                new_result = question_data[question_id].copy()
                new_result['ModelAnswer'] = 'ERROR: No chapter answers'
                new_result['Correct'] = False
                new_result['Analysis'] = 'No chapter answers found for processing'
                
                # 立即保存这个问题的结果
                save_results_by_question({question_id: new_result}, book_id, question_id, output_dir)
                continue
            
            logger.info(f"Summarizing answers for question {question_id}")
            
            # 总结回答
            summary_result = summarize_answers(llm, question_info)
            
            # 构建新的结果结构
            new_result = question_data[question_id].copy()
            new_result['ModelAnswer'] = summary_result.get('model_answer', '')
            new_result['Correct'] = summary_result.get('correct', False)
            new_result['Analysis'] = summary_result.get('full_response', '')
            
            # 立即保存这个问题的结果
            save_results_by_question({question_id: new_result}, book_id, question_id, output_dir)
            
            # 更新统计信息
            processed_count += 1
            if summary_result.get('correct', False):
                correct_count += 1
            
            logger.info(f"Question {question_id} - Correct Answer: {question_info['correct_answer']} - "
                       f"Model Answer: {summary_result['model_answer']} - "
                       f"Correct: {summary_result['correct']}")
        
        # 返回统计信息
        return {
            'processed_count': processed_count,
            'correct_count': correct_count
        }
        
    except KeyboardInterrupt:
        # 捕获键盘中断异常，退出进程
        raise
    except Exception as e:
        logging.error(f"Error processing book {book_id}: {e}", exc_info=True)
        return f"Error processing book {book_id}: {e}"

def process_book(
    book_id: str,
    input_dir: str,
    output_dir: str,
    model_name: str,
    api_key: str,
    skip_answered: bool
) -> str:
    """
    多进程处理书籍的包装函数
    """
    try:
        stats = process_book_results(book_id, input_dir, output_dir, model_name, api_key, skip_answered)
        if isinstance(stats, dict) and 'processed_count' in stats:
            book_questions = stats['processed_count']
            book_correct = stats['correct_count']
            if book_questions > 0:
                accuracy = book_correct / book_questions * 100
                return f"Book {book_id} processed successfully: {book_correct}/{book_questions} correct ({accuracy:.2f}%)"
            else:
                return f"Book {book_id} processed successfully: No questions processed"
        else:
            return f"Book {book_id} processed with errors: {stats}"
    except KeyboardInterrupt:
        # 捕获键盘中断异常，退出进程
        raise
    except Exception as e:
        logging.error(f"Error processing book {book_id}: {e}", exc_info=True)
        return f"Error processing book {book_id}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Summarize chapter answers using LLM with multiprocessing")
    parser.add_argument("--input_dir", type=str, default="./outputs/reduce/selected/prompt3", 
                       help="Directory containing input result files")
    parser.add_argument("--output_dir", type=str, default="./outputs/reduce/selected/gemini2.0-2.5", 
                       help="Directory to save summarized results")
    parser.add_argument("--book_id", type=str, default="all", 
                       help="Specific book ID to process, or 'all' for all books")
    parser.add_argument("--model", type=str, default="gemini2.5-pro", 
                       help="Model to use for summarization")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of worker processes to use")
    parser.add_argument("--skip_answered", action="store_true", help="Skip already answered questions")
    parser.add_argument("--no_skip_answered", action="store_false", dest="skip_answered", help="Do not skip already answered questions")
    parser.set_defaults(skip_answered=True)
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化API key
    if "gemini" in args.model.lower():
        api_key = os.environ.get("GEMINI_API_KEY")
    elif "deepseek" in args.model.lower():
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    else:
        api_key = None
    
    if not api_key:
        logger.error(f"API key not found for model {args.model}")
        return
    
    logger.info(f"Using model: {args.model}")
    
    # 获取要处理的书籍列表
    if args.book_id == "all":
        # 使用预定义的书籍列表，或者从输入目录获取
        if os.path.exists(args.input_dir):
            available_books = []
            for filename in os.listdir(args.input_dir):
                if filename.endswith('.json'):
                    book_id = filename[:-5]  # 移除.json扩展名
                    if book_id in BOOK_IDS:  # 只处理预定义列表中的书籍
                        available_books.append(book_id)
            book_ids = sorted(available_books)
        else:
            book_ids = BOOK_IDS
    else:
        book_ids = [args.book_id]
    
    logger.info(f"Processing {len(book_ids)} books: {book_ids}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 使用多进程处理
    max_workers = min(len(book_ids), args.max_workers)
    logger.info(f"Using {max_workers} worker processes")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交每本书的任务到进程池
        futures = {
            executor.submit(
                process_book,
                book_id,
                args.input_dir,
                args.output_dir,
                args.model,
                api_key,
                args.skip_answered
            ): book_id
            for book_id in book_ids
        }
        
        # 使用 tqdm 显示进度条
        total_questions = 0
        total_correct = 0
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing books"):
            book_id = futures[future]
            try:
                result = future.result()
                logger.info(result)
                
                # 尝试从结果中提取统计信息
                if "correct" in result:
                    # 解析结果字符串中的统计信息
                    import re
                    match = re.search(r'(\d+)/(\d+) correct', result)
                    if match:
                        correct = int(match.group(1))
                        total = int(match.group(2))
                        total_correct += correct
                        total_questions += total
                        
            except Exception as e:
                logging.error(f"Error processing book {book_id}: {e}", exc_info=True)
                logger.error(f"Error processing book {book_id}: {e}")
    
    # 输出总体统计
    if total_questions > 0:
        overall_accuracy = total_correct / total_questions * 100
        logger.info(f"Overall results: {total_correct}/{total_questions} correct ({overall_accuracy:.2f}%)")
    else:
        logger.info("No questions were processed")
    
    logger.info("Processing completed!")

if __name__ == "__main__":
    main()
