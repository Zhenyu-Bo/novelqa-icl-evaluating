from src.llm import LLM, get_llm
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader, QuestionLoader, BookMetaDataLoader
from src.chapterizer import Chapterizer
import unicodedata
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--book_id", type=str, default="B00")
parser.add_argument("--question_id", type=str, default="Q0756")
args = parser.parse_args()

def question_transform(question: str, llm: LLM) -> str:
    prompt: str = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel. However, the novel is too long, so I will give the novel chapter by chapter, and you need to transform the question for each chapter. You should make sure the user will be able to get the answer of the original question with only the answers of the transformed quesions for each chapter. For example, if the question is 'How many times has Alice mentioned in the novel?', the transformed question may be 'Is Alice mentioned in this chapter? If so, how many times has Alice mentioned in this chapter?', if the question is 'Which chapter mentions Alice.', the transformed question may be 'Is Alice mentioned in this chapter?', if the question is 'When Jane Eyre met Mr. Lloyd for the first time, what's her feeling towards him?', the transformed question should be 'If Jane Eyre met Mr. Lloyd in this chapter? If so, what's her feeling towards him in the first meeting?'. You output should be the following format: {{"question": "the transformed question"}}.You should give only one best transformed question, and not output anything else.\nThe given question is {question}."""
    print(question)
    response = llm.generate(prompt)
    print(response)
    transformed_question = response.split('"question": "')[1].split('"')[0]
    print(transformed_question)
    return transformed_question

def build_prompt_icl(chapter_content: str, question_options: str) -> str:
    """创建提示词
    基本的思路是，让模型分析问题，给出回答和对应的证据
    """
    return f"""You are a literature professor. I will provide you with the full text of a chapter from a novel along with a question. Please thoroughly analyze the chapter's content to accurately respond to the following question.\nChapter Content:{chapter_content};\nBook ends. Questions start here:\n{question_options}\nQuestions end here. Try your best to answer the question based on the given full text of the chapter. The answer should be the analysis of text content around the question with the evidence from the chapter, and the answer."""


path_builder = NovelQAPathBuilder('./data/NovelQA')

def remove_invisible_chars(s):
    return ''.join(c for c in s if unicodedata.category(c) not in ('Cc', 'Cf'))

BOOK_IDS = [f"B{i:02}" for i in range(0, 63)]
BOOK_IDS.remove("B06")
BOOK_IDS.remove("B30")
BOOK_IDS.remove("B45")
BOOK_IDS.remove("B48") # 内容太长，予以舍弃

llm: LLM = get_llm('deepseek')

for book_id in BOOK_IDS:
    if args.book_id != 'all' and book_id != args.book_id:
        continue
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
    for question_id, question in question_loader.get_whole().items():
        if args.question_id!= 'all' and question_id!= args.question_id:
            continue
        question = question_loader.get_by_id(question_id)
        transformed_question = question_transform(question.get_question_str(), llm)
        chapterizer = Chapterizer(book_content, book_title)
        structure = chapterizer.get_structure()
        prompt_final = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel, and a series of answers. The answers are to the question {transformed_question} for each chapter of the novel. You need to give the answer to the question based on the given answers. Here is the original question: {question.get_question_options()}, and the following are the answers to the transformed question for each chapter. """
        structure_dict, titles = chapterizer.get_chapter_contents()
        for title in titles:
            title_desc = title.split('_')[-1]
            for t in title.split('_')[:-1]:
                title_desc +=' of '+ t
            print(title_desc)
            content = structure_dict[title]
            prompt_chapter = build_prompt_icl(content, transformed_question)
            answer_chapter = llm.generate(prompt_chapter)
            print(answer_chapter)
            prompt_final += f"""The answer to the chapter {title_desc} is {answer_chapter}. """
        print(prompt_final)
        prompt_final += f"""Now give your analysis and then the best choice of the original question. Note that you should also reexamize each answer and evidences to the transformed question rather than directly use them."""
        answer_final = llm.generate(prompt_final)
        print(answer_final)
