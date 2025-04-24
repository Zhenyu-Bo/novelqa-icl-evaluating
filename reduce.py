from src.llm import LLM, get_llm
from src.path_builder import NovelQAPathBuilder
from src.loader import BookLoader, QuestionLoader
from src.utils import load_json, save_json
from src.chapterizer import Chapterizer
import unicodedata
import os

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

def fill_content(structure: dict, book_content: str) -> dict:
    if len(structure['structures']) == 0:
        structure.pop('structures')
        structure['content'] = book_content
        return structure
    lines = book_content.split('\n')
    lines = [remove_invisible_chars(line) for line in lines]
    while len(lines) > 0 and not lines[0].strip():
        lines.pop(0)
    assert lines[0].startswith(structure['title']), f"{lines[0]}\n{structure['title']}"
    lines.pop(0)
    current_structure_idx = 0
    structures = structure['structures']
    structure_contents = [structure['title']]
    while len(lines) > 0:
        while len(lines) > 0 and (current_structure_idx == len(structure['structures']) or lines[0].strip().lower() != structures[current_structure_idx]['title'].strip().lower()):
            structure_contents[-1] += '\n' + lines.pop(0)
        if len(lines):
            structure_contents.append(lines.pop(0))
        current_structure_idx += 1
    assert current_structure_idx == len(structures) + 1, f"{current_structure_idx} {len(structures)}, {structures}"
    structure['content'] = structure_contents[0]
    for i in range(1, len(structure_contents)):
        structure['structures'][i - 1] = fill_content(structure['structures'][i - 1], structure_contents[i])
    return structure

llm: LLM = get_llm('deepseek')

for book_id in BOOK_IDS:
    # break
    if not book_id == 'B02':
        continue
    book_path = path_builder.get_book_path(book_id)
    book_loader = BookLoader(book_path, book_id)
    book_loader.load()
    book_content = book_loader.get_content()
    question_path = path_builder.get_question_path(book_id)
    question_loader = QuestionLoader(question_path, book_id)
    question_loader.load()
    question = question_loader.get_by_id('Q0205')
    transformed_question = question_transform(question.get_question_str(), llm)
    structure = load_json(f'structures/{book_id}.json')
    chapterizer = Chapterizer(book_content, book_id, [])
    chapterizer.structure_from_nocontent_structure(structure)
    prompt_final = f"""You are a helpful assistant. I will give you a question, which is relevant to a novel, and a series of answers. The answers are to the question {transformed_question} for each chapter of the novel. You need to give the answer to the question based on the given answers. Here is the question: {question.get_question_options()}, and the following are the answers for each chapter. """
    for title, content in chapterizer.get_chapter_contents().items():
        titles = title.split('_')
        title_desc = titles[-1]
        for t in titles[:-1]:
            title_desc += ' of ' + t
        prompt_chapter = build_prompt_icl(content, transformed_question)
        answer_chapter = llm.generate(prompt_chapter)
        print(answer_chapter)
        prompt_final += f"""The answer to the chapter {title_desc} is {answer_chapter}. """
    print(prompt_final)
    prompt_final += f"""Now give your analysis and then the best choice of the original question."""
    answer_final = llm.generate(prompt_final)
    print(answer_final)
import sys; sys.exit(0)


if __name__ == '__main__':
    llm: LLM = get_llm('deepseek')
    question = r"""Who speaks with an Australian accent?"""
    question_transform(question, llm)
