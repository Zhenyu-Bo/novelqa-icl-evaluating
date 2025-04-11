import json
from functools import cached_property
from src.data import QuestionModel

class BookLoader:
    def __init__(self, book_path: str, book_id: str):
        self.book_path = book_path
        self.book_content = None
        self.word_count = None
        self.char_count = None
        self.book_id = book_id

    def load(self):
        with open(self.book_path, 'r') as file:
            self.book_content = file.read()
        self.word_count = len(self.book_content.split())
        self.char_count = len(self.book_content)

    def get_content(self) -> str:
        return self.book_content
    
    def get_char_count(self) -> int:
        return self.char_count

    def get_word_count(self) -> int:
        return self.word_count
    
    def get_id(self) -> str:
        return self.book_id

class QuestionLoader:
    def __init__(self, question_path: str, book_id: str):
        self.question_path = question_path
        self.book_id = book_id
        self.questions = {}
        self.current_question_pointer = 0

    def load(self):
        try:
            with open(self.question_path, 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
                self.current_question_pointer = 0
        except FileNotFoundError:
            print(f"错误：文件 {self.question_path} 未找到")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误：{e}")

    def get_whole(self) -> dict:
        return self.questions

    def get_ith_question(self, i: int) -> QuestionModel:
        if i < len(self.questions):
            return QuestionModel.from_dict(self.questions[self._question_ids[i]], self.book_id, self._question_ids[i])
        else:
            return None

    def get_next_question(self) -> QuestionModel:
        question = self.get_ith_question(self.current_question_pointer)
        if question:
            self.current_question_pointer += 1
        return question
    
    def get_by_id(self, question_id: str) -> QuestionModel:
        if question_id in self.questions:
            return QuestionModel.from_dict(self.questions[question_id], self.book_id, question_id)
        else:
            return None

    @cached_property
    def _question_ids(self):
        return list(self.questions.keys())
    
    def __len__(self):
        return len(self.questions)
    
    def __next__(self):
        question = self.get_next_question()
        if question is None:
            raise StopIteration
        return question
    def __getitem__(self, index: int) -> QuestionModel:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.questions))
            return [self.get_ith_question(i) for i in range(start, stop, step)]
        if index < len(self.questions):
            return self.get_ith_question(index)
        else:
            raise IndexError("Index out of range")
    def __setitem__(self, index: int, value: QuestionModel):
        if index < len(self.questions):
            self.questions[self._question_ids[index]] = value.to_dict()
        else:
            raise IndexError("Index out of range")
    def __delitem__(self, index: int):
        if index < len(self.questions):
            del self.questions[self._question_ids[index]]
        else:
            raise IndexError("Index out of range")
    def __contains__(self, item: str) -> bool:
        return item in self.questions
