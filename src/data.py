class QuestionModel:
    """
    问题数据结构，方便一些操作
    """
    def __init__(self, book_id: str, question_id: str, aspect: str, complexity: str, question: str, options: dict, answer: str):
        self.book_id = book_id
        self.aspect = aspect
        self.complexity = complexity
        self.options = options
        self.question = question
        self.answer = answer
        self.question_id = question_id

    def get_option(self, option: str) -> str:
        """获取某个选项的内容，比如获取选项A的内容，即调用 get_option('A')"""
        if option in self.options:
            return self.options[option]
        else:
            return None

    def get_options_str(self) -> str:
        """获取选项的字符串表示"""
        for option in self.options:
            self.options[option] = f"{option}. {self.options[option]}"
        return '\n'.join(self.options.values())
    
    def get_question_str(self) -> str:
        return self.question
    
    def get_aspect(self) -> str:
        """获取题目方面"""
        return self.aspect
    
    def get_complexity(self) -> str:
        """获取题目难度"""
        return self.complexity

    def get_question_options(self) -> str:
        """获取题目和选项的字符串表示"""
        return f"The Question{self.question_id} is: {self.question}\nThe options are:\n{self.get_options_str()}"

    @classmethod
    def from_dict(cls, data: dict, book_id: str, question_id: str):
        """从字典构造"""
        return cls(
            book_id=book_id,
            question_id=question_id,
            aspect=data['Aspect'],
            complexity=data['Complexity'],
            question=data['Question'],
            options=data['Options'],
            answer=data['Answer']
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'book_id': self.book_id,
            'question_id': self.question_id,
            'aspect': self.aspect,
            'complexity': self.complexity,
            'question': self.question,
            'options': self.options,
            'correct_answer': self.answer
        }
    
    def get_answer(self) -> str:
        """获取答案"""
        return self.answer
    
    def get_id(self) -> str:
        """获取问题id"""
        return self.question_id
