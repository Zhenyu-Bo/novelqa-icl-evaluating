def build_transform_question_prompt(question: str) -> str:
    """构建转换问题的提示词
    这个提示词的作用是将问题转换为可以在单个章节中回答的问题"""
    prompt: str = f"""
    You are a helpful assistant. I will give you a question related to a novel. 
    The novel is too long to analyze as a whole, so I will provide the novel chapter by chapter. 
    Your task is to transform the given question into a form that can be answered based on the content of a single chapter. 

    Please follow these rules:
    1. Preserve the logical relationships (e.g., temporal, causal, comparative) in the original question, and ensure the transformed question reflects these relationships.
    2. The transformed question must be specific to a single chapter and should not require information from other chapters.
    3. Your output must only include the transformed question, wrapped in the special tokens `<answer>` and `</answer>`.

    Examples:
    - Original question: "How many times has Alice been mentioned in the novel?"
    Your output may be: "<answer>How many times has Alice been mentioned in this chapter?</answer>"
    - Original question: "Which chapter mentions Alice?"
    Your output may be: "<answer>Does this chapter mention Alice?</answer>"
    - Original question: "When Jane Eyre met Mr. Lloyd for the first time, what was her feeling towards him?"
    Your output may be: "<answer>When Jane Eyre met Mr. Lloyd for the first time in this chapter, what was her feeling towards him?</answer>"

    The given question is: {question}.
    Now generate the transformed question.
    """
    return prompt


def build_prompt_icl(chapter_content: str, question_options: str) -> str:
    """创建提示词
    基本的思路是，让模型分析问题，给出回答和对应的证据
    """
    # return f"""You are a literature professor. I will provide you with the full text of a chapter from a novel along with a question. Please thoroughly analyze the chapter's content to accurately respond to the following question.\nChapter Content:{chapter_content};\nBook ends. Questions start here:\n{question_options}\nQuestions end here. Try your best to answer the question based on the given full text of the chapter. The answer should be the analysis of text content around the question with the evidence from the chapter, and the answer."""
    prompt =  f"""
    You are a literature professor specializing in analyzing novels. I will provide you with the full text of a chapter from a novel and a question. Your task is to thoroughly analyze the chapter's content and provide an accurate and well-supported answer to the question.
    Please follow these rules:
    1. Base your answer only on the content of the provided chapter. Do not use information from other chapters or external sources.
    2. Your answer must include:
    - A direct response to the question.
    - Evidence from the chapter to support your answer.
    - A brief explanation of how the evidence supports your answer.
    3. Format your output as follows:
    - Answer: [Your direct response]
    - Evidence: [Relevant evidence from the chapter]
    - Explanation: [How the evidence supports your answer]

    Here is the chapter content:
    {chapter_content}

    Here is the question:
    {question_options}

    Now provide your answer.
    """

    return prompt
    

def build_prompt_final(question: str) -> str:
    """创建最终的提示词"""
    prompt = f"""
    Now give your analysis and then the best choice of the original question.
    Note that you should also reexamize each answer and evidences to the transformed question rather than directly use them.
    Follow these steps:
    1. Carefully review the answers and evidence provided for each chapter.
    2. Identify all relevant evidence from the chapters that supports your final answer.
    3. Before giving your final answer, list all the evidence that supports it. For example:
    - Specify the chapter(s) where the evidence is found.
    - Provide quotes, descriptions, or specific details from the chapters that are relevant to the question.
    - Explain how each piece of evidence supports your final answer.
    4. Based on the evidence, synthesize the information and provide a well-supported final answer.

    At the end of your analysis, provide your final answer in the following format:
    <answer>my final answer: A, B, C, or D</answer>

    For example:
    - If your final answer is A, you should output: <answer>my final answer: A</answer>
    """
    
    return prompt