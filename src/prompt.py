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
    3. If the question involves a sequence or time constraint (e.g., "for the first time"), ensure the transformed question asks for time or sequence-related details.
    4. Your output must only include the transformed question, wrapped in the special tokens `<answer>` and `</answer>`.

    Examples:
    - Original question: "How many times has Alice been mentioned in the novel?"
    Your output may be: "<answer>How many times has Alice been mentioned in this chapter?</answer>"
    - Original question: "Which chapter mentions Alice?"
    Your output may be: "<answer>Does this chapter mention Alice?</answer>"
    - Original question: "Please list 3 aliases or designations of Valeria Brinton."
    Your output may be: "<answer>List all direct and explicit evidence that Valeria Brinton has aliases or designations in this chapter.</answer>"
    - Original question: "When Jane Eyre met Mr. Lloyd for the first time, what was her feeling towards him?"
    Your output may be: "<answer>Does Jane Eyre meet Mr. Lloyd in this chapter? If so, what was Jane Eyre's feelings towards Mr. Lloyd during their first meeting in this chapter? Additionally, if available, provide the context of their first meeting, including any time or sequence-related details.</answer>"

    The given question is: {question}.
    Now generate the transformed question.
    """
    return prompt


def build_prompt_icl(chapter_content: str, question_options: str) -> str:
    """创建提示词
    基本的思路是，让模型分析问题，给出回答和对应的证据
    """
    # return f"""You are a literature professor. I will provide you with the full text of a chapter from a novel along with a question. Please thoroughly analyze the chapter's content to accurately respond to the following question.\nChapter Content:{chapter_content};\nBook ends. Questions start here:\n{question_options}\nQuestions end here. Try your best to answer the question based on the given full text of the chapter. The answer should be the analysis of text content around the question with the evidence from the chapter, and the answer."""
    prompt = f"""
    You are a literature professor specializing in analyzing novels. I will provide you with the full text of a chapter from a novel and a question. Your task is to thoroughly analyze the chapter's content and provide an accurate and well-supported answer to the question.

    Please follow these steps:
    1. Carefully analyze the question to understand what is being asked.
    2. Identify and list all evidence from the chapter that is directly, explicitly, and unambiguously relevant to the question. **You can just say "No relevant evidence found in this chapter." if you cannot find any direct and explicit evidence**, so do not over-interpret, speculate, or use implied or ambiguous content as evidence just to find evidence! Do not include evidence that merely hints, suggests, or implies the answer. Only use evidence that clearly and unambiguously states the fact.
    3. For each piece of evidence, explain why it is directly and explicitly relevant to the question and how it supports a possible answer. Do not use evidence that only hints or suggests the answer.
    4. For counting questions (e.g., "How many times does xxx appear?"):
       - Only count events that are directly and explicitly described in the text. Do not count indirect mentions, implications, or references made by other characters.
       - For each counted event, provide the exact quote or description from the text that directly supports the count.
       - If multiple pieces of evidence refer to the same event, treat them as one occurrence and explain your reasoning for deduplication. For example, if the question is "How many times do xxx and yyy communicated?", you should count a series of consecutive conversions between xxx and yyy as one, do not count each conversion separately and do not divide the entire conversation into multiple parts.
    5. Before answering, review all the evidence you have listed and re-examine whether each one is clearly and directly relevant to the question. Remove any evidence that is not obviously and explicitly relevant.
    6. Based only on the remaining evidence, reason step by step and provide a direct and precise answer to the question if possible.
    7. Do not use information from other chapters or external sources.
    
    Format your output as follows:
    - Question Analysis: [Your analysis of the question]
    - Evidence:
      - [Quote or description 1] — [Explanation of its relevance]
      - [Quote or description 2] — [Explanation of its relevance]
      - ...
      - If no relevant evidence is found, write: "No relevant evidence found in this chapter."
    - Deduplication Reasoning (optional): [Your reasoning for deduplication, if any. Omit this step if there is no repetition.]
    - Summary Analysis: [Summarize and synthesize the evidence from all chapters, explain your reasoning step by step]
    - Answer: [Your direct response based only on the above evidence(for counting questions, avoid double counting the evidence if they refer to the same event)]

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
    Here is the original question which is related to the whole novel: {question}.
    
    Now analyze and select the best answer to the original question.
    
    Please strictly follow the steps below:

    1. Carefully review the answers and evidence for each chapter.
       - For each chapter, check whether the evidence is directly, explicitly, and unambiguously relevant to the question and truly supports the answer.
       - Exclude evidences that merely hints, suggests, or implies the answer. Only use evidence that clearly and unambiguously states the fact.

    2. After re-examination, collect the remaining relevant evidence from the chapters that supports your final answer.

    3. Before giving your final answer, list all supporting evidence:
       - Indicate the chapter(s) where the evidence is found.
       - Provide reference to the original text
       - Explain how each piece of evidence supports your final answer.

    4. For counting questions (e.g., "How many times does xxx appear?"):
       - Only count events that are directly and explicitly described in the text. Do not count indirect mentions, implications, or references made by other characters.
       - Analyze whether different pieces of evidence, including those from different chapters, refer to the same event or unique occurrences.
       - If evidence from different chapters or within the same chapter refers to the same event, avoid double-counting and clearly explain your deduplication reasoning.

    5. Based only on the remaining, clearly relevant evidence, synthesize the information and provide a precise, well-supported final answer. Do not use any external knowledge, assumptions, or information not explicitly given.

    6. If you cannot find a suitable answer among the given options:
       - Re-examine the logical reasoning in your previous steps to identify any potential errors or gaps.
       - If the question is a counting question, examine whether you have double-counted the evidences that refer to the same event.
       - If necessary, refine your analysis and provide a revised answer based on the updated reasoning.

    Output format (strictly follow this structure):

    1. Chapter Evidence and Analysis:
       - Only include chapters where direct and explicit evidence is found. Omit chapters with no relevant evidence.
       - Chapter X:
         - Evidence: [All direct and explicit evidence from this chapter]
         - Analysis: [Explain the relevance of each evidence]
       - Chapter Y:
         - Evidence: [...]
         - Analysis: [...]
       - ... (repeat for all chapters with evidence)

    2. Deduplication Reasoning (optional): [Your reasoning for deduplication, if any. Omit this step if there is no repetition.]
    
    3. Summary Analysis:
       - [Summarize and synthesize the evidence from all chapters, explain your reasoning step by step]
       
    4. Re-examination:
       - [If you cannot find a suitable answer from the given options based on the above analysis, re-examine your logical reasoning in the previous steps to identify any potential errors or gaps. If the question is a counting question, examine whether you have double-counted the evidences that refer to the same event. If so, you should count them as one occurrence. If necessary, refine your analysis and provide a revised answer based on the updated reasoning.]
       
    5. Final Answer Explanation:
       - [Give your detailed answer based on the above analysis and evidence.]
       
    6. Final Choice:
       <answer>my final answer: A, B, C, or D</answer>(Notice you can only choose one option, for example: <answer>my final answer: A</answer>)

    Example:
    1. Chapter Evidence and Analysis:
       - Chapter 1:
         - Evidence: "Quote1"
         - Analysis: "This directly shows..."
       - Chapter 2:
         - Evidence: "Quote2"
         - Analysis: "This is not relevant because..."
    2. Deduplication Reasoning: The evidence ... from Chapter 1 and ... from Chapter 2 refer to the same event, so I only count it once...
    3. Summary Analysis:
       - "Based on the above, the total times is ...(try to give a precise answer if possible)."
    4. Final Answer Explanation:
       - [Give your detailed answer based on the above analysis and evidence.]
    5. Final Choice:
       <answer>my final answer: A</answer>
    """
    
    return prompt


def build_transform_question_prompt2(question: str) -> str:
    """构建转换问题的提示词
    这个提示词的作用是将问题转换为可以在单个章节中回答的问题"""
    prompt: str = f"""
    You are a helpful assistant. I will give you a question related to a novel. 
    The novel is too long to analyze as a whole, so I will provide the novel chapter by chapter. 
    Your task is to transform the given question into a form that can be answered based on the content of a single chapter. 

    Please follow these rules:
    1. Preserve the logical relationships (e.g., temporal, causal, comparative) in the original question, and ensure the transformed question reflects these relationships.
    2. The transformed question must be specific to a single chapter and should not require information from other chapters.
    3. If the question involves a sequence or time constraint (e.g., "for the first time"), ensure the transformed question asks for time or sequence-related details.
    4. If the original question is about "how many times", "how many", "how often", "please list N...", or asks for a count or a specific number of items, do NOT ask for the number or the count in the transformed question. Instead, ask the model to list all direct and explicit evidence related to the event or entity in this chapter, regardless of the number.
    5. Your output must only include the transformed question, wrapped in the special tokens `<answer>` and `</answer>`.

    Examples:
    - Original question: "How many times has Alice been mentioned in the novel?"
    Your output may be: "<answer>List all direct and explicit evidence that Alice is mentioned in this chapter.</answer>"
    - Original question: "Which chapter mentions Alice?"
    Your output may be: "<answer>Does this chapter mention Alice? If so, provide the relevant evidence.</answer>"
    - Original question: "Please list 3 aliases or designations of Valeria Brinton."
    Your output may be: "<answer>List all direct and explicit evidence that Valeria Brinton has aliases or designations in this chapter.</answer>"
    - Original question: "When Jane Eyre met Mr. Lloyd for the first time, what was her feeling towards him?"
    Your output may be: "<answer>Does Jane Eyre meet Mr. Lloyd in this chapter? If so, what was Jane Eyre's feelings towards Mr. Lloyd during their first meeting in this chapter? Additionally, if available, provide the context of their first meeting, including any time or sequence-related details.</answer>"

    The given question is: {question}.
    Now generate the transformed question.
    """
    return prompt


def build_prompt_icl2(chapter_content: str, question_options: str) -> str:
    """
    创建提示词，要求模型分析问题，找出所有显然相关的证据，避免过度解读，对于次数类问题不需要回答次数，同一事件多次出现视为一次。
    """
    prompt = f"""
    You are a literature professor specializing in analyzing novels. I will provide you with the full text of a chapter from a novel and a question. Your task is to thoroughly analyze the chapter's content and provide an accurate and well-supported answer to the question.

    Please follow these steps:
    1. Carefully analyze the question to understand what is being asked.
    2. Identify and list all evidence from the chapter that is directly, explicitly, and unambiguously relevant to the question. Only include evidence that is clearly and obviously relevant; do not over-interpret, speculate, or use implied or ambiguous content as evidence just to find evidence. If you cannot find any direct and explicit evidence, write "No relevant evidence found in this chapter."
    3. For each piece of evidence, explain why it is directly and explicitly relevant to the question and how it supports a possible answer. Do not use evidence that only hints or suggests the answer.
    4. If the question involves counting occurrences (e.g., "How many times does xxx appear?"), you do NOT need to answer how many times. Just list all direct and explicit evidence related to the event in this chapter. If multiple pieces of evidence refer to the same event, treat them as one occurrence and explain your reasoning for deduplication.
    5. Before answering, review all the evidence you have listed and re-examine whether each one is clearly and directly relevant to the question. Remove any evidence that is not obviously and explicitly relevant.
    6. Based only on the remaining evidence, reason step by step and provide a direct and precise answer to the question if possible.
    7. Do not use information from other chapters or external sources.

    Format your output as follows:
    - Question Analysis: [Your analysis of the question]
    - Evidence:
      - [Quote or description 1] — [Explanation of its relevance]
      - [Quote or description 2] — [Explanation of its relevance]
      - ...
      - If no relevant evidence is found, write: "No relevant evidence found in this chapter."
    - Deduplication Reasoning (optional): [Your reasoning for deduplication, if any. Omit this step if there is no repetition.]
    - Answer: [Your direct response based only on the above evidence. For counting questions, do NOT answer the number of times, just provide the evidence.]

    Here is the chapter content:
    {chapter_content}

    Here is the question:
    {question_options}

    Now provide your answer.
    """

    return prompt


def build_prompt_icl_json(chapter_content: str, question_options: str) -> str:
    prompt = f"""
    You are a literature professor specializing in analyzing novels. I will provide you with the full text of a chapter from a novel and a question. Your task is to thoroughly analyze the chapter's content and provide an accurate and well-supported answer to the question.

    Please follow these steps:
    1. Carefully analyze the question to understand what is being asked.
    2. Identify and list all evidence from the chapter that is directly, explicitly, and unambiguously relevant to the question. Only include evidence that is clearly and obviously relevant; do not over-interpret, speculate, or use implied or ambiguous content as evidence just to find evidence. If you cannot find any direct and explicit evidence, write "No relevant evidence found in this chapter."
    3. For each piece of evidence, explain why it is directly and explicitly relevant to the question and how it supports a possible answer. Do not use evidence that only hints or suggests the answer.
    4. If the question involves counting occurrences (e.g., "How many times does xxx appear?"), you do NOT need to answer how many times. Just list all direct and explicit evidence related to the event in this chapter. If multiple pieces of evidence refer to the same event, treat them as one occurrence and explain your reasoning for deduplication.
    5. Before answering, review all the evidence you have listed and re-examine whether each one is clearly and directly relevant to the question. Remove any evidence that is not obviously and explicitly relevant.
    6. Based only on the remaining evidence, reason step by step and provide a direct and precise answer to the question if possible.
    7. Do not use information from other chapters or external sources.

    Format your output strictly as JSON in the following structure:
    {{
        "question_analysis": "Your analysis of the question",
        "evidence": [
            {{"quote": "Quote or description 1", "explanation": "Explanation of its relevance"}},
            {{"quote": "Quote or description 2", "explanation": "Explanation of its relevance"}}
            // ... more evidence if available
        ],
        "deduplication_reasoning(optional)": "Your reasoning for deduplication, if any",
        "answer": "Your direct response based only on the above evidence. For counting questions, do NOT answer the number of times, just provide the evidence."
    }}

    Example output:
    {{
        "question_analysis": "The question asks whether ...",
        "evidence": [
            {{"quote": "Quote1", "explanation": "This is directly relevant because ..."}},
            {{"quote": "Quote2", "explanation": "This is directly relevant because ..."}}
        ],
        "deduplication_reasoning": "Evidence 1 and 2 refer to the same event, so only counted once.",
        "answer": "Based on the above evidence, ..."
    }}

    Here is the chapter content:
    {chapter_content}

    Here is the question:
    {question_options}

    Now provide your answer in the specified JSON format.
    """
   
    return prompt


def build_prompt_final_json(question: str) -> str:
    prompt = f"""
    Here is the original question which is related to the whole novel: {question}.
    
    Now analyze and select the best answer to the original question.
    
    Please strictly follow the steps below:

    1. Carefully review the answers and evidence for each chapter.
       - For each chapter, check whether the evidence is directly, explicitly, and unambiguously relevant to the question and truly supports the answer.
       - Exclude evidences that merely hints, suggests, or implies the answer. Only use evidence that clearly and unambiguously states the fact.

    2. After re-examination, collect the remaining relevant evidence from the chapters that supports your final answer.

    3. Before giving your final answer, list all supporting evidence:
       - Indicate the chapter(s) where the evidence is found.
       - Provide reference to the original text.
       - Explain how each piece of evidence supports your final answer.

    4. For counting questions (e.g., "How many times does xxx appear?"):
       - Analyze whether different pieces of evidence, including those from different chapters, refer to the same event or unique occurrences.
       - If evidence from different chapters or within the same chapter refers to the same event, avoid double-counting and clearly explain your deduplication reasoning.

    5. Based only on the remaining, clearly relevant evidence, synthesize the information and provide a precise, well-supported final answer. Do not use any external knowledge, assumptions, or information not explicitly given.

    Output format (strictly follow this JSON structure):
    {{
        "question_analysis": "Your analysis of the question",
        "chapter_evidence_and_analysis": [
            {{
                "chapter": "Chapter X",
                "evidence": [
                    "Quote or description 1",
                    "Quote or description 2"
                ],
                "analysis": "Explain the relevance of each evidence"
            }},
            {{
                "chapter": "Chapter Y",
                "evidence": [
                    "Quote or description 3"
                ],
                "analysis": "Explain the relevance of each evidence"
            }}
            // ... repeat for all chapters with evidence
        ],
        "deduplication_reasoning": "Your reasoning for deduplication, if any. Omit this field if there is no repetition.",
        "summary_analysis": "Summarize and synthesize the evidence from all chapters, explain your reasoning step by step",
        "final_answer": "Give your detailed answer based on the above analysis and evidence",
        "final_choice": "<answer>my final answer: A, B, C, or D</answer>(Notice you can only choose one option, for example: <answer>my final answer: A</answer>)"
    }}

    Example output:
    {{
        "question_analysis": "The question asks whether ...",
        "chapter_evidence_and_analysis": [
            {{
                "chapter": "Chapter 1",
                "evidence": [
                    "Quote1",
                    "Quote2"
                ],
                "analysis": "This directly shows ..."
            }},
            {{
                "chapter": "Chapter 2",
                "evidence": [
                    "Quote3"
                ],
                "analysis": "This is not relevant because ..."
            }}
        ],
        "deduplication_reasoning": "Evidence from Chapter 1 and Chapter 2 refer to the same event, so only counted once.",
        "summary_analysis": "Based on the above, the total times is ...",
        "final_answer": "Based on the evidence, the answer is ...",
        "final_choice": "<answer>my final answer: A</answer>"
    }}

    Now provide your answer in the specified JSON format.
    """
    return prompt
