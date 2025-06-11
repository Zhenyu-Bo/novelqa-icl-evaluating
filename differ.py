import os
import json

# 文件夹路径
dir_icl = './outputs/icl'
dir_rag = '../RAG/ReadAgent-RAG/Evaluation/NovelQA/RAG/results'
dir_prompt = './outputs/reduce/selected/prompt'
dir1 = './outputs/reduce/selected/gemini2.0-2.5'
# dir1 = dir_icl
dir2 = dir_prompt + '3'

# 结果列表
wrong_in_2 = []
transform_question_fail = []
total_1 = 0
total_2 = 0
correct_1 = 0
correct_2 = 0

for fname in os.listdir(dir1):
    if not fname.endswith('.json'):
        continue
    path1 = os.path.join(dir1, fname)
    path2 = os.path.join(dir2, fname)
    if not os.path.exists(path1):
        continue
    with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f3:
        data1 = json.load(f1)
        data2 = json.load(f3)
        for qid in data1:
            if data1[qid].get("ModelAnswer") == "ERROR":
                continue
            if qid in data2:
                c1 = data1[qid].get("Correct")
                c2 = data2[qid].get("Correct")
                total_1 += 1
                total_2 += 1
                if c1 is True:
                    correct_1 += 1
                if c2 is True:
                    correct_2 += 1
                if c1 != c2:
                    wrong_in_2.append({
                        "file": fname,
                        "question_id": qid,
                        "question": data1[qid].get("Question"),
                        "aspect": data1[qid].get("Aspect"),
                        "complexity": data1[qid].get("Complexity"),
                        "prompt1_answer": data1[qid].get("ModelAnswer"),
                        "prompt2_answer": data2[qid].get("ModelAnswer"),
                        "Correct1": c1,
                        "Correct2": c2,
                    })
                if data2[qid].get("TransformedQuestion") == "":
                    transform_question_fail.append({
                        "file": fname,
                        "question_id": qid,
                        "question": data1[qid].get("Question"),
                        "aspect": data1[qid].get("Aspect"),
                        "complexity": data1[qid].get("Complexity"),
                    })

# 输出结果
for item in wrong_in_2:
    print(f"{item['file']} | {item['question_id']} | {item['aspect']} | {item['complexity']}")
    print(f"1: {item['prompt1_answer']} {item['Correct1']} | 2: {item['prompt2_answer']} {item['Correct2']}")
    print('-' * 60)
# for item in transform_question_fail:
#     print(f"TransformQuestion is empty: {item['file']} | {item['question_id']} | {item['aspect']} | {item['complexity']} | {item['question']}")
print(f"Total questions in {dir1}: {total_1}, Correct: {correct_1}, Accuracy: {correct_1 / total_1 * 100:.2f}%")
print(f"Total questions in {dir2}: {total_2}, Correct: {correct_2}, Accuracy: {correct_2 / total_2 * 100:.2f}%")