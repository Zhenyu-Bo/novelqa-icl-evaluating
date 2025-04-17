import re

def extract_entries_no_evidence(response_str: str) -> list[dict]:
    """解析模型返回的字符串，提取问题id、模型的分析、模型给出的答案"""
    entries = []
    current_id = None
    current_answer = None
    analysis_lines = []

    response_lines = response_str.split('\n')
    
    for line in response_lines:
        stripped_line = line.strip()
        match = re.match(r'^Q(\d+):', stripped_line)
        if match:
            if current_id is not None:
                entries.append({
                    'id': f"Q{current_id}",
                    'analysis': '\n'.join(analysis_lines).strip(),
                    'answer': current_answer,
                })
            current_id = match.group(1)
            analysis_lines = []
        elif re.match(r'^Answer: [A-D]', stripped_line):
            match = re.match(r'^Answer: [A-D]', stripped_line)
            current_answer = match.group(0).split(' ')[1]
        else:
            analysis_lines.append(line.rstrip('\n'))
    
    # 处理最后一个条目
    if current_id is not None:
        entries.append({
            'id': f"Q{current_id}",
            'analysis': '\n'.join(analysis_lines).strip(),
            'answer': current_answer,
        })
    
    return entries


def extract_entries(response_str: str) -> list[dict]:
    """解析模型返回的字符串，提取问题id、模型的分析、模型给出的答案、模型找到的证据"""
    entries = []
    current_id = None
    current_answer = None
    analysis_lines = []
    evidence_lines = []

    response_lines = response_str.split('\n')
    is_evidence_line = False
    
    for line in response_lines:
        stripped_line = line.strip()
        match = re.match(r'^Q(\d+):', stripped_line)
        if match:
            if current_id is not None:
                entries.append({
                    'id': f"Q{current_id}",
                    'analysis': '\n'.join(analysis_lines).strip(),
                    'answer': current_answer,
                    'evidence': '\n'.join(evidence_lines).strip()
                })
            current_id = match.group(1)
            analysis_lines = []
            evidence_lines = []
            is_evidence_line = False
        elif re.match(r'^Answer: [A-D]', stripped_line):
            match = re.match(r'^Answer: [A-D]', stripped_line)
            current_answer = match.group(0).split(' ')[1]
            is_evidence_line = True
        elif is_evidence_line:
            evidence_lines.append(line.rstrip('\n'))
        else:
            analysis_lines.append(line.rstrip('\n'))
    
    # 处理最后一个条目
    if current_id is not None:
        entries.append({
            'id': f"Q{current_id}",
            'analysis': '\n'.join(analysis_lines).strip(),
            'answer': current_answer,
            'evidence': '\n'.join(evidence_lines).strip()
        })
    
    return entries

def merge(entries: list[dict], question_dict: dict):
    """与既有的结果合并"""
    for entry in entries:
        question_id = entry['id']
        if question_id in question_dict:
            question_dict[question_id]["ModelAnswer"] = entry['answer']
            question_dict[question_id]['Analysis'] = entry['analysis']
            if 'evidence' in entry:
                question_dict[question_id]['Evidence'] = entry['evidence']
            question_dict[question_id]["Correct"] = entry['answer'] == question_dict[question_id]['Answer']

    for key in question_dict:
        if 'ModelAnswer' not in question_dict[key]:
            question_dict[key]["ModelAnswer"] = ""
            question_dict[key]["Correct"] = False

    return question_dict
