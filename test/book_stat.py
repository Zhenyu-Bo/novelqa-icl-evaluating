import os
import json
from collections import defaultdict

book_ids = [f"B{i:02}" for i in range(0, 63)]
titles_dir = './cache/structures/titles'
questions_dir = '../NovelQA/data/PublicDomain'
output_chapter_file = './outputs/book_stats/chapter.log'
output_question_file = './outputs/book_stats/question_stats.json'


def analyze_chapter_depth(titles_dir, book_ids, output_file=None):
    """
    分析书籍章节的最大深度和深度为最大深度的章节数。

    Args:
        file_path (str): 书籍章节文件路径。
        output_file (str): 输出文件路径。如果为 None，则输出到控制台。

    Returns:
        tuple: 最大深度和深度为最大深度的章节数。
    """
    results = ""
    for book_id in book_ids:
        file_name = f"{book_id}.txt"
        file_path = os.path.join(titles_dir, file_name)
        if not os.path.exists(file_path):
            # print(f"文件 {file_path} 不存在！")
            continue
        
        max_depth = 0
        max_depth_count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):  # 判断是否是章节标题
                    depth = line.count("#")  # 统计 '#' 的数量，表示深度
                    if depth > max_depth:
                        max_depth = depth
                        max_depth_count = 1  # 重置计数器
                    elif depth == max_depth:
                        max_depth_count += 1

        # 输出结果
        output = f"{book_id}: 最大深度: {max_depth} - 最大深度章节数: {max_depth_count}\n"
        if max_depth_count >= 15 and max_depth_count <= 55:
            print(output.strip())
        results += output
    if output_file:
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(results)
    else:
        print(results)


def analyze_questions(base_dir, book_ids, output_file=None):
    """
    统计每个文件中的问题总数，以及按 Aspect 和 Complexity 分类的问题数。

    Args:
        base_dir (str): 存储 JSON 文件的目录路径。
        output_file (str): 输出文件路径。如果为 None，则输出到控制台。

    Returns:
        None
    """
    if not os.path.exists(base_dir):
        print(f"目录 {base_dir} 不存在！")
        return

    results = {}

    # 遍历目录中的每个文件
    for book_id in book_ids:
        file_name = f"{book_id}.json"
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            # print(f"文件 {file_path} 不存在！")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 初始化统计变量
        total_questions = 0
        aspect_count = defaultdict(int)
        complexity_count = defaultdict(int)

        # 遍历文件中的每个问题
        for question_id, question_data in data.items():
            total_questions += 1
            aspect = question_data.get("Aspect", "Unknown")
            complexity = question_data.get("Complexity", "Unknown")
            aspect_count[aspect] += 1
            complexity_count[complexity] += 1

        # 构建当前文件的统计结果
        results[file_name] = {
            "total_questions": total_questions,
            "aspect_count": dict(aspect_count),
            "complexity_count": dict(complexity_count),
        }

    # 输出到文件或控制台
    if output_file:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=4)
    else:
        print("统计结果:")
        print(json.dumps(results, ensure_ascii=False, indent=4))


def analyze_selected_books(titles_dir, questions_dir, book_ids, output_file=None):
    total_questions = 0
    aspect_count = defaultdict(int)
    complexity_count = defaultdict(int)
    question_chapter_product_sum = 0

    for book_id in book_ids:
        # 获取章节文件路径
        title_file_path = os.path.join(titles_dir, f"{book_id}.txt")
        # 获取问题文件路径
        question_file_path = os.path.join(questions_dir, f"{book_id}.json")

        # 统计章节数
        chapter_count = 0
        if os.path.exists(title_file_path):
            with open(title_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):  # 判断是否是章节标题
                        chapter_count += 1

        # 统计问题数
        if os.path.exists(question_file_path):
            with open(question_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for question_id, question_data in data.items():
                    total_questions += 1
                    aspect = question_data.get("Aspect", "Unknown")
                    complexity = question_data.get("Complexity", "Unknown")
                    aspect_count[aspect] += 1
                    complexity_count[complexity] += 1

            # 计算问题数与章节数的乘积
            question_chapter_product_sum += total_questions * chapter_count

    # 构建统计结果
    stats = {
        "total_questions": total_questions,
        "aspect_count": dict(aspect_count),
        "complexity_count": dict(complexity_count),
        "question_chapter_product_sum": question_chapter_product_sum,
    }

    # 输出到文件或控制台
    if output_file:
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(stats, out_f, ensure_ascii=False, indent=4)
    else:
        print("统计结果:")
        print(json.dumps(stats, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    analyze_chapter_depth(titles_dir, book_ids, output_chapter_file)
    analyze_questions(questions_dir, book_ids, output_question_file)
    selected_book_ids = ["B00", "B05", "B09", "B13", "B14", "B16", "B17", "B20", "B22","B24",
                         "B25", "B29", "B33", "B34", "B37", "B43", "B44", "B53", "B55", "B60"]
    print("Total:")
    analyze_selected_books(titles_dir, questions_dir, book_ids, output_file=None)
    print("Selected:")
    analyze_selected_books(titles_dir, questions_dir, selected_book_ids, output_file=None)