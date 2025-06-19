"""
使用方法：
python gather.py --aspect 方面 --complexity 复杂度 --correct 正确与否 --output 输出目录名 --base_dir 待过滤结果目录名
例如：
python gather.py --aspect times --complexity mhp --correct true --output ./outputs --base_dir ./results
以收集特定方面、复杂度和正确与否的结果，并将其保存到指定的输出目录中。文件中未有该方面、复杂度和正确与否的结果将被忽略。
"""

import argparse

parser = argparse.ArgumentParser(description="Parse command line arguments.")
parser.add_argument('--aspect', type=str, required=False, help="Specify the aspect as a string.")
parser.add_argument('--complexity', type=str, required=False, help="Specify the compelity as a string.")
parser.add_argument('--correct', type=str, required=False, help="Specify the correct value as a string.")
parser.add_argument('--output', type=str, required=True, help="Specify the output directory name.")
parser.add_argument('--base_dir', type=str, required=True, help="Specify the base directory name.")

args = parser.parse_args()

aspect = args.aspect or 'all'
compelity = args.complexity or 'all'
correct = args.correct or 'all'
correct = correct.lower()
if correct == 'true':
    correct = True
elif correct == 'false':
    correct = False

print(aspect)
print(compelity)
print(correct)

base_dir = args.base_dir

import os

from src.utils import *

for filename in os.listdir(base_dir):
    if not filename.endswith('.json'):
        continue
    file_path = os.path.join(base_dir, filename)
    content = load_json(file_path)
    new_content = {}
    for key, value in content.items():
        if (aspect == 'all' or value['Aspect'] == aspect) and (compelity == 'all' or value['Complexity'] == compelity) and (correct == 'all' or value['Correct'] == correct):
            new_content[key] = value
    output_dir = f"{args.output}/{aspect}_{compelity}_{correct}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    if len(new_content) == 0:
        continue
    # print(output_path)
    save_json(new_content, output_path)

