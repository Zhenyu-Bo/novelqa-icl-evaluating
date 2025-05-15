# base_dir = "./outputs/in-context/output_71_85"
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_dir",
    type=str,
    default="./outputs",
    help="The base directory of the output files.",
)
args = parser.parse_args()
base_dir = args.base_dir
if not os.path.exists(base_dir):
    raise ValueError(f"Directory {base_dir} does not exist.")

test_aspects = ['character', 'meaning', 'times', 'all']
test_complexity = ['dtl', 'all']
BOOK_IDS = ["B00", "B05", "B09", "B13", "B14", "B16", "B17", "B20", "B22","B24",
            "B25", "B29", "B33", "B34", "B37", "B43", "B44", "B53", "B55", "B60"]

sum_correct = 0
sum_total = 0

aspect_correct = {}
aspect_total = {}

complexity_correct = {}
complexity_total = {}

for file in sorted(os.listdir(base_dir)):
    local_correct = 0
    local_total = 0
    if file.endswith(".json"):
        if file.replace(".json", "") not in BOOK_IDS:
            continue
        file_path = os.path.join(base_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f).values()
            for d in data:
                if 'all' not in test_aspects and 'all' not in test_complexity:
                    if d['Aspect'] not in test_aspects and d['Complexity'] not in test_complexity:
                        continue
                if 'Correct' not in d:
                    continue
                # if 'TransformedQuestion' in d:
                #     if d['TransformedQuestion'].lower() == "the transformed question" or d['TransformedQuestion'].lower() == "transformed question":
                #         # print(f"File: {file}, Question: {d['Question']}")
                #         # print("TransformedQuestion: ", d['TransformedQuestion'])
                #         continue
                if d["Aspect"] not in aspect_correct:
                    aspect_correct[d["Aspect"]] = 0
                    aspect_total[d["Aspect"]] = 0
                if d["Complexity"] not in complexity_correct:
                    complexity_correct[d["Complexity"]] = 0
                    complexity_total[d["Complexity"]] = 0
                if "Chapter" not in aspect_correct:
                    aspect_correct["Chapter"] = 0
                    aspect_total["Chapter"] = 0
                if d["Correct"]:
                    local_correct += 1
                    aspect_correct[d["Aspect"]] += 1
                    complexity_correct[d["Complexity"]] += 1
                    if "which chapter" in d["Question"].lower():
                        aspect_correct["Chapter"] += 1
                aspect_total[d["Aspect"]] += 1
                complexity_total[d["Complexity"]] += 1
                local_total += 1
                if "which chapter" in d["Question"].lower():
                    aspect_total["Chapter"] += 1
        if local_total != 0:
            print(f"File: {file}, Accuracy: {local_correct / local_total * 100:.2f}%({local_correct}/{local_total})")
        sum_correct += local_correct
        sum_total += local_total

print("Aspect Accuracy:")
for aspect in aspect_correct:
    if aspect_total[aspect] == 0:
        continue
    print(f"{aspect}: {aspect_correct[aspect] / aspect_total[aspect] * 100:.2f}%({aspect_correct[aspect]}/{aspect_total[aspect]})")
print("Complexity Accuracy:")
for complexity in complexity_correct:
    if complexity_total[complexity] == 0:
        continue
    print(
        f"{complexity}: {complexity_correct[complexity] / complexity_total[complexity] * 100:.2f}%({complexity_correct[complexity]}/{complexity_total[complexity]})"
    )
if sum_total != 0:
    print(f"Overall Accuracy: {sum_correct / sum_total * 100:.2f}%({sum_correct}/{sum_total})")
else:
    print("No data found.")
    sum_correct = 0
    sum_total = 0
    print("Overall Accuracy: 0.00%(0/0)")
