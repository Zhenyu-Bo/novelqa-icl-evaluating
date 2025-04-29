base_dir = "./results"

import json
import os

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
        file_path = os.path.join(base_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f).values()
            for d in data:
                if d["Aspect"] not in aspect_correct:
                    aspect_correct[d["Aspect"]] = 0
                    aspect_total[d["Aspect"]] = 0
                if d["Complexity"] not in complexity_correct:
                    complexity_correct[d["Complexity"]] = 0
                    complexity_total[d["Complexity"]] = 0
                if d["Correct"]:
                    local_correct += 1
                    aspect_correct[d["Aspect"]] += 1
                    complexity_correct[d["Complexity"]] += 1
                if "which chapter" in d["Question"].lower():
                    aspect_correct["Chapter"] += 1
                    aspect_total["Chapter"] += 1
                aspect_total[d["Aspect"]] += 1
                complexity_total[d["Complexity"]] += 1
                local_total += 1
        print(f"File: {file}, Accuracy: {local_correct / local_total * 100:.2f}%")
        sum_correct += local_correct
        sum_total += local_total

print("Aspect Accuracy:")
for aspect in aspect_correct:
    print(f"{aspect}: {aspect_correct[aspect] / aspect_total[aspect] * 100:.2f}%")
print("Complexity Accuracy:")
for complexity in complexity_correct:
    print(
        f"{complexity}: {complexity_correct[complexity] / complexity_total[complexity] * 100:.2f}%"
    )
print(f"Accuracy: {sum_correct / sum_total * 100:.2f}%")
