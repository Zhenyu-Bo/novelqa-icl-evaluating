from src.utils import load_json, save_json
import os

icl_dir = "./results_both/icl"
zs_dir = "./results_both/zs"

icl_files = sorted(os.listdir(icl_dir))
zs_files = sorted(os.listdir(zs_dir))

both_files = set(icl_files).intersection(set(zs_files))

icl_correct_but_zs_wrong_num = 0
zs_correct_but_icl_wrong_num = 0
both_correct_num = 0
both_wrong_num = 0

if not os.path.exists("./results_both/icl_correct_but_zs_wrong"):
    os.makedirs("./results_both/icl_correct_but_zs_wrong")
if not os.path.exists("./results_both/zs_correct_but_icl_wrong"):
    os.makedirs("./results_both/zs_correct_but_icl_wrong")
if not os.path.exists("./results_both/both_correct"):
    os.makedirs("./results_both/both_correct")
if not os.path.exists("./results_both/both_wrong"):
    os.makedirs("./results_both/both_wrong")

for file in both_files:
    if not file.endswith(".json"):
        continue
    icl_file_path = os.path.join(icl_dir, file)
    zs_file_path = os.path.join(zs_dir, file)
    icl_data = load_json(icl_file_path)
    zs_data = load_json(zs_file_path)
    icl_correct_but_zs_wrong_data = {}
    zs_correct_but_icl_wrong_data = {}
    both_correct_data = {}
    both_wrong_data = {}
    for key in icl_data:
        if icl_data[key]["Correct"] and not zs_data[key]["Correct"]:
            icl_correct_but_zs_wrong_data[key] = icl_data[key]
            icl_correct_but_zs_wrong_data[key]['Analysis'] = zs_data[key]['Analysis']
            icl_correct_but_zs_wrong_data[key]['icl_answer'] = icl_data[key]['ModelAnswer']
            icl_correct_but_zs_wrong_data[key]['zs_answer'] = zs_data[key]['ModelAnswer']
            icl_correct_but_zs_wrong_data[key].pop('Correct')
            icl_correct_but_zs_wrong_data[key].pop('ModelAnswer')
        elif zs_data[key]["Correct"] and not icl_data[key]["Correct"]:
            zs_correct_but_icl_wrong_data[key] = zs_data[key]
            zs_correct_but_icl_wrong_data[key]['Analysis'] = zs_data[key]['Analysis']
            zs_correct_but_icl_wrong_data[key]['icl_answer'] = icl_data[key]['ModelAnswer']
            zs_correct_but_icl_wrong_data[key]['zs_answer'] = zs_data[key]['ModelAnswer']
            zs_correct_but_icl_wrong_data[key].pop('Correct')
            zs_correct_but_icl_wrong_data[key].pop('ModelAnswer')
        elif icl_data[key]["Correct"] and zs_data[key]["Correct"]:
            both_correct_data[key] = icl_data[key]
            both_correct_data[key]['Analysis'] = zs_data[key]['Analysis']
            both_correct_data[key]['icl_answer'] = icl_data[key]['ModelAnswer']
            both_correct_data[key]['zs_answer'] = zs_data[key]['ModelAnswer']
            both_correct_data[key].pop('Correct')
            both_correct_data[key].pop('ModelAnswer')
        elif not icl_data[key]["Correct"] and not zs_data[key]["Correct"]:
            both_wrong_data[key] = icl_data[key]
            both_wrong_data[key]['Analysis'] = zs_data[key]['Analysis']
            both_wrong_data[key]['icl_answer'] = icl_data[key]['ModelAnswer']
            both_wrong_data[key]['zs_answer'] = zs_data[key]['ModelAnswer']
            both_wrong_data[key].pop('Correct')
            both_wrong_data[key].pop('ModelAnswer')
    icl_correct_but_zs_wrong_num += len(icl_correct_but_zs_wrong_data)
    zs_correct_but_icl_wrong_num += len(zs_correct_but_icl_wrong_data)
    both_correct_num += len(both_correct_data)
    both_wrong_num += len(both_wrong_data)
    if len(icl_correct_but_zs_wrong_data) > 0:
        save_json(icl_correct_but_zs_wrong_data, f"./results_both/icl_correct_but_zs_wrong/{file}")
    if len(zs_correct_but_icl_wrong_data) > 0:
        save_json(zs_correct_but_icl_wrong_data, f"./results_both/zs_correct_but_icl_wrong/{file}")
    if len(both_correct_data) > 0:
        save_json(both_correct_data, f"./results_both/both_correct/{file}")
    if len(both_wrong_data) > 0:
        save_json(both_wrong_data, f"./results_both/both_wrong/{file}")

print(f"icl_correct_but_zs_wrong_num: {icl_correct_but_zs_wrong_num}")
print(f"zs_correct_but_icl_wrong_num: {zs_correct_but_icl_wrong_num}")
print(f"both_correct_num: {both_correct_num}")
print(f"both_wrong_num: {both_wrong_num}")
