import os
import json
import re
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data_set")
output_dir = os.path.join(current_dir, "output")

if not os.path.exists(data_dir):
    print(f"找不到資料夾: {data_dir}")
    sys.exit(1)

# 只處理 result_*.json 檔，按檔名內的兩位數字排序
files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f.startswith('result_')]

if not files:
    print("在 data_set 中找不到任何 result_*.json 檔。")
    sys.exit(1)

def extract_order(fname):
    m = re.search(r"(\d{2})", fname)
    return int(m.group(1)) if m else 9999

files_sorted = sorted(files, key=extract_order)

results_array = []
results_prearray = []
for fname in files_sorted:
    path = os.path.join(data_dir, fname)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            one_file_data = {"1.0": 0,"0.9": 0, "0.8": 0,"0.7": 0, "0.6": 0, "0.5": 0,"0.4": 0, "0.3": 0, "0.2": 0, "0.1": 0,"0.0": 0}
            data = json.load(f)
            for key, entry in data.items():
                if "分析結果" in entry:
                    score = str(round(entry["分析結果"]['判定分數'], 1))
                    if score not in one_file_data:
                        one_file_data[score] = 0
                    one_file_data[score] += 1
    except Exception as e:
        print(f"讀取 {fname} 時發生錯誤: {e}")
        continue

    sorted_scores = sorted(one_file_data.items(), key=lambda x: float(x[0]), reverse=True)

    cumulative_data = {}
    running_total = 0
    for score, count in sorted_scores:
        running_total += count
        cumulative_data[score] = running_total

    one_file_data = dict(sorted_scores)

    results_array.append({
        "file": fname,
        "content": one_file_data
    })

    results_prearray.append({
        "file": fname,
        "content": cumulative_data
    })

# 輸出 JSON 檔
out_path_json = os.path.join(output_dir, 'all_results_ordered_cos.json')
with open(out_path_json, 'w', encoding='utf-8') as f:
    json.dump(results_array, f, ensure_ascii=False, indent=2)
out_path_json_pre = os.path.join(output_dir, 'all_results_preordered_cos.json')
with open(out_path_json_pre, 'w', encoding='utf-8') as f:
    json.dump(results_prearray, f, ensure_ascii=False, indent=2)

# 輸出 Excel 檔 - 普通結果
df_list_ordered = []
for item in results_array:
    file_name = item['file']
    # 批取檔案名称中的族名："result_01阿美語_merged_chinese.json" -> "01.阿美語"
    match = re.search(r'result_(\d{2})([^_]+)_merged', file_name)
    tribe_name = f"{match.group(1)}.{match.group(2)}" if match else file_name
    
    scores_dict = item['content']
    for score, count in scores_dict.items():
        df_list_ordered.append({
            '檔案': tribe_name,
            '分數': float(score),
            '筆數': count
        })

# Pivot：分數為 row，檔案為 column
df_temp = pd.DataFrame(df_list_ordered)
df_ordered = df_temp.pivot(index='分數', columns='檔案', values='筆數')
# 按分數降序排序（從 1.0 開始）
df_ordered = df_ordered.sort_index(ascending=False)
excel_path_ordered = os.path.join(output_dir, 'all_results_ordered_cos.xlsx')
df_ordered.to_excel(excel_path_ordered, sheet_name='結果統計')

# 輸出 Excel 檔 - 累積結果
df_list_preordered = []
for item in results_prearray:
    file_name = item['file']
    # 批取檔案名称中的族名："result_01阿美語_merged_chinese.json" -> "01.阿美語"
    match = re.search(r'result_(\d{2})([^_]+)_merged', file_name)
    tribe_name = f"{match.group(1)}.{match.group(2)}" if match else file_name
    
    scores_dict = item['content']
    for score, count in scores_dict.items():
        df_list_preordered.append({
            '檔案': tribe_name,
            '分數': float(score),
            '累積筆數': count
        })

# Pivot：分數為 row，檔案為 column
df_temp_pre = pd.DataFrame(df_list_preordered)
df_preordered = df_temp_pre.pivot(index='分數', columns='檔案', values='累積筆數')
# 按分數降序排序（從 1.0 開始）
df_preordered = df_preordered.sort_index(ascending=False)
excel_path_preordered = os.path.join(output_dir, 'all_results_preordered_cos.xlsx')
df_preordered.to_excel(excel_path_preordered, sheet_name='累積統計')

print(f"已建立結果陣列，共包含 {len(results_array)} 個檔案")
print(f"JSON 檔案儲存在: {out_path_json}")
print(f"Excel 檔案儲存在: {excel_path_ordered}")
print(f"累積結果 JSON 儲存在: {out_path_json_pre}")
print(f"累積結果 Excel 儲存在: {excel_path_preordered}")
