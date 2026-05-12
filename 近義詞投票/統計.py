import os
import json
import glob
import re

# 請將這個路徑換成你放這16個檔案的資料夾路徑
folder_path = './model_training/近義詞投票/data_set' 
file_pattern = 'result_*_merged_chinese.json'

# 用來儲存兩個獨立統計結果的字典
max_score_results = {}
max_cosine_results = {}

# 找出資料夾下所有符合檔名規則的 JSON
for filepath in glob.glob(os.path.join(folder_path, file_pattern)):
    filename = os.path.basename(filepath)
    
    # 透過正則表達式萃取族群名稱，例如從 'result_01阿美語_merged_chinese.json' 抓出 '阿美語'
    match = re.search(r'result_\d*(.*?)_merged', filename)
    tribe_name = match.group(1) if match else "未知族群"

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for key, entry in data.items():
            analysis = entry.get('分析結果', {})
            new_word = analysis.get('新詞')
            
            # 如果沒有新詞欄位則跳過
            if not new_word:
                continue
                
            # 處理判定分數 (轉為浮點數，避開字串或None的錯誤)
            try:
                score = float(analysis.get('判定分數', 0.0))
            except (ValueError, TypeError):
                score = 0.0
                
            # 處理 Cosine 分數
            try:
                cosine = float(analysis.get('餘弦相似度(Cosine)', 0.0))
            except (ValueError, TypeError):
                cosine = 0.0

            # 統計 1: 更新最高判定分數與對應族群
            if new_word not in max_score_results or score > max_score_results[new_word]['最高判定分數']:
                max_score_results[new_word] = {
                    '最高判定分數': score,
                    '來源族群': tribe_name
                }
                
            # 統計 2: 更新最高 Cosine 分數與對應族群
            if new_word not in max_cosine_results or cosine > max_cosine_results[new_word]['最高Cosine分數']:
                max_cosine_results[new_word] = {
                    '最高Cosine分數': cosine,
                    '來源族群': tribe_name
                }

# 將兩個統計結果分別輸出成獨立的 JSON 檔案
with open('max_score_results.json', 'w', encoding='utf-8') as f:
    json.dump(max_score_results, f, ensure_ascii=False, indent=4)

with open('max_cosine_results.json', 'w', encoding='utf-8') as f:
    json.dump(max_cosine_results, f, ensure_ascii=False, indent=4)

print("處理完成！結果已分別儲存至 max_score_results.json 與 max_cosine_results.json")