import json
import pandas as pd

# 定義判斷區間的函數
def get_bin(score):
    if score < 0.0:
        score = 0.0
        
    if score >= 1.0:
        return "x = 1.0"
    elif 0.9 <= score < 1.0:
        return "1.0 > x >= 0.9"
    elif 0.8 <= score < 0.9:
        return "0.9 > x >= 0.8"
    elif 0.7 <= score < 0.8:
        return "0.8 > x >= 0.7"
    elif 0.6 <= score < 0.7:
        return "0.7 > x >= 0.6"
    elif 0.5 <= score < 0.6:
        return "0.6 > x >= 0.5"
    elif 0.4 <= score < 0.5:
        return "0.5 > x >= 0.4"
    elif 0.3 <= score < 0.4:
        return "0.4 > x >= 0.3"
    elif 0.2 <= score < 0.3:
        return "0.3 > x >= 0.2"
    elif 0.1 <= score < 0.2:
        return "0.2 > x >= 0.1"
    else:
        return "0.1 > x >= 0.0"

# 建立用來排序和顯示的區間清單
bins = [
    "x = 1.0", 
    "1.0 > x >= 0.9", "0.9 > x >= 0.8", "0.8 > x >= 0.7",
    "0.7 > x >= 0.6", "0.6 > x >= 0.5", "0.5 > x >= 0.4", 
    "0.4 > x >= 0.3", "0.3 > x >= 0.2", "0.2 > x >= 0.1", 
    "0.1 > x >= 0.0"
]

# 初始化計數器
score_counts = {b: 0 for b in bins}
cosine_counts = {b: 0 for b in bins}

# 1. 讀取並統計「最高判定分數」
try:
    with open('max_score_results.json', 'r', encoding='utf-8') as f:
        score_data = json.load(f)
        for word, info in score_data.items():
            val = info.get('最高判定分數', 0.0)
            score_counts[get_bin(val)] += 1
except FileNotFoundError:
    print("找不到 max_score_results.json，請確認檔案是否存在。")

# 2. 讀取並統計「最高Cosine分數」
try:
    with open('max_cosine_results.json', 'r', encoding='utf-8') as f:
        cosine_data = json.load(f)
        for word, info in cosine_data.items():
            val = info.get('最高Cosine分數', 0.0)
            cosine_counts[get_bin(val)] += 1
except FileNotFoundError:
    print("找不到 max_cosine_results.json，請確認檔案是否存在。")

# 3. 轉換成 Pandas DataFrame 並計算累計數量
# 判定分數 DataFrame
df_score = pd.DataFrame(list(score_counts.items()), columns=['分數區間', '數量'])
df_score['累計數量'] = df_score['數量'].cumsum()

# Cosine分數 DataFrame
df_cosine = pd.DataFrame(list(cosine_counts.items()), columns=['分數區間', '數量'])
df_cosine['累計數量'] = df_cosine['數量'].cumsum()

# 4. 輸出成 Excel 檔案
output_filename = '分數統計結果_含累計.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_score.to_excel(writer, sheet_name='判定分數統計', index=False)
    df_cosine.to_excel(writer, sheet_name='Cosine分數統計', index=False)

print(f"處理完成！結果已儲存至 {output_filename}")