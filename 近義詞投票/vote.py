import json
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import re
from transformers import BertTokenizer, BertModel

# ==========================================
# 📂 1. 設定與路徑 (👉 請在這裡填入你的相對路徑)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 請填入包含那 16 個 JSON 檔案的資料夾相對路徑。
# 例如："./data" 或 "../my_json_files"
# 如果檔案就跟這支 python 程式放在一起，請填 "./"
TARGET_DIR = os.path.join(current_dir, "data_set")  

# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用運算裝置: {device}")

# 定義分數對應表
LEVEL_SCORES = {
    0: 1.0,
    1: 0.8,
    2: 0.6,
    3: 0.4,
    4: 0.2,
    -1: 0.0  # 都不是
}

# ==========================================
# 🏗️ 2. 模型架構與載入 (1536 維版)
# ==========================================
class BinaryMLP(nn.Module):
    def __init__(self, input_dim=1536):
        super(BinaryMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.LeakyReLU(0.1), nn.BatchNorm1d(2048), nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.LeakyReLU(0.1), nn.BatchNorm1d(1024), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.LeakyReLU(0.1), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        return self.network(x)

print("📦 正在載入 bert-base-chinese (共用特徵擷取器)...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval()

# 確保你有 level0.pth 到 level4.pth
model_paths = [os.path.join(current_dir,"bert", name) for name in ["level0.pth", "level1.pth", "level2.pth", "level3.pth", "level4.pth"]]
mlp_models = []

print("📦 正在載入 5 個 MLP 分類模型...")
for i, path in enumerate(model_paths):
    model = BinaryMLP(1536).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        mlp_models.append(model)
        print(f"  ✅ 成功載入 Level {i} 模型")
    else:
        print(f"  ❌ 找不到模型檔案 {path}，請確認路徑！")
        exit()

# ==========================================
# ⚙️ 3. 特徵萃取與投票邏輯
# ==========================================
def extract_bert_features(word1, word2):
    """將兩個詞轉換成 1536 維特徵"""
    with torch.no_grad():
        inputs = tokenizer([word1, word2], padding=True, truncation=True, return_tensors="pt", max_length=16).to(device)
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        vec1, vec2 = embeddings[0], embeddings[1]
        return np.concatenate([vec1, vec2])
    
def cosine_similarity(v1, v2):
    """計算兩個向量的餘弦相似度，或兩個詞的 BERT 向量相似度"""
    # 如果輸入是字串，先轉成 BERT 向量
    if isinstance(v1, str):
        with torch.no_grad():
            inputs1 = tokenizer(v1, return_tensors='pt', truncation=True, max_length=16).to(device)
            outputs1 = bert_model(**inputs1)
            v1 = outputs1.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    if isinstance(v2, str):
        with torch.no_grad():
            inputs2 = tokenizer(v2, return_tensors='pt', truncation=True, max_length=16).to(device)
            outputs2 = bert_model(**inputs2)
            v2 = outputs2.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def vote_synonym_level(word1, word2):
    """從高 Level(4, 最寬鬆) 往低 Level(0, 最嚴格) 檢查，判定最終 Level"""
    if not word1.strip() or not word2.strip():
        return -1, 0.0

    feature = extract_bert_features(word1, word2)
    input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 取得 5 個模型的預測結果
    predictions = []
    with torch.no_grad():
        for model in mlp_models:
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())  # 0 或 1
            
    # predictions 的 index 代表 level (0~4)
    # 邏輯：從 4 檢查到 0。只要有一個說「不是(0)」，就退回前一個 level。
    final_level = -1
    check_order = [4, 3, 2, 1, 0]
    
    current_best = -1
    for lvl in check_order:
        is_synonym = predictions[lvl]
        if is_synonym == 1:
            current_best = lvl # 它過關了，記錄下來，繼續往下(嚴格)看
        else:
            # 遇到第一個說「不是」的，終止檢查
            break 
            
    final_level = current_best
    return final_level, LEVEL_SCORES[final_level]

# ==========================================
# 🚀 4. 批次處理 JSON 檔案
# ==========================================
def process_json_files():
    # 檢查目標目錄是否存在
    if not os.path.exists(TARGET_DIR):
        print(f"⚠️ 找不到指定的資料夾: {TARGET_DIR}，請確認相對路徑是否正確！")
        return

    # 組合搜尋路徑，並讀取所有符合條件的 json 檔案
    search_pattern = os.path.join(TARGET_DIR, "*.json")
    json_files = glob.glob(search_pattern)
    
    # 排除掉可能之前產生的輸出檔案，避免重複處理 (檔名包含 result_ 的跳過)
    json_files = [f for f in json_files if not os.path.basename(f).startswith("result_")]
    
    if not json_files:
        print(f"⚠️ 在 {TARGET_DIR} 中找不到任何 JSON 檔案！")
        return

    print(f"\n📂 在 {TARGET_DIR} 中找到 {len(json_files)} 個 JSON 檔案準備處理...")

    for file_path in json_files:
        # 只取出檔案名稱，方便印出資訊
        file_name = os.path.basename(file_path)
        print(f"\n⏳ 正在處理檔案: {file_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        result_data = {}
        
        for amis_key, entry in data.items():
            new_word = entry.get("新詞中文", "").strip()
            dict_word_raw = entry.get("辭典中文", "").strip()
            
            # 準備用來記錄最高分的配對
            best_dict_word = ""
            best_level = -1
            best_score = -1.0
            
            if new_word and dict_word_raw:
                # 支援全形 `；` 和半形 `;` 進行切割，並去除多餘空白
                dict_words = [w.strip() for w in re.split(r'[;；]', dict_word_raw) if w.strip()]
                
                for d_word in dict_words:
                    level, score = vote_synonym_level(new_word, d_word)
                    # score = cosine_similarity(new_word, d_word)
                    # level = 6
                    
                    # 比較並保留最高分
                    if score > best_score:
                        best_score = score
                        best_level = level
                        best_dict_word = d_word
            
            # 將結果打包存入新字典
            result_data[amis_key] = entry
            result_data[amis_key]["分析結果"] = {
                "新詞": new_word,
                "最佳配對辭典詞": best_dict_word,
                "判定Level": best_level if best_level != -1 else "都不是",
                "判定分數": float(best_score) if best_score != -1.0 else 0.0
            }
        
        # 儲存結果為新的 JSON 檔案 (存回原本的資料夾)
        output_filename = os.path.join(TARGET_DIR, f"result_{file_name}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
            
        print(f"  ✅ 處理完成，已儲存至: {output_filename}")

if __name__ == "__main__":
    process_json_files()
    print("\n🎉 所有檔案處理完畢！")