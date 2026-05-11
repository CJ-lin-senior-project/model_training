import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import os

# ==========================================
# 📂 1. 設定與裝置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用運算裝置: {device}")

# ==========================================
# 🏗️ 2. 定義模型架構 (鎖定為 1536 維)
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

# ==========================================
# 🧠 3. 載入 BERT 與 5 個 MLP 模型
# ==========================================
print("📦 正在載入 bert-base-chinese (共用特徵擷取器)...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval()

# 設定 5 個權重檔案的名稱
model_paths = [
    os.path.join(current_dir, "level0.pth"), 
    os.path.join(current_dir, "level1.pth"), 
    os.path.join(current_dir, "level2.pth"), 
    os.path.join(current_dir, "level3.pth"), 
    os.path.join(current_dir, "level4.pth")
]

mlp_models = []
print("📦 正在載入 5 個 MLP 分類模型...")
for i, path in enumerate(model_paths):
    # 🌟 直接統一設定為 1536 維
    model = BinaryMLP(1536).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        mlp_models.append(model)
        print(f"  ✅ 成功載入 Level {i} 模型 ({path})")
    else:
        print(f"  ❌ 找不到模型檔案 {path}，請確認路徑！")
        exit()

# ==========================================
# ⚙️ 4. 特徵萃取與判定邏輯 (純 1536 維)
# ==========================================
def extract_bert_features(word1, word2):
    """只將兩個詞轉換成 1536 維的特徵 (768 + 768)"""
    with torch.no_grad():
        inputs = tokenizer([word1, word2], padding=True, truncation=True, return_tensors="pt", max_length=16).to(device)
        outputs = bert_model(**inputs)
        # 取 [CLS] token 作為詞彙表徵
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() 
        
        vec1, vec2 = embeddings[0], embeddings[1]
        
        # 🌟 只拼接 vec1 和 vec2，變成 1536 維
        feature = np.concatenate([vec1, vec2])
        return feature

def predict_synonym_level(word1, word2):
    """推論兩個詞的近義詞等級"""
    # 1. 取得 1536 維特徵
    feature = extract_bert_features(word1, word2)
    
    # 轉成 Tensor 並加上 Batch 維度 (變成 [1, 1536])
    input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. 讓 5 個模型同時進行推論
    predictions = []
    with torch.no_grad():
        for model in mlp_models:
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())  # 會是 0 或 1
            
    # 3. 判定層級邏輯 (找出第一個判斷為 1 的 Level)
    final_level = -1 # 預設 -1 代表「都不是近義詞」
    
    for level, is_synonym in enumerate(predictions):
        if is_synonym == 1:
            final_level = level
            break # 只要找到第一個說它是近義詞的，就鎖定該 Level 並跳出迴圈
            
    return final_level, predictions

# ==========================================
# 🚀 5. 實戰測試
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 近義詞多層級判定系統已啟動 (輸入 'quit' 離開)")
    print("="*50)
    
    while True:
        w1 = input("\n請輸入第一個詞: ")
        if w1.lower() == 'quit': break
        w2 = input("請輸入第二個詞: ")
        if w2.lower() == 'quit': break
        
        level, preds = predict_synonym_level(w1, w2)
        
        print("\n📊 模型判定結果明細:")
        for i, p in enumerate(preds):
            status = "✅ 近義詞 (1)" if p == 1 else "❌ 非近義 (0)"
            print(f"   - Level {i} 模型: {status}")
            
        print("-" * 30)
        if level != -1:
            print(f"🎯 綜合判定結果: 【 LEVEL {level} 近義詞 】")
        else:
            print(f"🎯 綜合判定結果: 【 均非近義詞 】")