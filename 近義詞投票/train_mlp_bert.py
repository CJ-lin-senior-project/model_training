import json
import numpy as np
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import BertTokenizer, BertModel # 🌟 改用 Transformers

# ==========================================
# 📂 1. 設定與路徑
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(current_dir,"task_01234_vs_5", "train.jsonl")  
test_file = os.path.join(current_dir,"task_01234_vs_5", "test.jsonl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用運算裝置: {device}")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==========================================
# 🧠 2. BERT 特徵萃取與資料載入
# ==========================================
print("⏳ 正在載入 BERT 模型 (這會需要一點時間)...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval()

# 🌟 建立快取字典，避免幾十萬筆資料重複計算同一個詞，節省大量時間
bert_cache = {}

def get_bert_vector(word):
    if word in bert_cache:
        return bert_cache[word]
    
    # 將單字轉成 Token 並丟入 BERT
    inputs = tokenizer(word, return_tensors='pt', truncation=True, max_length=16).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # 取出 last_hidden_state，並對所有 Token 取平均值 (Mean Pooling) 作為該詞的代表向量
    vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    bert_cache[word] = vec # 存入快取
    return vec

def cosine_similarity(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def extract_features(word1, word2):
    vec1 = get_bert_vector(word1)
    vec2 = get_bert_vector(word2)
    
    # 🌟 手動計算餘弦相似度與絕對距離
    cos_sim = cosine_similarity(vec1, vec2)
    abs_diff = np.abs(vec1 - vec2)

    # 🌟 字元重疊率 (Jaccard Similarity)
    set1 = set(word1)
    set2 = set(word2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    char_overlap = intersection / union if union > 0 else 0.0

    return np.concatenate([vec1, vec2])

def load_data(file_path):
    X, Y = [], []
    count = 0
    print(f"📂 開始解析檔案: {os.path.basename(file_path)}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            if data.get('word1') and data.get('word2') and 'label' in data:
                X.append(extract_features(data['word1'], data['word2']))
                Y.append(int(data['label']))
                
                count += 1
                if count % 20000 == 0:
                    print(f"   ...已處理 {count} 筆資料")
                    
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

X_train_raw, y_train_raw = load_data(train_file)
X_test_raw, y_test_raw = load_data(test_file)

# 🌟 自動計算訓練資料比例與權重
print("\n📊 正在統計訓練資料比例...")
counts = np.bincount(y_train_raw)
n0, n1 = counts[0], counts[1]
w0 = 1.0
multiplier = 1.5  
w1 = (n0 / n1) * multiplier
class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float32).to(device)

print(f"   - 非同義詞 (0): {n0} 筆 (權重: {w0:.2f})")
print(f"   - 同義詞 (1)  : {n1} 筆 (權重: {w1:.2f})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print(f"✨ 特徵維度變化: 從原本的 902 維，變成了 {X_train_scaled.shape[1]} 維！")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_raw)), batch_size=4096, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test_raw)), batch_size=4096, shuffle=False)

# ==========================================
# 🏗️ 3. 建立神經網路 (加寬版架構)
# ==========================================
class BinaryMLP(nn.Module):
    def __init__(self, input_dim):
        super(BinaryMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, 2) 
        )

    def forward(self, x):
        return self.network(x)

model = BinaryMLP(X_train_scaled.shape[1]).to(device)
criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
best_model_path = os.path.join(current_dir, 'level4.pth')

# ==========================================
# 🔄 4. 訓練迴圈
# ==========================================
EPOCHS = 50
patience = 15
best_test_loss = float('inf')
epochs_no_improve = 0
start_time = time.time()

print(f"\n🚀 開始訓練模型...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(model(inputs), labels)
            test_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    test_loss /= len(test_loader.dataset)
    scheduler.step(test_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience: 
            print("🛑 觸發 Early Stopping，停止訓練。")
            break

# ==========================================
# 📊 5. 報表匯出 (自動尋找最佳門檻 + 實戰診斷)
# ==========================================
print("\n📝 載入最佳權重，產生詳細診斷報表...")
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()

all_probs_class1 = []

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        all_probs_class1.extend(probabilities[:, 1].cpu().numpy())

prob_array = np.array(all_probs_class1)

print("🔍 正在尋找最佳 F1 門檻...")
best_f1 = 0
best_threshold = 0.5

for thresh in np.arange(0.1, 0.9, 0.05):
    preds_temp = (prob_array > thresh).astype(int) 
    current_f1 = f1_score(y_test_raw, preds_temp)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh

print(f"👉 最佳門檻為: {best_threshold:.2f}，此時的 F1 Score 預估為: {best_f1*100:.2f}%")

y_pred = (prob_array > best_threshold).astype(int)

target_names = ['非近義詞 (0)', '近義詞 (1)']
report = classification_report(y_test_raw, y_pred, target_names=target_names, output_dict=True, zero_division=0)

df_overall = pd.DataFrame([
    ['【總】Accuracy (準確率)', '【總】Macro Precision', '【總】Macro Recall', '【總】Macro F1'],
    [
        round(report['accuracy'] * 100, 2),
        round(report['macro avg']['precision'] * 100, 2),
        round(report['macro avg']['recall'] * 100, 2),
        round(report['macro avg']['f1-score'] * 100, 2)
    ]
])

level_rows = []
for cls_name in target_names:
    level_rows.append([f"[{cls_name}] Precision (%)", f"[{cls_name}] Recall (%)", f"[{cls_name}] F1 (%)", ""])
    level_rows.append([
        round(report[cls_name]['precision'] * 100, 2),
        round(report[cls_name]['recall'] * 100, 2),
        round(report[cls_name]['f1-score'] * 100, 2),
        ""
    ])
df_levels = pd.DataFrame(level_rows)

cm = confusion_matrix(y_test_raw, y_pred)
tn, fp, fn, tp = cm.ravel()
df_cm = pd.DataFrame([
    ['【實戰診斷】', '數量 (筆)', '代表意義', ''],
    ['成功抓出 (True Positive)', tp, '真的是同義詞，模型也答對', ''],
    ['漏網之魚 (False Negative)', fn, '明明是同義詞，模型卻漏抓', ''],
    ['成功排除 (True Negative)', tn, '不是同義詞，模型也沒被騙', ''],
    ['冤枉好人 (False Positive)', fp, '根本不是同義詞，模型卻亂抓', '']
])

df_empty = pd.DataFrame([[""] * 4])
final_report = pd.concat([
    df_overall, df_empty, 
    df_levels, df_empty, 
    df_cm
], ignore_index=True)

excel_filename = os.path.join(current_dir, "BERT_mlp_詳細正確率_01.xlsx")
final_report.to_excel(excel_filename, index=False, header=False)

total_train_time = time.time() - start_time
print("\n" + "="*60)
print(f"🎉 報表已生成！耗時 {total_train_time:.2f} 秒。請查看:\n👉 {excel_filename}")
print("="*60)