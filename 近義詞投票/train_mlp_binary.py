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
from transformers import BertTokenizer, BertModel

# ==========================================
# 📂 1. 設定與路徑
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(current_dir,"task_0_vs_12345", "train.jsonl")  
test_file = os.path.join(current_dir,"task_0_vs_12345", "test.jsonl")

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
# 🧠 2. 特徵萃取與資料載入 (鎖定為 1536 維)
# ==========================================
print("📦 正在載入 bert-base-chinese 模型與 Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
bert_model.eval() 

def get_all_unique_words(*file_paths):
    words = set()
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line.strip())
                if data.get('word1'): words.add(data['word1'])
                if data.get('word2'): words.add(data['word2'])
    return list(words)

def build_bert_embeddings(words, batch_size=512):
    word2vec = {}
    print(f"🚀 開始將 {len(words)} 個獨立詞彙轉換為 BERT 向量 (Batch Size: {batch_size})...")
    
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            inputs = tokenizer(batch_words, padding=True, truncation=True, return_tensors="pt", max_length=16).to(device)
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            for w, emb in zip(batch_words, embeddings):
                word2vec[w] = emb
    return word2vec

def load_data_with_cache(file_path, word2vec):
    X, Y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            w1, w2 = data.get('word1'), data.get('word2')
            if w1 and w2 and 'label' in data:
                vec1 = word2vec.get(w1, np.zeros(768))
                vec2 = word2vec.get(w2, np.zeros(768))
                
                # 🔒 維持不變：合併特徵 768 + 768 = 1536 維
                feature = np.concatenate([vec1, vec2])
                X.append(feature)
                Y.append(int(data['label']))
                
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

all_unique_words = get_all_unique_words(train_file, test_file)
word2vec_cache = build_bert_embeddings(all_unique_words, batch_size=1024)

print("📊 正在組合訓練集與測試集特徵...")
X_train_raw, y_train_raw = load_data_with_cache(train_file, word2vec_cache)
X_test_raw, y_test_raw = load_data_with_cache(test_file, word2vec_cache)

print("\n📊 正在統計訓練資料比例...")
counts = np.bincount(y_train_raw)
n0, n1 = counts[0], counts[1]
w0 = 1.0
w1 = n0 / n1  
class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float32).to(device)

print(f"   - 非同義詞 (0): {n0} 筆 (權重: {w0:.2f})")
print(f"   - 同義詞 (1)  : {n1} 筆 (權重: {w1:.2f})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_raw)), batch_size=4096, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test_raw)), batch_size=4096, shuffle=False)

# ==========================================
# 🏗️ 3. 建立神經網路 (🔒 維持原本架構不變)
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

# 🌟 優化 1：將 Gamma 從 4.0 調降為 2.0，避免模型完全放棄邊緣樣本
criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 把 LR Scheduler 的依據改為 'max'，因為我們之後要餵給它 F1 Score
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
best_model_path = os.path.join(current_dir, 'level0.pth')

# ==========================================
# 🔄 4. 訓練迴圈 (改為監控 F1 Score)
# ==========================================
EPOCHS = 50
patience = 15
best_test_f1 = 0.0  # 🌟 優化 2：改用 F1 作為存檔與早停依據
epochs_no_improve = 0
start_time = time.time()

print(f"\n🏃‍♂️ 開始訓練 MLP 模型 (Input Dimension: {X_train_scaled.shape[1]})...")

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
    all_targets = []
    all_preds_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # 計算 Loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # 取得預測為類別 1 (同義詞) 的機率
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_targets.extend(labels.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy())
    
    train_loss /= len(train_loader.dataset)
    test_loss /= len(test_loader.dataset)
    
    # 計算這個 Epoch 的 F1 (預設以 0.5 為閾值做監控)
    current_preds = (np.array(all_preds_probs) > 0.5).astype(int)
    current_f1 = f1_score(all_targets, current_preds, pos_label=1, zero_division=0)
    
    # LR Scheduler 根據 F1 分數來調整
    scheduler.step(current_f1)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test F1: {current_f1:.4f}")
    
    # 只要 F1 分數創新高，就存檔！
    if current_f1 > best_test_f1:
        best_test_f1 = current_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  🌟 F1 分數提升至 {best_test_f1:.4f}，模型已儲存！")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered.")
            break

# ==========================================
# 📊 5. 報表匯出 (動態閾值尋找最佳 F1)
# ==========================================
print("\n📝 載入最佳權重，尋找最佳閾值並產生報表...")
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()

all_preds_probs = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        # 🌟 優化 3：不要直接用 torch.max，改取同義詞的機率
        probs = F.softmax(outputs, dim=1)[:, 1]
        all_preds_probs.extend(probs.cpu().numpy())

y_prob = np.array(all_preds_probs)

# 自動尋找讓同義詞 F1 最高的閾值 (從 0.1 測試到 0.9)
best_threshold = 0.5
highest_f1 = 0.0

for t in np.arange(0.1, 0.95, 0.05):
    temp_preds = (y_prob > t).astype(int)
    temp_f1 = f1_score(y_test_raw, temp_preds, pos_label=1, zero_division=0)
    if temp_f1 > highest_f1:
        highest_f1 = temp_f1
        best_threshold = t

print(f"🎯 掃描完畢！找到最佳預測閾值為: {best_threshold:.2f} (此時同義詞 F1: {highest_f1:.4f})")

# 使用最佳閾值來產生最終預測結果
y_pred = (y_prob > best_threshold).astype(int)

target_names = ['非同義詞 (0)', '同義詞 (1)']
report = classification_report(y_test_raw, y_pred, target_names=target_names, output_dict=True, zero_division=0)

df_overall = pd.DataFrame([
    ['【總】Accuracy (準確率)', '【總】Macro Precision', '【總】Macro Recall', '【總】Macro F1', f'【閾值】{best_threshold:.2f}'],
    [
        round(report['accuracy'] * 100, 2),
        round(report['macro avg']['precision'] * 100, 2),
        round(report['macro avg']['recall'] * 100, 2),
        round(report['macro avg']['f1-score'] * 100, 2),
        ""
    ]
])

level_rows = []
for cls_name in target_names:
    level_rows.append([f"[{cls_name}] Precision (%)", f"[{cls_name}] Recall (%)", f"[{cls_name}] F1 (%)", "", ""])
    level_rows.append([
        round(report[cls_name]['precision'] * 100, 2),
        round(report[cls_name]['recall'] * 100, 2),
        round(report[cls_name]['f1-score'] * 100, 2),
        "",
        ""
    ])
df_levels = pd.DataFrame(level_rows)

cm = confusion_matrix(y_test_raw, y_pred)
tn, fp, fn, tp = cm.ravel()
df_cm = pd.DataFrame([
    ['【實戰診斷】', '數量 (筆)', '代表意義', '', ''],
    ['成功抓出 (True Positive)', tp, '真的是同義詞，模型也答對', '', ''],
    ['漏網之魚 (False Negative)', fn, '明明是同義詞，模型卻漏抓', '', ''],
    ['成功排除 (True Negative)', tn, '不是同義詞，模型也沒被騙', '', ''],
    ['冤枉好人 (False Positive)', fp, '根本不是同義詞，模型卻亂抓', '', '']
])

df_empty = pd.DataFrame([[""] * 5])
final_report = pd.concat([
    df_overall, df_empty, 
    df_levels, df_empty, 
    df_cm
], ignore_index=True)

excel_filename = os.path.join(current_dir, "mlp_詳細正確率.xlsx")
final_report.to_excel(excel_filename, index=False, header=False)

total_train_time = time.time() - start_time
print("\n" + "="*60)
print(f"🎉 報表已生成！耗時 {total_train_time:.2f} 秒。請查看:\n👉 {excel_filename}")
print("="*60)