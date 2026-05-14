import argparse
import json
import os
import pickle
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from transformers import BertModel, BertTokenizer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
DEFAULT_INPUT_DIR = os.path.join(CURRENT_DIR, "data_set")
DEFAULT_MODEL_DIR = os.path.join(CURRENT_DIR, "bert")
DEFAULT_DICT_DIR = os.path.join(PROJECT_DIR, "ilrdf_dict_6000")

DATA_FOLDERS = [
    "task_0_vs_12345",
    "task_01_vs_2345",
    "task_012_vs_345",
    "task_0123_vs_45",
    "task_01234_vs_5",
]

LEVEL_SCORES = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, -1: 0.0}
TERM_SPLIT_RE = re.compile(r"[;；、,，/／\n\r\t]+")


class BinaryMLP(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
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
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.network(x)


def split_terms(raw_text):
    if not raw_text:
        return []
    terms = []
    seen = set()
    for term in TERM_SPLIT_RE.split(str(raw_text)):
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def collect_dictionary_terms(data):
    terms = []
    seen = set()
    for entry in data.values():
        for term in split_terms(entry.get("辭典中文", "")):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def collect_ilrdf_terms(dict_path):
    data = load_json(dict_path)
    terms = []
    seen = set()

    def add_terms(raw_text):
        for term in split_terms(raw_text):
            if term not in seen:
                seen.add(term)
                terms.append(term)

    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for sense in entry.get("sense", []) or []:
                if isinstance(sense, dict):
                    add_terms(sense.get("ch_sense", ""))
    return terms


def build_dict_file_map(dict_dir):
    mapping = {}
    for name in os.listdir(dict_dir):
        if name.endswith(".json") and "_ilrdf_dict_" in name:
            mapping[name.split("_", 1)[0]] = os.path.join(dict_dir, name)
    return mapping


def build_bert_vectors(words, tokenizer, bert_model, device, batch_size=256):
    vectors = {}
    words = [w for w in words if w]
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            batch_words = words[start : start + batch_size]
            inputs = tokenizer(
                batch_words,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=16,
            ).to(device)
            outputs = bert_model(**inputs)
            # Keep this aligned with train_mlp_bert.py: mean pooling over tokens.
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            for word, emb in zip(batch_words, embeddings):
                vectors[word] = emb.astype(np.float32)
    return vectors


def make_pair_features(pairs, word_vectors):
    features = np.empty((len(pairs), 1536), dtype=np.float32)
    zero = np.zeros(768, dtype=np.float32)
    for i, (word1, word2) in enumerate(pairs):
        vec1 = word_vectors.get(word1, zero)
        vec2 = word_vectors.get(word2, zero)
        features[i] = np.concatenate([vec1, vec2])
    return features


def load_models(model_dir, device):
    models = []
    for level in range(5):
        path = os.path.join(model_dir, f"level{level}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 BERT level 模型: {path}")
        model = BinaryMLP(1536).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
    return models


def scaler_path(model_dir, level):
    return os.path.join(model_dir, f"level{level}_scaler.pkl")


def load_scalers(model_dir):
    scalers = []
    missing = []
    for level in range(5):
        path = scaler_path(model_dir, level)
        if os.path.exists(path):
            with open(path, "rb") as f:
                scalers.append(pickle.load(f))
        else:
            scalers.append(None)
            missing.append(level)
    return scalers, missing


def fit_missing_scalers(missing_levels, dataset_dir, model_dir, tokenizer, bert_model, device, batch_size=8192):
    for level in missing_levels:
        folder = DATA_FOLDERS[level]
        train_path = os.path.join(dataset_dir, folder, "train.jsonl")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到 scaler 需要的訓練資料: {train_path}")

        print(f"⚖️  level{level} 缺少 scaler，正在依 train.jsonl 重新 fit StandardScaler...")
        pairs = []
        unique_words = set()

        with open(train_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                word1 = item.get("word1", "").strip()
                word2 = item.get("word2", "").strip()
                if not word1 or not word2:
                    continue
                pairs.append((word1, word2))
                unique_words.add(word1)
                unique_words.add(word2)
                if line_no % 100000 == 0:
                    print(f"   ...level{level} scaler 已讀取 {line_no} 筆")

        print(f"   level{level}: {len(pairs)} 組 pair，{len(unique_words)} 個唯一詞，開始建立 BERT 向量")
        vectors = build_bert_vectors(sorted(unique_words), tokenizer, bert_model, device, batch_size=256)

        scaler = StandardScaler()
        for start in range(0, len(pairs), batch_size):
            x_batch = make_pair_features(pairs[start : start + batch_size], vectors)
            scaler.partial_fit(x_batch)
            if start and start % (batch_size * 20) == 0:
                print(f"   ...level{level} scaler 已 fit {start} 組 pair")

        path = scaler_path(model_dir, level)
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"✅ 已儲存 scaler: {path}")


def predict_probabilities(features, models, scalers, device, batch_size=1024):
    all_probs = np.zeros((features.shape[0], 5), dtype=np.float32)
    for level, model in enumerate(models):
        x = features
        if scalers[level] is not None:
            x = scalers[level].transform(features).astype(np.float32)
        probs = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32).to(device)
                output = model(batch)
                probs.append(F.softmax(output, dim=1)[:, 1].cpu().numpy())
        all_probs[:, level] = np.concatenate(probs)
    return all_probs


def decide_level(level_probs, threshold):
    for level, prob in enumerate(level_probs):
        if prob >= threshold:
            return level, LEVEL_SCORES[level], float(prob)
    return -1, 0.0, float(np.max(level_probs))


def normalize_for_match(term):
    return re.sub(r"\s+", "", term or "")


def process_file(path, output_dir, models, scalers, tokenizer, bert_model, device, args, dict_file_map):
    data = load_json(path)
    file_prefix = os.path.basename(path)[:2]
    dict_source_label = "merged_chinese.json: 辭典中文"
    if args.dictionary_source == "ilrdf":
        dict_path = dict_file_map.get(file_prefix)
        if not dict_path:
            raise FileNotFoundError(f"找不到 {os.path.basename(path)} 對應的 ilrdf 字典檔")
        dict_terms = collect_ilrdf_terms(dict_path)
        dict_source_label = os.path.basename(dict_path)
    else:
        dict_terms = collect_dictionary_terms(data)
    if not dict_terms:
        print(f"⚠️  {os.path.basename(path)} 沒有可比對的辭典中文詞，略過。")
        return None

    new_words = []
    for entry in data.values():
        new_word = (entry.get("新詞中文") or "").strip()
        if new_word and new_word not in new_words:
            new_words.append(new_word)

    all_words = sorted(set(new_words + dict_terms))
    print(
        f"🧠 {os.path.basename(path)}: {len(new_words)} 個新詞 × "
        f"{len(dict_terms)} 個辭典詞 ({dict_source_label})"
    )
    word_vectors = build_bert_vectors(all_words, tokenizer, bert_model, device, batch_size=args.bert_batch_size)

    summary_rows = []
    result_data = OrderedDict()

    for key, entry in data.items():
        new_word = (entry.get("新詞中文") or "").strip()
        original_terms = split_terms(entry.get("辭典中文", ""))
        original_norms = {normalize_for_match(t) for t in original_terms}

        if not new_word:
            result_data[key] = entry
            continue

        pairs = [(new_word, dict_word) for dict_word in dict_terms]
        features = make_pair_features(pairs, word_vectors)
        probs = predict_probabilities(features, models, scalers, device, batch_size=args.predict_batch_size)

        candidates = []
        for dict_word, level_probs in zip(dict_terms, probs):
            level, level_score, tie_prob = decide_level(level_probs, args.threshold)
            candidates.append(
                {
                    "dict_word": dict_word,
                    "level": level,
                    "level_score": level_score,
                    "tie_prob": tie_prob,
                    "max_prob": float(np.max(level_probs)),
                    "level_probs": [float(x) for x in level_probs],
                }
            )

        candidates.sort(
            key=lambda item: (
                item["level_score"],
                item["tie_prob"],
                item["max_prob"],
            ),
            reverse=True,
        )
        best = candidates[0]
        is_original = normalize_for_match(best["dict_word"]) in original_norms

        top_candidates = candidates[: args.top_k]
        result_entry = entry
        result_entry["BERT全字典比對結果"] = {
            "新詞中文": new_word,
            "比對字典來源": dict_source_label,
            "原本辭典中文": original_terms,
            "最佳配對辭典詞": best["dict_word"],
            "是否命中原本使用詞": bool(is_original),
            "判定Level": best["level"] if best["level"] != -1 else "都不是",
            "判定分數": best["level_score"],
            "同Level排序機率": best["tie_prob"],
            "五層Level機率": {
                f"level{i}": best["level_probs"][i] for i in range(5)
            },
            "Top候選": [
                {
                    "辭典詞": item["dict_word"],
                    "Level": item["level"] if item["level"] != -1 else "都不是",
                    "分數": item["level_score"],
                    "同Level排序機率": item["tie_prob"],
                    "五層Level機率": {
                        f"level{i}": item["level_probs"][i] for i in range(5)
                    },
                }
                for item in top_candidates
            ],
        }
        result_data[key] = result_entry

        summary_rows.append(
            {
                "檔案": os.path.basename(path),
                "比對字典來源": dict_source_label,
                "key": key,
                "新詞中文": new_word,
                "原本辭典中文": "；".join(original_terms),
                "最佳配對辭典詞": best["dict_word"],
                "是否命中原本使用詞": bool(is_original),
                "判定Level": best["level"] if best["level"] != -1 else "都不是",
                "判定分數": best["level_score"],
                "同Level排序機率": best["tie_prob"],
                "level0機率": best["level_probs"][0],
                "level1機率": best["level_probs"][1],
                "level2機率": best["level_probs"][2],
                "level3機率": best["level_probs"][3],
                "level4機率": best["level_probs"][4],
            }
        )

    output_prefix = "bert_ilrdf_all_pairs" if args.dictionary_source == "ilrdf" else "bert_all_pairs"
    output_json = os.path.join(output_dir, f"{output_prefix}_{os.path.basename(path)}")
    dump_json(output_json, result_data)
    print(f"✅ 已輸出: {output_json}")
    return summary_rows


def resolve_input_dir(raw_input_dir):
    if os.path.exists(raw_input_dir):
        return raw_input_dir
    fallback = DEFAULT_INPUT_DIR
    if os.path.exists(fallback):
        print(f"⚠️  找不到指定資料夾 {raw_input_dir}，改用 {fallback}")
        return fallback
    raise FileNotFoundError(f"找不到輸入資料夾: {raw_input_dir}")


def main():
    parser = argparse.ArgumentParser(description="BERT 版 16 族新詞全字典比對實驗")
    parser.add_argument("--input-dir", default=os.path.join(PROJECT_DIR, "model_training", "data_set"))
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dict-dir", default=DEFAULT_DICT_DIR)
    parser.add_argument("--dictionary-source", choices=["ilrdf", "merged"], default="ilrdf")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bert-batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--no-fit-missing-scalers", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0)
    args = parser.parse_args()

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用運算裝置: {device}")
    print("📦 載入 bert-base-chinese...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    bert_model = BertModel.from_pretrained("bert-base-chinese").to(device)
    bert_model.eval()

    models = load_models(args.model_dir, device)
    dict_file_map = build_dict_file_map(args.dict_dir) if args.dictionary_source == "ilrdf" else {}
    scalers, missing = load_scalers(args.model_dir)
    if missing and args.no_fit_missing_scalers:
        print(f"⚠️  缺少 scaler: {missing}，將直接使用未標準化特徵推論。")
    elif missing:
        fit_missing_scalers(missing, args.dataset_dir, args.model_dir, tokenizer, bert_model, device)
        scalers, missing = load_scalers(args.model_dir)

    files = [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.endswith(".json")
        and not name.startswith("result_")
        and not name.startswith("bert_all_pairs_")
        and not name.startswith("bert_ilrdf_all_pairs_")
    ]
    if args.limit_files:
        files = files[: args.limit_files]

    all_rows = []
    for path in files:
        rows = process_file(path, output_dir, models, scalers, tokenizer, bert_model, device, args, dict_file_map)
        if rows:
            all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        excel_name = (
            "BERT_16族ILRDF完整字典比對結果.xlsx"
            if args.dictionary_source == "ilrdf"
            else "BERT_16族全字典比對結果.xlsx"
        )
        output_excel = os.path.join(output_dir, excel_name)
        hit_rate = (
            df.groupby("檔案")["是否命中原本使用詞"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "新詞數", "sum": "命中數"})
        )
        hit_rate["命中率"] = hit_rate["命中數"] / hit_rate["新詞數"]
        with pd.ExcelWriter(output_excel) as writer:
            df.to_excel(writer, sheet_name="逐詞最佳結果", index=False)
            hit_rate.to_excel(writer, sheet_name="各族命中率", index=False)
        print(f"📊 已輸出 Excel: {output_excel}")


if __name__ == "__main__":
    main()
