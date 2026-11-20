import os
import re
import time
import json
import random
import platform
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("saved_models", exist_ok=True)

# 1. Dataset (cardiffnlp/tweet_eval - offensive)
dataset = load_dataset("cardiffnlp/tweet_eval", "offensive")
train_texts = list(dataset["train"]["text"])
train_labels = list(dataset["train"]["label"])
test_texts = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

# 2. Area B: Regex Normalization + Edit-Distance Lexicon + TF-IDF Baseline
LEET_MAP = str.maketrans({"@": "a", "3": "e", "1": "i", "!": "i", "0": "o", "$": "s", "5": "s", "7": "t"})
BAD_LEXICON = {"idiot", "stupid", "moron", "trash", "loser", "shut", "hate", "dumb", "suck", "hell", "damn", "bitch", "asshole", "fuck", "shit"}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.translate(LEET_MAP)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def edit_distance(s1, s2):
    dp = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        prev, dp[0] = dp[0], i
        for j, c2 in enumerate(s2, 1):
            temp = dp[j]
            dp[j] = prev if c1 == c2 else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[-1]

def lexicon_flag(norm_txt):
    for w in norm_txt.split():
        if len(w) >= 4:
            for bad in BAD_LEXICON:
                if abs(len(w) - len(bad)) <= 1 and edit_distance(w, bad) <= 1:
                    return 1
    return 0

train_norm = [normalize_text(t) for t in train_texts]
test_norm = [normalize_text(t) for t in test_texts]

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_norm)
clf_baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf_baseline.fit(X_train_tfidf, train_labels)

def predict_baseline(raw_text):
    nt = normalize_text(raw_text)
    vec = tfidf.transform([nt])
    prob = clf_baseline.predict_proba(vec)[0][1]
    final_prob = max(prob, 0.85) if lexicon_flag(nt) == 1 else prob
    return int(final_prob >= 0.5), float(final_prob)

baseline_preds = [predict_baseline(t)[0] for t in test_texts]
baseline_f1 = f1_score(test_labels, baseline_preds, average="macro")
joblib.dump((tfidf, clf_baseline), "saved_models/baseline.joblib")
baseline_size_mb = os.path.getsize("saved_models/baseline.joblib") / (1024 * 1024)

# 3. Area A: Custom Skip-Gram Word2Vec + Feedforward Neural Network (SGD)
word_counts = {}
for t in train_norm:
    for w in t.split():
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = {"<PAD>": 0, "<UNK>": 1}
for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
    if c >= 2 and len(vocab) < 8000:
        vocab[w] = len(vocab)

pairs = []
for t in train_norm:
    ids = [vocab.get(w, 1) for w in t.split()]
    for idx, center in enumerate(ids):
        if center <= 1:
            continue
        for ctx in ids[max(0, idx - 2): min(len(ids), idx + 3)]:
            if ctx != center and ctx > 1:
                pairs.append((center, ctx))

random.shuffle(pairs)
pairs = pairs[:80000]

EMB_DIM = 64
class SkipGramW2V(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, emb_dim)
        self.out_emb = nn.Linear(emb_dim, vocab_size)
    def forward(self, x):
        return self.out_emb(self.in_emb(x))

w2v = SkipGramW2V(len(vocab), EMB_DIM).to(device)
w2v_opt = torch.optim.SGD(w2v.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

pair_tensor = torch.tensor(pairs, dtype=torch.long)
for epoch in range(3):
    for i in range(0, len(pair_tensor), 512):
        batch = pair_tensor[i:i+512].to(device)
        w2v_opt.zero_grad()
        loss = loss_fn(w2v(batch[:, 0]), batch[:, 1])
        loss.backward()
        w2v_opt.step()

pretrained_weights = w2v.in_emb.weight.detach().cpu()

class FeedforwardModerationNet(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.fc1 = nn.Linear(weights.shape[1], 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        avg = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc2(self.dropout(self.relu(self.fc1(avg))))

def encode_texts(texts, max_len=40):
    arr = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()[:max_len]]
        arr[i, :len(ids)] = ids
    return torch.tensor(arr, dtype=torch.long)

X_train_nn = encode_texts(train_norm)
y_train_nn = torch.tensor(train_labels, dtype=torch.long)
X_test_nn = encode_texts(test_norm)

ff_model = FeedforwardModerationNet(pretrained_weights).to(device)
ff_opt = torch.optim.SGD(ff_model.parameters(), lr=0.05, momentum=0.9)
ff_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], device=device))

ff_model.train()
for epoch in range(12):
    perm = torch.randperm(len(X_train_nn))
    for i in range(0, len(X_train_nn), 64):
        idx = perm[i:i+64]
        bx, by = X_train_nn[idx].to(device), y_train_nn[idx].to(device)
        ff_opt.zero_grad()
        loss = ff_criterion(ff_model(bx), by)
        loss.backward()
        ff_opt.step()

ff_model.cpu().eval()
with torch.no_grad():
    ff_preds = ff_model(X_test_nn).argmax(dim=1).numpy()
ff_f1 = f1_score(test_labels, ff_preds, average="macro")
torch.save({"state_dict": ff_model.state_dict(), "vocab": vocab, "emb_dim": EMB_DIM}, "saved_models/ff_nn.pt")
ff_size_mb = os.path.getsize("saved_models/ff_nn.pt") / (1024 * 1024)

# 4. Area C: Fine-Tuned Transformer (DistilBERT)
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i], "labels": self.labels[i]}

train_loader = DataLoader(TextDataset(train_texts, train_labels), batch_size=32, shuffle=True)
test_loader = DataLoader(TextDataset(test_texts, test_labels), batch_size=64)

bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)

bert_model.train()
for epoch in range(2):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = bert_model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
        outputs.loss.backward()
        optimizer.step()

bert_model.cpu().eval()
bert_preds = []
with torch.no_grad():
    for batch in test_loader:
        bert_preds.extend(bert_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.argmax(dim=1).numpy())
bert_f1 = f1_score(test_labels, bert_preds, average="macro")
bert_model.save_pretrained("saved_models/distilbert_fp32")
tokenizer.save_pretrained("saved_models/distilbert_fp32")
bert_size_mb = sum(os.path.getsize(os.path.join("saved_models/distilbert_fp32", f)) for f in os.listdir("saved_models/distilbert_fp32")) / (1024 * 1024)

# 5. Area D: Dynamic INT8 Quantization
quantized_bert = torch.ao.quantization.quantize_dynamic(bert_model, {torch.nn.Linear}, dtype=torch.qint8)
quantized_bert.eval()
q_preds = []
with torch.no_grad():
    for batch in test_loader:
        q_preds.extend(quantized_bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.argmax(dim=1).numpy())
q_f1 = f1_score(test_labels, q_preds, average="macro")
torch.save(quantized_bert.state_dict(), "saved_models/distilbert_int8.pt")
q_size_mb = os.path.getsize("saved_models/distilbert_int8.pt") / (1024 * 1024)

# 6. Benchmark (100 runs on CPU)
sample_msg = "You are such a stupid loser, shut up!"
sample_enc = tokenizer(sample_msg, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
sample_nn = encode_texts([normalize_text(sample_msg)])

def measure_ms(fn, runs=120, warmup=20):
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    return float(np.mean(times))

results = [
    {"Model": "1. Classical Baseline (TF-IDF + Regex + Lexicon)", "Macro-F1": round(baseline_f1, 4), "Size (MB)": round(baseline_size_mb, 2), "Latency (ms)": round(measure_ms(lambda: predict_baseline(sample_msg)), 2)},
    {"Model": "2. Neural Network (Custom Word2Vec + FFNN)", "Macro-F1": round(ff_f1, 4), "Size (MB)": round(ff_size_mb, 2), "Latency (ms)": round(measure_ms(lambda: ff_model(sample_nn)), 2)},
    {"Model": "3. Transformer (DistilBERT FP32)", "Macro-F1": round(bert_f1, 4), "Size (MB)": round(bert_size_mb, 2), "Latency (ms)": round(measure_ms(lambda: bert_model(input_ids=sample_enc["input_ids"], attention_mask=sample_enc["attention_mask"])), 2)},
    {"Model": "4. Quantized Transformer (DistilBERT INT8)", "Macro-F1": round(q_f1, 4), "Size (MB)": round(q_size_mb, 2), "Latency (ms)": round(measure_ms(lambda: quantized_bert(input_ids=sample_enc["input_ids"], attention_mask=sample_enc["attention_mask"])), 2)},
]

with open("saved_models/benchmark_results.json", "w") as f:
    json.dump({"hardware": platform.processor() or platform.machine(), "results": results}, f, indent=2)
print("Benchmark finished:", results)