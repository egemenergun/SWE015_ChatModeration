import os
import re
import time
import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Real-Time Chat Moderation", page_icon="🛡️", layout="wide")

MODEL_DIR = (
    "saved_models/saved_models"
    if os.path.exists("saved_models/saved_models/baseline.joblib")
    else "saved_models"
)

# --- 1. KLASİK NLP ÖN İŞLEME VE LEXICON (AREA B) ---
LEET_MAP = str.maketrans({"@": "a", "3": "e", "1": "i", "!": "i", "0": "o", "$": "s", "5": "s", "7": "t"})
BAD_LEXICON = {
    "idiot", "stupid", "moron", "trash", "loser", "shut", "hate",
    "dumb", "suck", "hell", "damn", "bitch", "asshole", "fuck", "shit"
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.translate(LEET_MAP)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def edit_distance(s1: str, s2: str) -> int:
    dp = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        prev, dp[0] = dp[0], i
        for j, c2 in enumerate(s2, 1):
            temp = dp[j]
            dp[j] = prev if c1 == c2 else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[-1]

def lexicon_flag(norm_txt: str) -> int:
    for w in norm_txt.split():
        if len(w) >= 4:
            for bad in BAD_LEXICON:
                if abs(len(w) - len(bad)) <= 1 and edit_distance(w, bad) <= 1:
                    return 1
    return 0

# --- 2. NEURAL NETWORK MİMARİSİ (AREA A) ---
class FeedforwardModerationNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        avg = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc2(self.dropout(self.relu(self.fc1(avg))))

# --- 3. TÜM MODELLERİ YÜKLE ---
@st.cache_resource
def load_all_models():
    tfidf, clf_baseline = joblib.load(f"{MODEL_DIR}/baseline.joblib")

    ff_ckpt = torch.load(f"{MODEL_DIR}/ff_nn.pt", map_location="cpu", weights_only=False)
    vocab = ff_ckpt["vocab"]
    ff_model = FeedforwardModerationNet(len(vocab), ff_ckpt["emb_dim"])
    ff_model.load_state_dict(ff_ckpt["state_dict"])
    ff_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/distilbert_fp32")
    bert_fp32 = AutoModelForSequenceClassification.from_pretrained(f"{MODEL_DIR}/distilbert_fp32")
    bert_fp32.eval()

    try:
        torch.backends.quantized.engine = "fbgemm"
    except Exception:
        pass
    bert_base_for_q = AutoModelForSequenceClassification.from_pretrained(f"{MODEL_DIR}/distilbert_fp32")
    bert_int8 = torch.ao.quantization.quantize_dynamic(
        bert_base_for_q, {torch.nn.Linear}, dtype=torch.qint8
    )
    bert_int8.load_state_dict(
        torch.load(f"{MODEL_DIR}/distilbert_int8.pt", map_location="cpu", weights_only=False)
    )
    bert_int8.eval()

    return tfidf, clf_baseline, vocab, ff_model, tokenizer, bert_fp32, bert_int8

tfidf, clf_baseline, vocab, ff_model, tokenizer, bert_fp32, bert_int8 = load_all_models()

def run_inference(text: str, model_choice: str):
    t0 = time.perf_counter()

    if model_choice.startswith("4."):
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = bert_int8(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            probs = torch.softmax(logits, dim=1)[0].numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

    elif model_choice.startswith("3."):
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = bert_fp32(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            probs = torch.softmax(logits, dim=1)[0].numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

    elif model_choice.startswith("2."):
        nt = normalize_text(text)
        ids = [vocab.get(w, 1) for w in nt.split()[:40]]
        arr = np.zeros((1, 40), dtype=np.int64)
        arr[0, :len(ids)] = ids
        x_tensor = torch.tensor(arr, dtype=torch.long)
        with torch.no_grad():
            logits = ff_model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

    else:
        nt = normalize_text(text)
        vec = tfidf.transform([nt])
        off_prob = float(clf_baseline.predict_proba(vec)[0][1])
        if lexicon_flag(nt) == 1:
            off_prob = max(off_prob, 0.85)
        pred = int(off_prob >= 0.5)
        conf = off_prob if pred == 1 else (1.0 - off_prob)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return pred, conf, latency_ms

# --- 4. SOHBET ARAYÜZÜ VE YAN PANEL ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = []

with st.sidebar:
    st.header("⚙️ Model & Latency Panel")
    selected_model = st.selectbox(
        "Active Moderation Model",
        [
            "4. Quantized Transformer (DistilBERT INT8) [Default]",
            "3. Transformer (DistilBERT FP32)",
            "2. Neural Network (Custom Word2Vec + FFNN)",
            "1. Classical Baseline (TF-IDF + Regex + Lexicon)",
        ],
        index=0
    )
    st.caption("Varsayılan olarak CPU üzerinde optimize edilmiş Quantized INT8 modeli çalışır.")

    if st.button("Clear Chat & Logs"):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()

    st.divider()
    st.subheader("⏱️ Live Response Time Log")
    if st.session_state.logs:
        for item in reversed(st.session_state.logs):
            badge = "🔴 OFFENSIVE" if item["label"] == "OFFENSIVE" else "🟢 CLEAN"
            st.markdown(
                f"**{badge}** ({item['conf']:.1f}%) — `{item['latency']:.1f} ms`  \n"
                f"<small>Model: {item['model']} | Msg: \"{item['short_msg']}\"</small>",
                unsafe_allow_html=True
            )
            st.markdown("---")
    else:
        st.info("Henüz mesaj gönderilmedi.")

st.title("🛡️ Real-Time Gaming Chat Moderation")
st.write("Mesaj yazın; sistem canlı olarak **Clean / Offensive** analizi yapar, sakıncalı mesajları maskeler ve yan panelde gecikme süresini (ms) kaydeder.")

for msg in st.session_state.messages:
    with st.chat_message("user"):
        if msg["is_offensive"]:
            st.error(f"🚫 **[MESSAGE MASKED — OFFENSIVE]** *(Confidence: {msg['conf']:.1f}% | {msg['latency']:.1f} ms | {msg['model']})*")
        else:
            st.write(msg["text"])
            st.caption(f"✅ CLEAN *(Confidence: {msg['conf']:.1f}% | {msg['latency']:.1f} ms | {msg['model']})*")

user_input = st.chat_input("Type a chat message to test moderation...")

if user_input:
    pred, conf, latency_ms = run_inference(user_input, selected_model)
    is_off = (pred == 1)
    short_name = selected_model.split(" (")[0]

    st.session_state.messages.append({
        "text": user_input,
        "is_offensive": is_off,
        "conf": conf * 100,
        "latency": latency_ms,
        "model": short_name
    })
    st.session_state.logs.append({
        "short_msg": (user_input[:28] + "...") if len(user_input) > 28 else user_input,
        "label": "OFFENSIVE" if is_off else "CLEAN",
        "conf": conf * 100,
        "latency": latency_ms,
        "model": short_name
    })
    st.rerun()