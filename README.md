# Real-Time Gaming Chat Moderation (SWE015 Homework)

**Student:** Egemen Efe Ergün (2309111072)  
**Course:** SWE015 – Introduction to Large Language Models  
**Use Case:** Use Case 4 – Real-Time Chat Moderation  

## Overview
This application detects and masks offensive gaming chat messages in real time on CPU using a four-tier model ladder of increasing complexity:
1. **Classical Baseline:** Regex normalization (URLs, mentions, leetspeak, repeated characters) + Levenshtein edit-distance lexicon + TF-IDF + Logistic Regression
2. **Neural Network:** Custom Skip-Gram Word2Vec embeddings + Feedforward Neural Network trained with SGD and backpropagation
3. **Transformer:** Fine-tuned DistilBERT (`distilbert-base-uncased`)
4. **Quantized Transformer (Default in App):** Dynamic INT8 quantized DistilBERT for low-latency CPU inference

## Benchmark Results
Evaluated on the same held-out test set (`cardiffnlp/tweet_eval` offensive split: **860 test messages**, **11,916 train messages**). Latency is averaged over **100 runs** on CPU (`x86_64`).

| Model | Macro-F1 | Size (MB) | Latency (ms) |
| :--- | :---: | :---: | :---: |
| 1. Classical Baseline (TF-IDF + Regex + Lexicon) | 0.6627 | 0.44 | 3.14 |
| 2. Neural Network (Custom Word2Vec + FFNN) | 0.5920 | 2.12 | 0.86 |
| 3. Transformer (DistilBERT FP32) | 0.7872 | 256.10 | 280.31 |
| 4. Quantized Transformer (DistilBERT INT8) | 0.7555 | 132.29 | 163.64 |

## Setup & Run Instructions
1. Clone this repository and open the project folder:
   `git clone https://github.com/egemenergun/SWE015_ChatModeration.git && cd SWE015_ChatModeration`
2. Install the required Python libraries:
   `pip install -r requirements.txt`
3. Obtain the trained model weights (choose one option):
   - **Option A (Recommended – Fast):** Download `saved_models.zip` from Google Drive (https://drive.google.com/file/d/1pOT3sjAoK8ZdWtca6XdLpGmkiYHYXpK/view?usp=sharing) and extract it into the project root directory so that `saved_models/` is located next to `app.py`.
   - **Option B (Train from scratch):** Run `python train.py` to download the dataset, train all four models, run the 100-iteration CPU benchmark, and save the weights into `saved_models/`.
4. Launch the interactive Streamlit application:
   `streamlit run app.py`