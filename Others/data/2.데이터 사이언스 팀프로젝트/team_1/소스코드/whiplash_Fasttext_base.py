# -----------------------
# 프로젝트명 : 뉴스 데이터 학습을 통한 가짜 뉴스 분류 
# 사용 모델 : FastText + CNN (불균형 처리 & trainable embedding)
# -----------------------

import os, re
import chardet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import FastText
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, precision_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import cosine

# -----------------------
# Config
# -----------------------
TRAIN_CSV = r"c:\workspaces\fake\data\mission1_train.csv"
TEST_CSV = r"c:\workspaces\fake\data\mission1_test.csv"
EMBED_DIM = 100
MAXLEN = 200
NUM_WORDS = 30000
BATCH_SIZE = 512          # batch 너무 크면 학습 안됨
EPOCHS = 50

# FastText 최적화 옵션
FT_EPOCHS = 5
FT_MIN_COUNT = 2
FT_WINDOW = 5
FT_WORKERS = 4
FT_NGRAMS = 3


# -----------------------
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_split_data(train_path=TRAIN_CSV, test_path=TEST_CSV):
    def safe_read_csv(path):
        with open(path, 'rb') as f:
            enc = chardet.detect(f.read(200000))['encoding']
        print(f"[INFO] Detected encoding for {path}: {enc}")
        tried = []
        for enc_try in [enc, 'euc-kr', 'cp949', 'utf-8-sig', 'utf-8', 'latin1']:
            if not enc_try: continue
            try:
                df = pd.read_csv(path, encoding=enc_try)
                print(f"[INFO] ✅ Successfully loaded with encoding={enc_try}")
                return df
            except Exception as e:
                tried.append(enc_try)
                print(f"[WARN] Failed with {enc_try}: {e}")
        print(f"[FATAL] ❌ All failed: {tried} → using utf-8(errors='replace')")
        return pd.read_csv(path, encoding='utf-8', errors='replace')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/Test CSV 파일을 모두 확인하세요.")
    
    train_df = safe_read_csv(train_path)
    test_df = safe_read_csv(test_path)

    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns = [c.strip().lower() for c in test_df.columns]

    rename_map = {
        '뉴스제목': 'title', '제목': 'title', 'headline': 'title',
        '본문': 'content', '내용': 'content', '기사본문': 'content',
        '가짜뉴스여부': 'label', '라벨': 'label', 'target': 'label',
        'label': 'label', 'Label': 'label'
    }
    train_df.rename(columns=rename_map, inplace=True)
    test_df.rename(columns=rename_map, inplace=True)

    for c in ["title", "content", "label"]:
        if c not in train_df.columns or c not in test_df.columns:
            raise ValueError("CSV에 'title', 'content', 'label' 컬럼이 필요합니다.")

    for df in [train_df, test_df]:
        df["title"] = df["title"].fillna("")
        df["content"] = df["content"].fillna("")
        df["text"] = df["title"] + " " + df["content"]
        df["label"] = df["label"].fillna(0)
        df["label"] = df["label"].replace("", 0)
        df["label"] = df["label"].astype(int)

    print("✅ CSV 로드 완료!")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

# -----------------------
def train_fasttext(sentences, vector_size=EMBED_DIM, window=FT_WINDOW,
                   min_count=FT_MIN_COUNT, workers=FT_WORKERS, epochs=FT_EPOCHS, ngrams=FT_NGRAMS):
    tokenized = [s.split() for s in sentences]
    model = FastText(vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                     min_n=2, max_n=ngrams)
    model.build_vocab(tokenized)
    model.train(tokenized, total_examples=len(tokenized), epochs=epochs)
    return model

def build_embedding_matrix(tokenizer, ft_model, vector_size=EMBED_DIM):
    vocab_size = len(tokenizer.word_index) + 1
    emb = np.random.normal(0, 0.1, (vocab_size, vector_size))
    for w, i in tokenizer.word_index.items():
        if w in ft_model.wv:
            emb[i] = ft_model.wv[w]
    return emb

# -----------------------
def build_model(vocab_size, emb_matrix):
    model = Sequential()
    # ⚡ trainable=True로 변경 → 임베딩 학습 가능
    model.add(Embedding(input_dim=vocab_size, output_dim=emb_matrix.shape[1],
                        weights=[emb_matrix], input_length=MAXLEN, trainable=True))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------
def cosine_similarity_from_fasttext(df, ft_model):
    def sentence_vector(sentence):
        words = [w for w in sentence.split() if w in ft_model.wv]
        if not words: return np.zeros(ft_model.vector_size)
        return np.mean([ft_model.wv[w] for w in words], axis=0)
    sims = []
    for _, row in df.iterrows():
        v1, v2 = sentence_vector(row["title"]), sentence_vector(row["content"])
        if np.all(v1 == 0) or np.all(v2 == 0):
            sims.append(0.0)
        else:
            sims.append(1 - cosine(v1, v2))
    df["cosine_similarity"] = sims
    return df

# -----------------------
def main():
    train_df, test_df = load_split_data()
    X_train, y_train = train_df["text"], train_df["label"].values
    X_test, y_test = test_df["text"], test_df["label"].values

    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAXLEN, padding='post')

    print("Training FastText...")
    ft_model = train_fasttext(X_train.tolist())
    emb_matrix = build_embedding_matrix(tokenizer, ft_model)
    vocab_size = emb_matrix.shape[0]

    test_df = cosine_similarity_from_fasttext(test_df, ft_model)

    model = build_model(vocab_size, emb_matrix)
    model.summary()

    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)

    # ⚡ 클래스 불균형 처리
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train_seq, y_train_cat, validation_split=0.2,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[es], class_weight=class_weights_dict)

    y_prob = model.predict(X_test_seq)[:,1]
    # ⚡ threshold 0.4로 조정 (0.5보다 낮춰서 가짜 뉴스 예측 가능)
    y_pred = (y_prob >= 0.4).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("AUROC:", roc_auc_score(y_test, y_prob))
    print("Pearson r:", pearsonr(y_test, y_prob)[0])
    print(classification_report(y_test, y_pred, digits=4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d'); plt.show()

if __name__ == "__main__":
    main()
