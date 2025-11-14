# =========================================
# 📰 프로젝트명 : 뉴스 데이터 학습을 통한 가짜 뉴스 분류
# 개선 목표 : AUROC ≥ 0.5 + Cosine Similarity 분석 추가
# =========================================
# ✅ 주요 기능 요약
# 1. 텍스트 전처리 및 데이터 로드
# 2. Word2Vec 학습 및 임베딩 매트릭스 생성
# 3. BiLSTM + CNN 하이브리드 모델 구성
# 4. AUROC, Precision, Pearson r 계산
# 5. 제목-본문 코사인 유사도 분석 + 시각화
# =========================================

import os, re
import chardet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    accuracy_score, confusion_matrix
)
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, concatenate,
    Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# -----------------------
# Config
# -----------------------
TRAIN_CSV = r"c:\workspaces\fake\data\mission1_train.csv"
TEST_CSV = r"c:\workspaces\fake\data\mission1_test.csv"
EMBED_DIM = 128
MAXLEN = 500
NUM_WORDS = 30000
BATCH_SIZE = 512
EPOCHS = 100
FILTER_SIZE = 3
NUM_FILTERS = 1024
DROPOUT_RATE = 0.4
L2_ALPHA = 0.001

# -----------------------
# 텍스트 정제
# -----------------------
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    stopwords = ["기자","사진","연합뉴스","뉴스","보도","오늘","대한","에서"]
    for sw in stopwords:
        text = text.replace(sw, "")
    return text

# -----------------------
# 데이터 로드
# -----------------------
def load_split_data(train_path=TRAIN_CSV, test_path=TEST_CSV):
    def safe_read_csv(path):
        with open(path, 'rb') as f:
            enc = chardet.detect(f.read(200000))['encoding']
        for enc_try in [enc, 'utf-8-sig', 'euc-kr', 'cp949']:
            try:
                return pd.read_csv(path, encoding=enc_try)
            except: continue
        return pd.read_csv(path, encoding='utf-8', errors='replace')

    train_df = safe_read_csv(train_path)
    test_df = safe_read_csv(test_path)
    train_df.columns = [c.lower().strip() for c in train_df.columns]
    test_df.columns = [c.lower().strip() for c in test_df.columns]
    rename_map = {'뉴스제목':'title','본문':'content','가짜뉴스여부':'label'}
    train_df.rename(columns=rename_map, inplace=True)
    test_df.rename(columns=rename_map, inplace=True)

    for df in [train_df, test_df]:
        df["title"] = df["title"].fillna("").apply(clean_text)
        df["content"] = df["content"].fillna("").apply(clean_text)
        df["text"] = (df["title"] + " " + df["content"]).str.strip()
        df["label"] = df["label"].fillna(0).astype(int)

    print("✅ CSV 로드 완료:", train_df.shape, test_df.shape)
    return train_df, test_df

# -----------------------
# Word2Vec 학습
# -----------------------
def train_word2vec(sentences, vector_size=EMBED_DIM, window=8, min_count=1):
    tokenized = [s.split() for s in sentences]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size,
                     window=window, min_count=min_count, workers=4, sg=1)
    return model

# -----------------------
# 임베딩 매트릭스 구성
# -----------------------
def build_embedding_matrix(tokenizer, w2v_model, vector_size=EMBED_DIM):
    vocab_size = len(tokenizer.word_index) + 1
    emb = np.random.normal(0, 0.1, (vocab_size, vector_size))
    for w, i in tokenizer.word_index.items():
        if w in w2v_model.wv:
            emb[i] = w2v_model.wv[w]
    return emb

# -----------------------
# 코사인 유사도 계산
# -----------------------
def cosine_similarity_from_word2vec(df, w2v_model):
    def sentence_vector(sentence):
        words = [w for w in sentence.split() if w in w2v_model.wv]
        if not words: return np.zeros(w2v_model.vector_size)
        return np.mean([w2v_model.wv[w] for w in words], axis=0)
    sims = []
    for _, row in df.iterrows():
        v1, v2 = sentence_vector(row["title"]), sentence_vector(row["content"])
        sims.append(0 if np.all(v1==0) or np.all(v2==0) else 1 - cosine(v1, v2))
    df["cosine_similarity"] = sims
    return df

# -----------------------
# 하이브리드 모델 정의 (BiLSTM + CNN)
# -----------------------
def build_hybrid_model(vocab_size, emb_matrix):
    inp = Input(shape=(MAXLEN,))
    x = Embedding(input_dim=vocab_size, output_dim=emb_matrix.shape[1],
                  weights=[emb_matrix], input_length=MAXLEN, trainable=True)(inp)
    
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    conv3 = Conv1D(64, 3, activation='relu', padding='same')(lstm_out)
    conv4 = Conv1D(64, 4, activation='relu', padding='same')(lstm_out)
    conv5 = Conv1D(64, 5, activation='relu', padding='same')(lstm_out)
    
    merged = concatenate([GlobalMaxPooling1D()(conv3),
                          GlobalMaxPooling1D()(conv4),
                          GlobalMaxPooling1D()(conv5)])
    merged = BatchNormalization()(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    out = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------
# 학습 곡선 시각화
# -----------------------
def plot_training_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title("Accuracy Curve")

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss Curve")
    plt.show()

# -----------------------
# 코사인 유사도 분석 시각화
# -----------------------
def visualize_cosine_similarity(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df["cosine_similarity"], bins=30, kde=True, color='skyblue')
    plt.title("Cosine Similarity Distribution (Title vs Content)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()

    print("\n📊 코사인 유사도 통계:")
    print(df["cosine_similarity"].describe())

    top5 = df.sort_values("cosine_similarity", ascending=False).head(5)[["title", "cosine_similarity"]]
    low5 = df.sort_values("cosine_similarity", ascending=True).head(5)[["title", "cosine_similarity"]]
    print("\n🟢 유사도 상위 기사 5개:")
    print(top5.to_string(index=False))
    print("\n🔴 유사도 하위 기사 5개:")
    print(low5.to_string(index=False))

# -----------------------
# 메인 로직
# -----------------------
def main():
    # 1️⃣ 데이터 로드 및 전처리
    train_df, test_df = load_split_data()
    X_train, y_train = train_df["text"], train_df["label"].values
    X_test, y_test = test_df["text"], test_df["label"].values

    # 2️⃣ 토크나이저 및 시퀀스 변환
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAXLEN, padding='post')

    # 3️⃣ Word2Vec 학습 및 임베딩 구성
    print("Training Word2Vec...")
    w2v = train_word2vec(X_train.tolist())
    emb_matrix = build_embedding_matrix(tokenizer, w2v)
    vocab_size = emb_matrix.shape[0]

    # 4️⃣ 제목-본문 코사인 유사도 계산
    test_df = cosine_similarity_from_word2vec(test_df, w2v)

    # 5️⃣ 모델 구성 및 학습
    model = build_hybrid_model(vocab_size, emb_matrix)
    model.summary()

    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    history = model.fit(
        X_train_seq, y_train_cat, validation_split=0.2,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, rl],
        class_weight=class_weights, verbose=1
    )

    # 6️⃣ 예측 및 성능 평가
    y_prob = model.predict(X_test_seq)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n🎯 평가 결과")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("AUROC:", roc_auc_score(y_test, y_prob))
    print("Pearson r:", pearsonr(y_test, y_prob)[0])
    print(classification_report(y_test, y_pred, digits=4))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

    plot_training_history(history)

    # 7️⃣ 코사인 유사도 분석 시각화
    visualize_cosine_similarity(test_df)

# -----------------------
if __name__ == "__main__":
    main()
