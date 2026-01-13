import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


DATA_PATH = "data/processed/result.csv"
MODEL_PATH = "data/processed/model.pkl"
FEATURES = ["avg_day_noise", "avg_night_noise", "wgs84Lat", "wgs84Lon"]
TARGET = "need_improvement"

def train_ml_model():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    train_ml_model()