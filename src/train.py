import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_full_processed.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def sample_balanced(df):
    # Target approximate sizes
    target = {
        "AI_ML": 2000,
        "SWE": 4000,
        "DEVOPS": 5000,
        "OTHER": 5000,
    }

    dfs = []
    for label, size in target.items():
        subset = df[df["label"] == label].sample(
            n = min(len(df[df["label"] == label]), size),
            random_state = 42
        )
        dfs.append(subset)

    balanced_df = pd.concat(dfs).sample(frac=1, random_state=42)
    return balanced_df

def train_baseline(df):
    x = df["text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=50000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train_vec, y_train)

    preds = model.predict(x_test_vec)

    print("Classification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return model, vectorizer

def main():
    df = load_data()
    df_balanced = sample_balanced(df)

    print("Balanced training distribution:")
    print(df_balanced["label"].value_counts())

    model, vectorizer = train_baseline(df_balanced)

if __name__ == "__main__":
    main()