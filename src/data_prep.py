from pathlib import Path
import pandas as pd
from typing import Literal
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_full.csv"

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Expected to find {RAW_DATA_PATH}, but it does not exist.")
    
    title_col = "title"
    desc_col = "description"

    df = pd.read_csv(RAW_DATA_PATH)
    df["clean_description"] = df[desc_col].astype(str).apply(clean_text)

    df[title_col] = df[title_col].astype(str)
    df[desc_col] = df[desc_col].astype(str)

    df["label"] = [label_row(t, d) for t, d in zip(df[title_col], df[desc_col])]

    df["text"] = (df[title_col] + " " + df[desc_col]).astype(str)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nHead:")
    print(df.head())

    print("\nExample description:")
    print(df.loc[0, desc_col])

    print("\nCleaned description example:")
    print(df.loc[0, "clean_description"])

    print("\nLabels:")
    print(df["label"].value_counts())

    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_full_processed.csv"
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_processed = df[["text", "label"]].copy()
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nSaved processed data to {PROCESSED_DATA_PATH}")

def clean_text(text: str):
    if text is None:
        return ""
    
    return re.sub(r"\s+", " ", text).lower().strip()

def label_row(title: str, description: str):
    text = f"{title} {description}".lower()

    ai_ml_keywords = [
        "machine learning",
        "ml engineer",
        "ai engineer",
        "data scientist",
        "deep learning",
        "computer vision",
        "nlp",
        "natural language processing",
        "mlops"
    ]

    devops_keywords = [
        "devops",
        "site reliability",
        "sre",
        "infrastructure",
        "kubernetes",
        "k8s",
        "docker",
        "ci/cd",
        "terraform",
    ]

    swe_keywords = [
        "software engineer",
        "software developer",
        "full stack",
        "backend engineer",
        "frontend engineer",
        "api developer",
        ".net",
        "c#",
        "java developer",
        "python developer",
        "golang",
        "react",
        "node.js",
        "typescript",
    ]

    if any(k in text for k in ai_ml_keywords):
        return "AI_ML"
    if any(k in text for k in devops_keywords):
        return "DEVOPS"
    if any(k in text for k in swe_keywords):
        return "SWE"
    
    return "OTHER"

if __name__ == "__main__":
    main()