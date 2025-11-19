from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_sample.csv"

def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Expected to find {RAW_DATA_PATH}, but it does not exist.")
    
    df = pd.read_csv(RAW_DATA_PATH)
    df["clean_description"] = df["description"].astype(str).apply(clean_text)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nHead:")
    print(df.head())

    print("\nExample description:")
    print(df.loc[0, "description"])

    print("\nCleaned description example:")
    print(df.loc[0, "clean_description"])

    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_sample_processed.csv"
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nSaved processed data to {PROCESSED_DATA_PATH}")

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.lower()

    return text

if __name__ == "__main__":
    main()