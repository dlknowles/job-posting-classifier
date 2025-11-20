from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_full_processed.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert-job-classifier"

# ---------- Data loading & balancing ----------

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Expect columns: "text", "label"
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected 'text' and 'label' columns in {DATA_PATH}")
    return df

def sample_balanced(df: pd.DataFrame):
    # Same target sizes as the baseline
    target = {
        "AI_ML": 2000,
        "SWE": 4000,
        "DEVOPS": 5000,
        "OTHER": 5000,
    }

    dfs = []
    for label, size in target.items():
        subset = df[df["label"] == label]
        if subset.empty:
            continue
        sampled = subset.sample(
            n=min(len(subset), size),
            random_state=42,
        )
        dfs.append(sampled)

    balanced_df = pd.concat(dfs).sample(frac=1, random_state=42)
    return balanced_df

# ---------- Dataset wrapper ----------

class JobPostingsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256,):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item

# ---------- Metrics ----------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

# ---------- Main training pipeline ----------

def main():
    print(f"Loading data from {DATA_PATH}")

    df = load_data()
    df_balanced = sample_balanced(df)

    print("Balanced label distribution:")
    print(df_balanced["label"].value_counts())

    # Encode labels
    unique_labels = sorted(df_balanced["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print("Label mapping:", label2id)

    texts = df_balanced["text"].astype(str).tolist()
    labels = [label2id[l] for l in df_balanced["label"]]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = JobPostingsDataset(X_train, y_train, tokenizer)
    eval_dataset = JobPostingsDataset(X_test, y_test, tokenizer)
    num_labels = len(unique_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # Extra: full classification report for inspection
    preds_output = trainer.predict(eval_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\nClassification report (DistilBERT):")
    print(classification_report(y_test, preds, target_names=unique_labels))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    print(f"Saving model to {MODEL_OUTPUT_DIR}")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # Save label mapping for inference
    label_map_path = MODEL_OUTPUT_DIR / "label_mapping.json"
    import json
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            f,
            indent=2,
        )
    print(f"Saved label mapping to {label_map_path}")

if __name__ == "__main__":
    main()
