import json
import sys
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-job-classifier"
LABEL_MAPPING_PATH = MODEL_DIR / "label_mapping.json"

def load_model_and_tokenizer():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    print(f"Loading model from {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    if LABEL_MAPPING_PATH.exists():
        with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping: Dict = json.load(f)
        id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        # Fallback to config if JSON missing
        id2label = model.config.id2label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, id2label, device

def predict(text: str):
    model, tokenizer, id2label, device = load_model_and_tokenizer()

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id = int(probs.argmax())
    pred_label = id2label[pred_id]
    pred_conf = float(probs[pred_id])

    return {
        "label": pred_label,
        "confidence": pred_conf,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict \"job description text...\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    result = predict(text)
    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()
