# Job Posting Classifier

Classify job postings into four categories:

- `AI_ML` – AI / Machine Learning / Data Science–type roles  
- `SWE` – Software Engineer / Developer roles  
- `DEVOPS` – DevOps / SRE / infra-focused roles  
- `OTHER` – Everything else

The goal is **not** to build a perfect model, but to demonstrate a complete, realistic NLP pipeline:

- data ingestion and preprocessing  
- auto-labeling via rules  
- baseline classical ML model  
- enhanced transformer model (DistilBERT)  
- local inference script for predictions

This project is a compact example of how to go from raw text to a working classifier that can be queried from the command line.

___

## Data

This project expects a raw dataset located at:
data/raw/jobs_full.csv


It is not included in the repository.

You can download it from:
[Kaggle](https://www.kaggle.com/datasets/saidddd65215/real-data-job-posting)

After downloading:
1. Place it in `data/raw/` and rename to jobs_full.csv
2. Run:  
   ```bash
   python -m src.data_prep


---

## Project Structure

```text
job-posting-classifier/
  data/
    raw/
      jobs_full.csv                 # original job postings dataset
    processed/
      jobs_full_processed.csv       # cleaned + labeled data
  models/
    distilbert-job-classifier/      # saved DistilBERT model + tokenizer + label mapping
  src/
    __init__.py
    data_prep.py                    # data loading, cleaning, auto-labeling
    train.py                        # TF-IDF + LogisticRegression baseline
    train_distilbert.py             # DistilBERT fine-tuning
    predict.py                      # CLI inference using DistilBERT
  README.md
  requirements.txt
````

---

## Data Pipeline

1. **Source**
   A job postings dataset (CSV) with at least:

   * job title
   * job description / full text

   Saved as:

   ```text
   data/raw/jobs_full.csv
   ```

2. **Preprocessing & Auto-Labeling** (`src/data_prep.py`)

   * Normalize text (strip, lowercase)

   * Combine title + description into a single `text` field

   * Apply simple keyword-based rules to assign one of:

     * `AI_ML`
     * `SWE`
     * `DEVOPS`
     * `OTHER`

   * Save the processed dataset as:

   ```text
   data/processed/jobs_full_processed.csv
   ```

3. **Class Balancing**

   The raw dataset is heavily skewed toward `OTHER`. For training, a balanced subset is sampled:

   ```text
   AI_ML   ~2000
   SWE     ~4000
   DEVOPS  ~5000
   OTHER   ~5000
   ```

   This makes the classifier actually learn the differences between the classes instead of predicting `OTHER` for everything.

Run data prep:

```bash
python -m src.data_prep
```

---

## Models

### 1. Baseline: TF-IDF + Logistic Regression

File: `src/train.py`

* Vectorizer: `TfidfVectorizer`
* Model: `LogisticRegression`
* Train/test split on the balanced dataset

**Baseline performance (approximate):**

* Accuracy: ~0.71
* Macro F1: ~0.72

Class-wise F1 (roughly):

* `AI_ML`: ~0.82
* `SWE`:   ~0.71
* `DEVOPS`: ~0.67
* `OTHER`: ~0.70

Run baseline training:

```bash
python -m src.train
```

---

### 2. Enhanced: DistilBERT

File: `src/train_distilbert.py`

* Base model: `distilbert-base-uncased`
* Fine-tuned for 4-way classification on the same balanced dataset
* Uses HuggingFace `Trainer`

**DistilBERT performance (from eval):**

```text
Eval results: {
  "eval_loss": 0.67,
  "eval_accuracy": 0.75,
  "eval_macro_f1": 0.76,
  ...
}
```

Classification report (summary):

* Accuracy: **0.75**
* Macro F1: **0.76**

Approximate per-class F1:

* `AI_ML`: 0.82
* `SWE`:   0.73
* `DEVOPS`: 0.73
* `OTHER`: 0.74

So the transformer clearly improves over the TF-IDF baseline, especially in macro F1.

Run DistilBERT training:

```bash
python -m src.train_distilbert
```

This saves:

* model weights
* tokenizer
* label mapping

under:

```text
models/distilbert-job-classifier/
```

---

## Local Inference (CLI)

File: `src.predict`

Usage:

```bash
python -m src.predict "We are looking for a senior machine learning engineer with experience in Python, PyTorch, and deploying models to production."
```

Example outputs:

```text
Prediction: AI_ML (confidence: 0.997)

python -m src.predict "We need a full stack engineer with React, .NET, and SQL Server experience."
Prediction: SWE (confidence: 0.982)

python -m src.predict "Role focused on CI/CD pipelines, Kubernetes, Docker, and cloud infrastructure."
Prediction: DEVOPS (confidence: 0.989)

python -m src.predict "Handle inbound customer calls and support tickets for billing and technical questions."
Prediction: OTHER (confidence: 0.940)
```

The script:

* loads the saved DistilBERT model + tokenizer
* runs a forward pass
* returns the predicted class + confidence

---

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include at least:

```text
pandas
numpy
scikit-learn
torch
transformers
datasets
accelerate
```

(Add FastAPI/uvicorn/etc. only if you later decide to expose an HTTP endpoint.)

---

## Limitations

* Labels are created via **simple keyword rules**, not human annotation.
* Class boundaries overlap in reality (e.g., infra-heavy SWE vs DEVOPS).
* No hyperparameter tuning; configuration is intentionally minimal.
* Everything runs on CPU; training is slower but still manageable.

The point of the project is to demonstrate:

* practical handling of imbalanced, noisy real-world text data
* end-to-end model workflow (data → baseline → transformer → inference)
* awareness of trade-offs and limitations
