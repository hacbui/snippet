"""
predict_reason.py
-----------------------------------------
Hierarchical prediction pipeline for Auto Claims call summaries.
Loads trained models (Level 1, Level 2, Level 3)
and predicts hierarchical taxonomy labels + confidence scores.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE_MODEL = "microsoft/deberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("models")
DATA_FILE = Path("data/new_calls.csv")   # or .parquet
OUTPUT_FILE = Path("data/predictions.parquet")

# ----------------------------
# LOAD TOKENIZER
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ----------------------------
# LOAD LABEL ENCODERS
# ----------------------------
def load_classes(level_name):
    classes = pd.read_csv(MODEL_DIR / "label_encoders" / f"{level_name}_classes.csv", header=None)[0].tolist()
    return classes

classes_l1 = load_classes("level_1")
classes_l2 = load_classes("level_2")
classes_l3 = load_classes("level_3")

# ----------------------------
# LOAD MODELS
# ----------------------------
def load_model(level_name):
    model_path = MODEL_DIR / level_name / "final"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model

model_l1 = load_model("level_1")
model_l2 = load_model("level_2")
model_l3 = load_model("level_3")

# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict_batch(model, texts, classes):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        labels = [classes[i] for i in pred_idx.cpu().numpy()]
    return labels, conf.cpu().numpy()

# ----------------------------
# LOAD DATA
# ----------------------------
if DATA_FILE.suffix == ".csv":
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.read_parquet(DATA_FILE)

if "summary" not in df.columns:
    raise ValueError("Input file must contain a 'summary' column.")

print(f"✅ Loaded {len(df):,} call summaries for prediction")

# ----------------------------
# HIERARCHICAL INFERENCE
# ----------------------------
batch_size = 16
results = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    summaries = batch["summary"].tolist()

    # Level 1
    lvl1_labels, lvl1_conf = predict_batch(model_l1, summaries, classes_l1)

    # Level 2
    lvl2_labels, lvl2_conf = predict_batch(model_l2, summaries, classes_l2)

    # Level 3
    lvl3_labels, lvl3_conf = predict_batch(model_l3, summaries, classes_l3)

    tmp = pd.DataFrame({
        "summary": summaries,
        "level_1": lvl1_labels,
        "level_1_conf": lvl1_conf,
        "level_2": lvl2_labels,
        "level_2_conf": lvl2_conf,
        "level_3": lvl3_labels,
        "level_3_conf": lvl3_conf
    })

    results.append(tmp)

pred_df = pd.concat(results).reset_index(drop=True)

# ----------------------------
# AGGREGATE CONFIDENCE (optional)
# ----------------------------
pred_df["overall_conf"] = (
    0.4 * pred_df["level_1_conf"] +
    0.35 * pred_df["level_2_conf"] +
    0.25 * pred_df["level_3_conf"]
)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
pred_df.to_parquet(OUTPUT_FILE, index=False)
print(f"✅ Saved predictions → {OUTPUT_FILE}")
print(pred_df.head(5))
