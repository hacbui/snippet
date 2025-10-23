"""
predict_reason_process_mode.py
---------------------------------------
For Process Engineering review:
- Loads Phase 2+ classifier models
- Runs hierarchical inference
- Aggregates results by taxonomy level
- Generates summary tables & confidence insights
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === Configuration ===
BASE_MODEL = "microsoft/deberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("models")
DATA_FILE = Path("data/new_calls.parquet")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# === Load Label Classes ===
def load_classes(level_name):
    file_path = MODEL_DIR / "label_encoders" / f"{level_name}_classes.csv"
    return pd.read_csv(file_path, header=None)[0].tolist()

classes_l1 = load_classes("level_1")
classes_l2 = load_classes("level_2")
classes_l3 = load_classes("level_3")

# === Load Models ===
def load_model(level_name):
    model_path = MODEL_DIR / level_name / "final"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model

model_l1 = load_model("level_1")
model_l2 = load_model("level_2")
model_l3 = load_model("level_3")

# === Predict Function ===
def predict_batch(model, texts, classes):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        labels = [classes[i] for i in pred_idx.cpu().numpy()]
    return labels, conf.cpu().numpy()

# === Load Data ===
df = pd.read_parquet(DATA_FILE)
print(f"✅ Loaded {len(df):,} call summaries")

# === Run Hierarchical Prediction ===
batch_size = 16
all_preds = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    texts = batch["summary"].tolist()

    lvl1, conf1 = predict_batch(model_l1, texts, classes_l1)
    lvl2, conf2 = predict_batch(model_l2, texts, classes_l2)
    lvl3, conf3 = predict_batch(model_l3, texts, classes_l3)

    temp = pd.DataFrame({
        "summary": texts,
        "level_1": lvl1,
        "level_1_conf": conf1,
        "level_2": lvl2,
        "level_2_conf": conf2,
        "level_3": lvl3,
        "level_3_conf": conf3
    })
    all_preds.append(temp)

pred_df = pd.concat(all_preds).reset_index(drop=True)
pred_df["overall_conf"] = (
    0.4 * pred_df["level_1_conf"] +
    0.35 * pred_df["level_2_conf"] +
    0.25 * pred_df["level_3_conf"]
)

pred_df.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)
print(f"✅ Saved raw predictions → {OUTPUT_DIR}/predictions.parquet")

# === Aggregated Insights ===
agg = (
    pred_df.groupby(["level_1", "level_2"])
    .agg(
        calls=("summary", "count"),
        avg_conf=("overall_conf", "mean")
    )
    .reset_index()
    .sort_values("calls", ascending=False)
)
agg["avg_conf"] = agg["avg_conf"].round(3)

# === Emerging Categories ===
# Identify small but growing (low-confidence) categories
low_conf = agg[agg["avg_conf"] < 0.75]
low_conf["priority_flag"] = "Review"
top_conf = agg[agg["avg_conf"] >= 0.75]
top_conf["priority_flag"] = "Stable"

summary = pd.concat([top_conf, low_conf])
summary.to_csv(OUTPUT_DIR / "process_insight_report.csv", index=False)
print(f"📊 Saved summary report → {OUTPUT_DIR}/process_insight_report.csv")

# === Optional: Visualization ===
plt.figure(figsize=(10, 6))
top10 = agg.head(10)
plt.barh(top10["level_2"], top10["calls"])
plt.title("Top 10 Reasons for Call (Level 2)")
plt.xlabel("Number of Calls")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_reasons.png")
plt.close()

plt.figure(figsize=(6, 4))
pred_df["overall_conf"].hist(bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Overall Confidence")
plt.ylabel("Call Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_histogram.png")
plt.close()

print("📈 Charts saved: top_reasons.png, confidence_histogram.png")

# === Print Key Findings ===
print("\n=== PROCESS INSIGHT SUMMARY ===")
print(summary.head(15))
print("\nLow-confidence areas to review:")
print(low_conf[["level_1", "level_2", "calls", "avg_conf"]].head(10))
