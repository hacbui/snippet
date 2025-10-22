"""
Auto Claims - Dynamic LLM Labeling Batch Generator (3-Level Taxonomy Version)
-----------------------------------------------------------------------------
Automatically generates balanced LLM labeling batches using your
taxonomy JSON (with 3 levels: Process → Reason → Root Cause/Empty).
"""

import pandas as pd
import json
from pathlib import Path

# ========== CONFIGURATION ==========
INPUT_FILE = "all_call_summaries.parquet"       # Raw call summaries file
TAXONOMY_FILE = "taxonomy/taxonomy_2025-10-04.json"  # Your taxonomy JSON
OUTPUT_DIR = Path("llm_labeling_batches_dynamic")

TOTAL_TARGET = 3300  # total records to label this round
PHASE_SPLIT = {"Phase_1": 1000, "Phase_2": 2000, "Phase_3": 300}

RANDOM_STATE = 42
# ====================================


def load_data():
    """Load summarized calls dataset."""
    df = pd.read_parquet(INPUT_FILE)
    if "process_area" not in df.columns:
        raise ValueError("Input file must include a 'process_area' column.")
    print(f"✅ Loaded {len(df):,} total call summaries.")
    return df


def load_taxonomy():
    """Load 3-level taxonomy JSON and count sublabels."""
    with open(TAXONOMY_FILE, "r") as f:
        taxonomy = json.load(f)

    process_reason_counts = {}
    for process_area, sub_dict in taxonomy.items():
        # Count how many Level-2 items exist under each Process Area
        n_reasons = len(sub_dict.keys())
        # Handle empty or placeholder processes
        if n_reasons == 0:
            n_reasons = 1
        process_reason_counts[process_area] = n_reasons

    print(f"📚 Loaded taxonomy with {len(process_reason_counts)} process areas:")
    for k, v in process_reason_counts.items():
        print(f" - {k}: {v} reason(s)")
    return process_reason_counts


def build_sampling_plan(process_reason_counts, total_target):
    """Compute samples per process area proportional to number of reasons."""
    total_reasons = sum(process_reason_counts.values())
    samples_per_reason = total_target / total_reasons

    plan = {
        area: int(samples_per_reason * count)
        for area, count in process_reason_counts.items()
    }

    print("\n🎯 Computed sampling plan:")
    for area, count in plan.items():
        print(f" - {area}: {count} samples (≈ {count / process_reason_counts[area]:.0f} per reason)")
    return plan


def stratified_sample(df, plan):
    """Sample calls proportionally per process area."""
    sampled = []
    for area, n_target in plan.items():
        subset = df[df["process_area"] == area]
        if subset.empty:
            print(f"⚠️ No records for {area}. Skipping.")
            continue
        n = min(n_target, len(subset))
        sample = subset.sample(n=n, random_state=RANDOM_STATE)
        sampled.append(sample)
        print(f"📦 Sampled {len(sample):,} from {area} (target={n_target}).")
    return pd.concat(sampled).reset_index(drop=True)


def split_into_phases(df):
    """Split the total batch into 3 LLM labeling phases."""
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    total = len(df)
    splits = []
    start = 0
    for phase, n in PHASE_SPLIT.items():
        end = min(start + n, total)
        splits.append((phase, df.iloc[start:end].copy()))
        start = end
    return splits


def save_batches(splits):
    """Save each phase batch as Parquet."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    for phase, data in splits:
        path = OUTPUT_DIR / f"{phase.lower()}_batch.parquet"
        data.to_parquet(path, index=False)
        print(f"💾 Saved {len(data):,} records → {path}")


def summarize(splits):
    """Print a summary table."""
    print("\n=== Labeling Batch Summary ===")
    for phase, df in splits:
        print(f"\n📍 {phase} - {len(df):,} total")
        print(df["process_area"].value_counts())


def main():
    df = load_data()
    process_reason_counts = load_taxonomy()
    plan = build_sampling_plan(process_reason_counts, TOTAL_TARGET)
    sampled_df = stratified_sample(df, plan)
    splits = split_into_phases(sampled_df)
    save_batches(splits)
    summarize(splits)
    print("\n✅ Dynamic taxonomy-aware labeling batches generated successfully.")


if __name__ == "__main__":
    main()
