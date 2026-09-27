from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score


# --------------------------------------------------
# Paths
# --------------------------------------------------

CURRENT = Path(__file__).resolve()

PROJECT_ROOT = next(
    p for p in CURRENT.parents
    if (p / "student_resource").exists()
)

OUTPUT = PROJECT_ROOT / "AlgoAstra_submission" / "output"

FEATURE_FILE = OUTPUT / "training_features_sample.tsv"
MODEL_FILE = OUTPUT / "matching_model.pkl"


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("1. Loading training features...", flush=True)

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

print(f"   Rows: {len(df):,}", flush=True)


# --------------------------------------------------
# Features
# --------------------------------------------------

ID_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]

TARGET = "label"

FEATURE_COLUMNS = [
    c for c in df.columns
    if c not in ID_COLUMNS + [TARGET]
]

print(f"   Features: {len(FEATURE_COLUMNS)}", flush=True)


X = df[FEATURE_COLUMNS].astype(np.float32)
y = df[TARGET].astype(np.int8)

print("2. Data prepared.", flush=True)


# --------------------------------------------------
# Fast Source1-level split
# --------------------------------------------------

print("3. Creating train/validation split...", flush=True)

rng = np.random.default_rng(42)

unique_ids = df["source1_entity_id"].unique()

rng.shuffle(unique_ids)

split_point = int(len(unique_ids) * 0.80)

train_ids = set(unique_ids[:split_point])

train_mask = df["source1_entity_id"].isin(train_ids).to_numpy()

X_train = X.iloc[train_mask]
y_train = y.iloc[train_mask]

X_valid = X.iloc[~train_mask]
y_valid = y.iloc[~train_mask]

print(f"   Train rows: {len(X_train):,}", flush=True)
print(f"   Valid rows: {len(X_valid):,}", flush=True)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("4. Training HistGradientBoosting model...", flush=True)

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.10,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

print("5. Training complete.", flush=True)


# --------------------------------------------------
# Validation
# --------------------------------------------------

print("6. Calculating validation probabilities...", flush=True)

probabilities = model.predict_proba(X_valid)[:, 1]

print("7. Searching thresholds...", flush=True)

best_threshold = 0.5
best_f05 = -1

for threshold in np.arange(0.10, 0.96, 0.05):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_valid,
        predictions,
        beta=0.5,
        zero_division=0
    )

    print(
        f"threshold={threshold:.2f} "
        f"precision={precision:.4f} "
        f"recall={recall:.4f} "
        f"F0.5={f05:.4f}",
        flush=True
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold


# --------------------------------------------------
# Save
# --------------------------------------------------

print("\n8. Best validation result", flush=True)
print(f"   Threshold: {best_threshold:.2f}", flush=True)
print(f"   F0.5:      {best_f05:.4f}", flush=True)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "threshold": best_threshold,
        },
        f
    )

print("\nModel saved:", flush=True)
print(MODEL_FILE, flush=True)