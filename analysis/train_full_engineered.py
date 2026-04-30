"""Train engineered model on full `data/train.parquet` and evaluate on `data/dev.parquet`.
Saves `model.pkl` (overwriting baseline) and `zone_pair_stats.pkl` for inference.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TRAIN = DATA / "train.parquet"
DEV = DATA / "dev.parquet"
MODEL_OUT = ROOT / "model.pkl"
STATS_OUT = ROOT / "zone_pair_stats.pkl"

FEATURES = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
    "month",
    "passenger_count",
    "zone_pair_mean",
    "zone_pair_count",
]


def engineer(df: pd.DataFrame, pair_stats: dict | None = None):
    ts = pd.to_datetime(df["requested_at"])
    out = pd.DataFrame({
        "pickup_zone": df["pickup_zone"].astype("int32"),
        "dropoff_zone": df["dropoff_zone"].astype("int32"),
        "hour": ts.dt.hour.astype("int8"),
        "dow": ts.dt.dayofweek.astype("int8"),
        "month": ts.dt.month.astype("int8"),
        "passenger_count": df["passenger_count"].astype("int8"),
    })

    if pair_stats is None:
        out["zone_pair_mean"] = 0.0
        out["zone_pair_count"] = 0
        return out

    gm = pair_stats.get("__global_mean", 0.0)
    means = []
    counts = []
    for pz, dz in zip(out["pickup_zone"].values, out["dropoff_zone"].values):
        mm = pair_stats.get((int(pz), int(dz)))
        if mm is None:
            means.append(gm)
            counts.append(0)
        else:
            means.append(mm[0])
            counts.append(mm[1])
    out["zone_pair_mean"] = np.array(means, dtype=np.float32)
    out["zone_pair_count"] = np.array(counts, dtype=np.int32)
    return out


def compute_pair_stats(df: pd.DataFrame):
    print("Computing pair stats (this may take a few minutes)...")
    g = df.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].agg(["count", "mean"]).reset_index()
    stats = { (int(r["pickup_zone"]), int(r["dropoff_zone"])): (float(r["mean"]), int(r["count"])) for _, r in g.iterrows() }
    stats["__global_mean"] = float(df["duration_seconds"].mean())
    return stats


def main():
    if not TRAIN.exists() or not DEV.exists():
        raise SystemExit("Missing train.parquet or dev.parquet in data/. Run data/download_data.py first.")

    print("Loading train and dev...")
    train = pd.read_parquet(TRAIN)
    dev = pd.read_parquet(DEV)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    pair_stats = compute_pair_stats(train)
    print(f"Unique pairs: {len([k for k in pair_stats.keys() if k != '__global_mean'])}")

    X_train = engineer(train, pair_stats)[FEATURES]
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer(dev, pair_stats)[FEATURES]
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost on full train with engineered features...")
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time()-t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(STATS_OUT, "wb") as f:
        pickle.dump(pair_stats, f)
    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved pair stats to {STATS_OUT}")


if __name__ == "__main__":
    main()
