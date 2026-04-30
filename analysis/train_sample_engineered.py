"""Train a quick model on data/sample_1M.parquet with a simple engineered feature set.
Saves model_sample_engineered.pkl and zone_pair_stats.pkl in the repo root.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "data" / "sample_1M.parquet"
MODEL_OUT = ROOT / "model_sample_engineered.pkl"
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
        # caller will compute
        out["zone_pair_mean"] = np.nan
        out["zone_pair_count"] = 0
        return out

    # map pair stats
    keys = list(pair_stats.keys())
    # pair_stats: {(pz,dz): (mean, count)}
    means = []
    counts = []
    gm = pair_stats.get("__global_mean", 0.0)
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
    g = df.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].agg(["count", "mean"]).reset_index()
    stats = { (int(r["pickup_zone"]), int(r["dropoff_zone"])): (float(r["mean"]), int(r["count"])) for _, r in g.iterrows() }
    stats["__global_mean"] = float(df["duration_seconds"].mean())
    return stats


def main():
    print("Loading sample...", SAMPLE)
    df = pd.read_parquet(SAMPLE)
    print("Rows:", len(df))

    # split
    train_df, hold_df = train_test_split(df, test_size=0.1, random_state=42)
    print("Train rows:", len(train_df), "hold rows:", len(hold_df))

    pair_stats = compute_pair_stats(train_df)
    print("Unique pairs:", len([k for k in pair_stats.keys() if k != "__global_mean"]))

    X_train = engineer(train_df, pair_stats)[FEATURES]
    y_train = train_df["duration_seconds"].to_numpy()
    X_hold = engineer(hold_df, pair_stats)[FEATURES]
    y_hold = hold_df["duration_seconds"].to_numpy()

    print("Training XGBoost on engineered features...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    preds = model.predict(X_hold)
    mae = float(np.mean(np.abs(preds - y_hold)))
    print(f"Holdout MAE: {mae:.1f} seconds")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(STATS_OUT, "wb") as f:
        pickle.dump(pair_stats, f)
    print("Saved model to", MODEL_OUT)
    print("Saved pair stats to", STATS_OUT)


if __name__ == "__main__":
    main()
