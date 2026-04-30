"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"
_PAIR_STATS_PATH = Path(__file__).parent / "zone_pair_stats.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)
# Disable xgboost's feature-name validation so we can predict on a bare
# numpy array (skips per-call DataFrame construction overhead).
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

# Load pair stats if available (optional at dev time)
try:
    with open(_PAIR_STATS_PATH, "rb") as _f:
        _PAIR_STATS = pickle.load(_f)
except Exception:
    _PAIR_STATS = None

# Feature order must match baseline.py:
#   pickup_zone, dropoff_zone, hour, dow, month, passenger_count


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    ts = datetime.fromisoformat(request["requested_at"])
    # Basic features
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    hour = ts.hour
    dow = ts.weekday()
    month = ts.month
    pc = int(request["passenger_count"])

    # Engineered pair features (fallback to global mean / zero count)
    if _PAIR_STATS is None:
        pair_mean = 0.0
        pair_count = 0
    else:
        gm = _PAIR_STATS.get("__global_mean", 0.0)
        mm = _PAIR_STATS.get((pz, dz))
        if mm is None:
            pair_mean = gm
            pair_count = 0
        else:
            pair_mean, pair_count = mm[0], mm[1]

    x = np.array(
        [[
            pz,
            dz,
            hour,
            dow,
            month,
            pc,
            float(pair_mean),
            int(pair_count),
        ]],
        dtype=np.float32,
    )
    return float(_MODEL.predict(x)[0])
