# ETA Challenge Submission

This repo predicts NYC taxi trip duration from request-time features: `pickup_zone`, `dropoff_zone`, `requested_at`, and `passenger_count`.

## Final score

Dev MAE: **294.0 s**

## Approach

The final model is an XGBoost regressor trained on the full 2023 training set with a small but high-signal engineered feature set. The core baseline features are `pickup_zone`, `dropoff_zone`, `hour`, `dow`, `month`, and `passenger_count`. I added two pair-level features computed from the full training data: historical mean duration for each `(pickup_zone, dropoff_zone)` pair and the pair's observation count. At inference time `predict.py` loads both `model.pkl` and `zone_pair_stats.pkl`, then reconstructs the same features for the request.

This ended up being the strongest simple setup: it kept inference fast, avoided extra dependencies, and captured a large amount of the route-specific variation that the baseline model missed.

## What I tried that didn't work

I first relied on the starter baseline with only coarse temporal and zone features. That was a decent starting point but left a lot of route-specific signal on the table. I also tried the same engineered feature set on a 1M-row sample to validate the idea quickly; that helped, but the full-train version was clearly better. I did not pursue heavier feature engineering like shapefile-based routing or weather joins because they would have added complexity and runtime cost without guaranteeing a better return than the simple pair-stat features.

## Where AI tooling sped me up most

The fastest win was using AI-assisted code generation for the data exploration and training script scaffolding. It helped me move quickly from the baseline to a working engineered experiment, and it was also useful for tightening the submission writeup. It was less useful for judging model quality; the actual gains came from looking at the data and validating the simplest high-signal features.

## Next experiments

If I had more time, I would try one of two follow-ups: zone centroid plus haversine distance from the public taxi zone lookup, or recent-history aggregates such as 7-day and 30-day pair means. Both are likely to add signal without making inference much slower.

## How to reproduce

```bash
python data/download_data.py
python analysis/train_full_engineered.py
python grade.py
```

That will rebuild `train.parquet` / `dev.parquet`, retrain the full engineered model, save `model.pkl` and `zone_pair_stats.pkl`, and report the local dev MAE.

## Files of interest

- `predict.py` — submission entrypoint used by the grader
- `analysis/train_full_engineered.py` — trains the final model
- `analysis/train_sample_engineered.py` — quick prototype on the 1M sample
- `Dockerfile` — packages the submission for scoring