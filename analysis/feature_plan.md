Feature exploration summary and plan

Findings from `data/sample_1M.parquet`:
- 1,000,000 rows; no nulls in core columns.
- Duration mean ~988s, median ~766s; long tail up to ~10k seconds.
- Passenger counts include a few zeros and up to 9.
- Strong time-of-day and day-of-week signals (mean durations vary by hour and dow).
- Top zone pairs show large variation; some intra-zone trips are short while cross-zone trips longer.

Proposed quick features to add (fast, cheap):
- `hour`, `dow`, `month` (from `requested_at`).
- `zone_pair_mean`: historical mean duration for (pickup_zone, dropoff_zone), computed from training data (fallback to global mean).
- `zone_pair_count`: count of observations for that pair (helps with low-support pairs).
- `is_peak`: boolean for rush hours (7-9, 16-18).

Planned experiment (fast, reproducible):
- Train on `data/sample_1M.parquet` (90% train / 10% holdout) using features above.
- Use XGBoost with modest size (200 trees) for quick feedback.
- Save `model_sample_engineered.pkl` and `zone_pair_stats.pkl`.

Commands to run locally:

```bash
python analysis/train_sample_engineered.py
```

Next: I'll create and run `analysis/train_sample_engineered.py` to get a quick MAE baseline with these features.