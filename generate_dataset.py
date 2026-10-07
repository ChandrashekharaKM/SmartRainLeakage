"""
generate_dataset.py
-------------------
Creates a realistic *hourly* household water-consumption CSV with injected leak events,
using exactly the columns proposed in the synopsis:

    Date, Time, FlowRate, Consumption, Duration, Pressure, Leakage

Use this when you do not have a real dataset yet (or as a labelled benchmark: because we
inject the leaks ourselves we KNOW the ground truth and can compute precision / recall).

    python generate_dataset.py --days 120 --out data/water_consumption.csv
"""
import os
import argparse
import numpy as np
import pandas as pd

# Typical litres per hour for each hour of the day (4-person home, scaled below)
HOURLY_PROFILE = np.array([1, .6, .4, .4, .8, 3, 9, 14, 10, 6, 4, 4,
                           5, 4, 3, 3, 4, 7, 11, 13, 10, 7, 4, 2], float)


def generate(days=120, seed=42, n_leaks=10, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour, dow = idx.hour.values, idx.dayofweek.values

    weekend = np.where(dow >= 5, 1.15, 1.0)                      # more usage on weekends
    season = 1 + 0.08 * np.sin(np.linspace(0, 2 * np.pi, len(idx)))
    base = HOURLY_PROFILE[hour] * 4 * weekend * season
    consumption = base * rng.lognormal(0, 0.2, len(idx))          # natural randomness (L/h)

    # ---- inject leak events (never in the first 14 days so models get clean history)
    leak = np.zeros(len(idx), int)
    extra = np.zeros(len(idx))
    placed, tries = 0, 0
    while placed < n_leaks and tries < 500:
        tries += 1
        length = int(rng.integers(8, 49))                         # 8-48 hours
        s = int(rng.integers(14 * 24, len(idx) - length - 24))
        if leak[max(0, s - 48): s + length + 48].any():
            continue
        leak[s:s + length] = 1
        extra[s:s + length] = rng.uniform(12, 30)                 # constant extra L/h
        placed += 1
    consumption = consumption + extra

    # ---- derive the other sensor-style columns
    duration = np.clip(consumption / rng.uniform(5, 8, len(idx)), 0, 60)   # minutes of flow/hour
    duration = np.where(leak == 1, rng.uniform(55, 60, len(idx)), duration)  # leaks flow nonstop
    flow = np.where(duration > 0, consumption / np.maximum(duration, 1), 0)   # L/min while flowing
    pressure = 3.5 - 0.004 * consumption - 0.15 * leak + rng.normal(0, 0.05, len(idx))

    df = pd.DataFrame({
        "Date": idx.strftime("%Y-%m-%d"), "Time": idx.strftime("%H:%M"),
        "FlowRate": flow.round(2), "Consumption": consumption.round(1),
        "Duration": duration.round(1), "Pressure": pressure.round(2), "Leakage": leak,
    })

    # ---- deliberately add dirty data so the preprocessing module has work to do
    miss = rng.choice(len(df), size=int(0.005 * len(df)), replace=False)
    df.loc[miss, "Consumption"] = np.nan                           # missing values
    bad = rng.choice(len(df), size=5, replace=False)
    df.loc[bad, "Consumption"] = -999                              # data-entry errors
    dup = df.sample(15, random_state=seed)
    df = pd.concat([df, dup]).sort_index(kind="stable").reset_index(drop=True)  # duplicates
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--leaks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/water_consumption.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    d = generate(a.days, a.seed, a.leaks)
    d.to_csv(a.out, index=False)
    print(f"Saved {len(d)} rows -> {a.out}  (leak hours: {int(d.Leakage.sum())})")
