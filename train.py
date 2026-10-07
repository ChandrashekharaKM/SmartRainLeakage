"""
train.py
--------
Complete training and initialization pipeline:
1. Generates or loads the dataset
2. Cleans & standardises through ml_pipeline
3. Trains Regression Models (Linear, Decision Tree, Random Forest, XGBoost) & Isolation Forest
4. Saves trained bundle to models/model.joblib
5. Saves evaluation metrics to models/metrics.json
6. Seeds SQLite database with cleaned readings and initial detections
"""
import os
import json
import joblib
import pandas as pd
import ml_pipeline as ml
import generate_dataset as gen
from db import init_db, conn
import service

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "water_consumption.csv")
MODELS_DIR = os.path.join(BASE, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")


def train(csv_path=DATA_PATH, days=120, leaks=10, z=ml.Z_THRESHOLD, n_persist=ml.MIN_PERSIST, seed=42):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # 1. Dataset Generation if missing
    if not os.path.exists(csv_path):
        print(f"[1/5] Dataset not found at {csv_path}. Generating synthetic {days}-day dataset...")
        df_raw = gen.generate(days=days, seed=seed, n_leaks=leaks)
        df_raw.to_csv(csv_path, index=False)
        print(f"      Saved {len(df_raw)} records to {csv_path}")
    else:
        print(f"[1/5] Loading existing dataset from {csv_path}...")
        df_raw = pd.read_csv(csv_path)

    # 2. Standardise & Preprocess
    print("[2/5] Standardising columns and preprocessing time series...")
    std_df = ml.standardise(df_raw)
    clean_df, prep_stats = ml.preprocess(std_df)
    print(f"      Raw rows: {prep_stats['rows_in']} -> Cleaned hourly intervals: {prep_stats['rows_out']}")
    print(f"      Duplicates removed: {prep_stats['duplicates_removed']}, "
          f"Errors nulled: {prep_stats['entry_errors_nulled']}, "
          f"Missing interpolated: {prep_stats['missing_values_filled']}")

    # 3. Model Training & Evaluation
    print(f"[3/5] Training models (Linear Regression, Decision Tree, Random Forest, XGBoost, Isolation Forest)...")
    bundle, metrics, score_df = ml.train_and_evaluate(clean_df, z=z, n_persist=n_persist)
    metrics["preprocessing_stats"] = prep_stats

    print(f"      Best Model Selected: {metrics['best_model']}")
    for m_name, m_res in metrics["models"].items():
        print(f"      - {m_name:18}: Calibration RMSE={m_res['calibration_rmse']:.2f}, "
              f"Test RMSE={m_res['RMSE']:.2f}, R2={m_res['R2']:.3f}, MAE={m_res['MAE']:.2f}")

    if "detection" in metrics:
        det = metrics["detection"]["event_level_regression"]
        print(f"      Leak Event Detection: {det['detected']}/{det['true_events']} events caught "
              f"({det['event_recall'] * 100:.1f}%), False Alerts={det['false_alerts']}")

    # 4. Save Model Bundle & Metrics
    print(f"[4/5] Saving model bundle to {MODEL_PATH} and metrics to {METRICS_PATH}...")
    joblib.dump(bundle, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 5. Seed Database & Compute Initial Rescoring
    print("[5/5] Initializing database and ingesting readings into water.db...")
    init_db(reset=False)
    ingested = service.ingest_df(hid=1, df=clean_df)
    print(f"      Ingested {ingested} hourly readings for Household 1.")
    scored_cnt = service.rescore(hid=1)
    print(f"      Rescored {scored_cnt} records; persistent alerts generated in database.")

    print("\nTraining and database setup complete successfully!")
    return metrics


if __name__ == "__main__":
    train()
