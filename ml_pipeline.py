"""
ml_pipeline.py  -  Modules 2-5 of the synopsis
==============================================
Data collection/standardising -> preprocessing -> feature extraction ->
regression model -> residual-based leakage detection (+ Isolation Forest comparison).

IDEA IN ONE PARAGRAPH
    A regression model learns what *normal* hourly consumption looks like for a house
    (time of day, weekday, and what it used at the same time on previous days / weeks).
    Every hour we compare ACTUAL with PREDICTED consumption. If actual is much higher than
    predicted ("z-score" above a threshold) for several hours in a row, we raise
    "Possible Leakage".  Isolation Forest is kept as the label-free alternative.
"""
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_recall_fscore_support, confusion_matrix)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:          # xgboost is optional
    HAS_XGB = False

Z_THRESHOLD = 3.0          # how many "sigmas" above prediction counts as abnormal
MIN_PERSIST = 3            # abnormal hours in a row before it counts as persistent
MERGE_GAP = 12             # alerts closer than this many hours are merged into one event
MAX_L_PER_H = 3000         # physically impossible above this -> data-entry error
MODEL_COLS = ["hour", "dow", "weekend", "hour_sin", "hour_cos", "hist_week", "hist_day"]


# ---------------------------------------------------------------- 1. COLLECT / STANDARDISE
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find(cols, exact=(), prefix=()):
    n = {c: _norm(c) for c in cols}
    for c, v in n.items():
        if v in exact:
            return c
    for c, v in n.items():
        if any(v.startswith(p) for p in prefix):
            return c
    return None


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Map ANY reasonable CSV (e.g. a Kaggle download) onto the project's standard columns:
    timestamp, consumption, flow_rate, duration, pressure, leak."""
    cols = list(raw.columns)
    out = pd.DataFrame(index=raw.index)

    ts = _find(cols, exact=("timestamp", "datetime"), prefix=("timestamp", "datetime"))
    if ts:
        out["timestamp"] = pd.to_datetime(raw[ts], errors="coerce")
    else:
        d, t = _find(cols, exact=("date",)), _find(cols, exact=("time",))
        if d is None:
            raise ValueError("CSV needs a Timestamp column or Date (+Time) columns")
        s = raw[d].astype(str) + ((" " + raw[t].astype(str)) if t else "")
        out["timestamp"] = pd.to_datetime(s, errors="coerce")

    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c else np.nan
    cons = _find(cols, exact=("consumption", "waterconsumption", "volume", "usage"))
    flow = _find(cols, prefix=("flowrate", "flow"))
    out["consumption"] = num(cons)
    out["flow_rate"] = num(flow)
    if cons is None:                      # only a flow rate is available -> assume L/min, hourly
        if flow is None:
            raise ValueError("CSV needs a Consumption or FlowRate column")
        out["consumption"] = out["flow_rate"] * 60
    out["duration"] = num(_find(cols, prefix=("duration",)))
    out["pressure"] = num(_find(cols, prefix=("pressure",)))

    lk = _find(cols, exact=("leakage", "leak", "leakstatus", "leaklabel", "label"), prefix=("leak",))
    if lk:
        v = pd.to_numeric(raw[lk], errors="coerce")
        if v.isna().all():
            v = raw[lk].astype(str).str.lower().isin(["yes", "true", "leak", "leakage", "1"]).astype(float)
        out["leak"] = v
    else:
        out["leak"] = np.nan
    return out


# ---------------------------------------------------------------- 2. PREPROCESS
def preprocess(df: pd.DataFrame):
    """Clean + convert to a regular HOURLY time series. Returns (clean_df, stats_dict)."""
    st = {"rows_in": int(len(df))}
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    st["duplicates_removed"] = int(df.duplicated("timestamp").sum())
    df = df.drop_duplicates("timestamp", keep="first")

    bad = (df["consumption"] < 0) | (df["consumption"] > MAX_L_PER_H)
    st["entry_errors_nulled"] = int(bad.sum())
    df.loc[bad, "consumption"] = np.nan

    r = df.set_index("timestamp").resample("h")
    out = pd.DataFrame({
        "consumption": r["consumption"].sum(min_count=1),
        "flow_rate": r["flow_rate"].mean(),
        "duration": r["duration"].sum(min_count=1),
        "pressure": r["pressure"].mean(),
        "leak": r["leak"].max(),
    })
    st["missing_values_filled"] = int(out["consumption"].isna().sum())
    for c in ["consumption", "flow_rate", "duration", "pressure"]:
        out[c] = out[c].interpolate(method="time", limit=6, limit_area="inside").ffill().bfill()
    out["consumption"] = out["consumption"].fillna(0)
    out["flow_rate"] = out["flow_rate"].fillna(out["consumption"] / 60)
    out["duration"] = out["duration"].fillna((out["consumption"] / 6).clip(0, 60))
    if out["leak"].notna().any():
        out["leak"] = out["leak"].fillna(0).astype(int)
    st.update(rows_out=int(len(out)), hours=int(len(out)),
              start=str(out.index.min()), end=str(out.index.max()),
              labelled=bool(out["leak"].notna().any()),
              has_pressure=bool(out["pressure"].notna().any()))
    return out, st


# ---------------------------------------------------------------- 3. FEATURES
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features + 'previous consumption' features.
    hist_day  = MEDIAN of the same hour over the previous 7 days
    hist_week = MEDIAN of the same hour-of-week over the previous 4 weeks
    Medians (not means / last value) so that a short leak does not 'teach' the model it is normal."""
    f = df.copy()
    i = f.index
    f["hour"], f["dow"] = i.hour, i.dayofweek
    f["weekend"] = (f["dow"] >= 5).astype(int)
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * f["hour"] / 24), np.cos(2 * np.pi * f["hour"] / 24)
    c = f["consumption"]
    wk = np.vstack([c.shift(168 * k).values for k in range(1, 5)])
    dy = np.vstack([c.shift(24 * k).values for k in range(1, 8)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["hist_week"], f["hist_day"] = np.nanmedian(wk, axis=0), np.nanmedian(dy, axis=0)
    f["dev_hist"] = c - f["hist_day"]
    return f


# ---------------------------------------------------------------- helpers: persistence
def run_mask(flags, min_len):
    """True for every position that belongs to a run of >= min_len consecutive True values."""
    flags = np.asarray(flags, bool)
    out = np.zeros(len(flags), bool)
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def events_from_mask(mask, max_gap=MERGE_GAP):
    """Turn a boolean mask into [(start_idx, end_idx)] events, merging gaps <= max_gap."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    ev = []
    for k in idx:
        if ev and k - ev[-1][1] <= max_gap + 1:
            ev[-1][1] = int(k)
        else:
            ev.append([int(k), int(k)])
    return [tuple(e) for e in ev]


# ---------------------------------------------------------------- 4. SCORING
def score(f: pd.DataFrame, b: dict) -> pd.DataFrame:
    """Apply a trained bundle to a feature frame -> predictions, z-scores, flags."""
    ok = f.dropna(subset=MODEL_COLS + ["consumption"]).copy()
    if ok.empty:
        return ok
    ok["predicted"] = np.clip(b["model"].predict(ok[MODEL_COLS]), 0, None)
    ok["residual"] = ok["consumption"] - ok["predicted"]
    ok["sigma"] = np.maximum(b["c0"] + b["c1"] * ok["predicted"], 0.5) * 1.2533
    ok["z"] = ok["residual"] / ok["sigma"]
    ok["anomaly"] = (ok["z"] > b["z_threshold"]).astype(int)
    ok["persistent"] = run_mask(ok["anomaly"].values, b["n_persist"]).astype(int)
    X = ok.reindex(columns=b["if_cols"]).fillna(0)
    ok["if_flag"] = (b["iforest"].predict(X) == -1).astype(int)
    ok["if_score"] = -b["iforest"].score_samples(X)
    return ok


# ---------------------------------------------------------------- 5. TRAIN + EVALUATE
def _reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"MAE": float(mean_absolute_error(y, p)), "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y, p))}


def _cls_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _event_eval(true_mask, pred_mask):
    """Event-level: did we catch each real leak? how many false alarms?"""
    te, pe = events_from_mask(true_mask, 0), events_from_mask(pred_mask)
    caught = sum(any(not (p[1] < t[0] or p[0] > t[1]) for p in pe) for t in te)
    false = sum(not any(not (p[1] < t[0] or p[0] > t[1]) for t in te) for p in pe)
    return {"true_events": len(te), "detected": int(caught), "alert_events": len(pe),
            "false_alerts": int(false),
            "event_recall": float(caught / len(te)) if te else None}


def train_and_evaluate(df: pd.DataFrame, z=Z_THRESHOLD, n_persist=MIN_PERSIST):
    f = build_features(df).dropna(subset=MODEL_COLS + ["consumption"])
    n = len(f)
    tr, ca, te = f.iloc[:int(n * .5)], f.iloc[int(n * .5):int(n * .7)], f.iloc[int(n * .7):]
    labelled = bool(f["leak"].notna().any())
    normal = (lambda d: d[d["leak"] == 0]) if labelled else (lambda d: d)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=.05,
                                         subsample=.9, random_state=42, verbosity=0)

    table, fitted = {}, {}
    for name, m in models.items():                                   # train on NORMAL behaviour only
        m.fit(normal(tr)[MODEL_COLS], normal(tr)["consumption"])
        fitted[name] = m
        ca_n, te_n = normal(ca), normal(te)
        table[name] = {"calibration_rmse": _reg_metrics(ca_n["consumption"], m.predict(ca_n[MODEL_COLS]))["RMSE"],
                       **_reg_metrics(te_n["consumption"], m.predict(te_n[MODEL_COLS]))}
    best = min(table, key=lambda k: table[k]["calibration_rmse"])
    model = fitted[best]

    # noise model: |residual| grows with the predicted level (peaks are noisier than nights)
    ca_n = normal(ca)
    pred = model.predict(ca_n[MODEL_COLS])
    c1, c0 = np.polyfit(pred, np.abs(ca_n["consumption"].values - pred), 1)

    # Isolation Forest: unsupervised, never sees the labels
    if_cols = ["dev_hist", "flow_rate", "duration", "hour_sin", "hour_cos"]
    if df["pressure"].notna().any():
        if_cols.append("pressure")
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, contamination=0.04, random_state=42))
    iso.fit(tr[if_cols].fillna(0))

    bundle = dict(model=model, model_name=best, c0=float(c0), c1=float(c1), z_threshold=z,
                  n_persist=n_persist, iforest=iso, if_cols=if_cols, model_cols=MODEL_COLS)

    metrics = {"models": table, "best_model": best, "z_threshold": z, "min_persist_hours": n_persist,
               "split": {"train": len(tr), "calibration": len(ca), "test": len(te)},
               "labelled": labelled, "isolation_forest_features": if_cols}

    sc = score(te, bundle)
    if labelled:
        y = sc["leak"].astype(int).values
        persistent_events = events_from_mask(sc["persistent"].values)
        ev_mask = np.zeros(len(sc), bool)
        for a, b_ in persistent_events:
            ev_mask[a:b_ + 1] = True
        metrics["detection"] = {
            "regression_threshold_only": _cls_metrics(y, sc["anomaly"]),
            "regression_persistent": _cls_metrics(y, sc["persistent"]),
            "isolation_forest": _cls_metrics(y, sc["if_flag"]),
            "event_level_regression": _event_eval(y.astype(bool), ev_mask),
            "event_level_isolation_forest": _event_eval(y.astype(bool), run_mask(sc["if_flag"].values, n_persist)),
        }
        # sensitivity: how does the threshold z change precision/recall?
        sens = []
        for zz in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            a = (sc["z"] > zz).astype(int)
            p = run_mask(a.values, n_persist).astype(int)
            m = _cls_metrics(y, p)
            sens.append({"z": zz, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
        metrics["threshold_sensitivity"] = sens
    return bundle, metrics, sc
