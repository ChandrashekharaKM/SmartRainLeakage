"""
service.py  -  glue between the ML pipeline, the database and the alert module
=============================================================================
ingest_df()   : store cleaned readings            (Data Collection + Database modules)
rescore()     : run the model, store detections   (ML + Leakage Detection modules)
              : create / update alerts            (Alert module)
simulate()    : fake "live sensor" readings so you can demo without hardware
"""
import os
import datetime as dt
import joblib
import numpy as np
import pandas as pd
import ml_pipeline as ml
from db import conn, q, BASE

MODEL_PATH = os.path.join(BASE, "models", "model.joblib")
ALERT_LOG = os.path.join(BASE, "logs", "alerts.log")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def reload_bundle():
    global _bundle
    _bundle = None
    return get_bundle()


def ingest_df(hid, df):
    """df = cleaned hourly frame (index = timestamp)."""
    rows = [(hid, ts.strftime("%Y-%m-%d %H:%M"), float(r.consumption), float(r.flow_rate),
             float(r.duration), None if pd.isna(r.pressure) else float(r.pressure),
             None if pd.isna(r.leak) else int(r.leak)) for ts, r in df.iterrows()]
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO readings VALUES(?,?,?,?,?,?,?)", rows)
    return len(rows)


def severity(hours, excess):
    if hours >= 24 or excess >= 400:
        return "High"
    return "Medium" if hours >= 8 or excess >= 120 else "Low"


def notify(hid, start, end, hours, excess, sev):
    """Alert module. Replace the body with e-mail / SMS / Telegram / push notification."""
    msg = (f"{dt.datetime.now():%Y-%m-%d %H:%M} | household {hid} | POSSIBLE LEAKAGE ({sev}) "
           f"{start} -> {end} ({hours} h, ~{excess:.0f} L above normal)")
    os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
    with open(ALERT_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    print("[ALERT]", msg)


def rescore(hid, tail_hours=None, send_notifications=True):
    """Score stored readings. tail_hours=None -> everything, else only the newest N hours
    (older rows are still read as history for the 'previous consumption' features)."""
    b = get_bundle()
    limit = "" if tail_hours is None else f"LIMIT {int(tail_hours) + 900}"
    rows = q(f"SELECT * FROM (SELECT * FROM readings WHERE household_id=? ORDER BY ts DESC {limit}) ORDER BY ts",
             (hid,))
    if not rows:
        return 0
    df = pd.DataFrame([dict(r) for r in rows])
    df.index = pd.to_datetime(df.pop("ts"))
    df = df.drop(columns=["household_id"]).rename(columns={"leak_label": "leak"})
    df = df.asfreq("h").interpolate(limit=6).ffill().bfill()
    sc = ml.score(ml.build_features(df), b)
    if sc.empty:
        return 0
    if tail_hours is not None:
        sc = sc[sc.index >= sc.index.max() - pd.Timedelta(hours=tail_hours)]
    t = lambda x: x.strftime("%Y-%m-%d %H:%M")

    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO detections VALUES(?,?,?,?,?,?,?,?)",
                      [(hid, t(i), round(float(r.predicted), 2), round(float(r.residual), 2),
                        round(float(r.z), 2), int(r.anomaly), int(r.persistent), int(r.if_flag))
                       for i, r in sc.iterrows()])
        # ---- alerts: merge persistent runs into events
        events = ml.events_from_mask(sc["persistent"].values)
        keep = []
        for a, e in events:
            seg = sc.iloc[a:e + 1]
            start, end = t(seg.index[0]), t(seg.index[-1])
            hours = len(seg)
            excess = float(seg.loc[seg["anomaly"] == 1, "residual"].clip(lower=0).sum())
            sev = severity(hours, excess)
            keep.append(start)
            old = c.execute("SELECT id FROM alerts WHERE household_id=? AND start_ts=?", (hid, start)).fetchone()
            if old:
                c.execute("UPDATE alerts SET end_ts=?,hours=?,excess_litres=?,severity=? WHERE id=?",
                          (end, hours, round(excess, 1), sev, old["id"]))
            else:
                c.execute("INSERT INTO alerts(household_id,start_ts,end_ts,hours,excess_litres,severity) "
                          "VALUES(?,?,?,?,?,?)", (hid, start, end, hours, round(excess, 1), sev))
                if send_notifications:
                    notify(hid, start, end, hours, excess, sev)
        lo, hi = t(sc.index.min()), t(sc.index.max())
        marks = ",".join("?" * len(keep)) or "''"
        c.execute(f"DELETE FROM alerts WHERE household_id=? AND status='open' AND start_ts BETWEEN ? AND ? "
                  f"AND start_ts NOT IN ({marks})", (hid, lo, hi, *keep))
    return len(sc)


def simulate(hid, hours=1, leak=False):
    """Append `hours` fake sensor readings after the newest stored one (normal or leaking)."""
    last = q("SELECT MAX(ts) m FROM readings WHERE household_id=?", (hid,), one=True)
    last_ts = last["m"] if last else None
    if not last_ts:
        raise ValueError("No historical readings found for this household. Please ingest data first.")
    ts = pd.Timestamp(last_ts)
    rng = np.random.default_rng()
    rows = []
    for _ in range(hours):
        ts += pd.Timedelta(hours=1)
        hist = q("SELECT consumption FROM readings WHERE household_id=? AND substr(ts,12,2)=? "
                 "ORDER BY ts DESC LIMIT 7", (hid, f"{ts.hour:02d}"))
        base = float(np.median([h["consumption"] for h in hist])) if hist else 20.0
        cons = base * rng.lognormal(0, .15) + (rng.uniform(15, 30) if leak else 0)
        dur = float(rng.uniform(55, 60)) if leak else float(min(60, cons / rng.uniform(5, 8)))
        rows.append((hid, ts.strftime("%Y-%m-%d %H:%M"), round(float(cons), 1),
                     round(float(cons / max(dur, 1)), 2),
                     round(float(dur), 1),
                     round(float(3.5 - .004 * cons - (.15 if leak else 0) + rng.normal(0, .05)), 2),
                     1 if leak else 0))
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO readings VALUES(?,?,?,?,?,?,?)", rows)
    rescore(hid, tail_hours=300)
    return {
        "added_hours": hours,
        "mode": "leak" if leak else "normal",
        "latest_ts": ts.strftime("%Y-%m-%d %H:%M"),
        "latest_consumption": round(float(cons), 1)
    }


def get_dashboard_summary(hid=1):
    """Aggregates metrics for the high-level dashboard cards and widgets."""
    latest_r = q("SELECT * FROM readings WHERE household_id=? ORDER BY ts DESC LIMIT 1", (hid,), one=True)
    latest_d = q("SELECT * FROM detections WHERE household_id=? ORDER BY ts DESC LIMIT 1", (hid,), one=True)
    open_alerts = q("SELECT COUNT(*) c FROM alerts WHERE household_id=? AND status='open'", (hid,), one=True)["c"]
    high_alerts = q("SELECT COUNT(*) c FROM alerts WHERE household_id=? AND status='open' AND severity='High'", (hid,), one=True)["c"]

    # 24 hour stats
    last_24 = q("SELECT SUM(consumption) s, AVG(flow_rate) f, AVG(pressure) p FROM ("
                "SELECT consumption, flow_rate, pressure FROM readings WHERE household_id=? ORDER BY ts DESC LIMIT 24)",
                (hid,), one=True)

    # 7-day sum
    last_7d = q("SELECT SUM(consumption) s FROM ("
                "SELECT consumption FROM readings WHERE household_id=? ORDER BY ts DESC LIMIT 168)",
                (hid,), one=True)

    # Total recorded hours
    total_hours = q("SELECT COUNT(*) c FROM readings WHERE household_id=?", (hid,), one=True)["c"]

    status = "Normal"
    if latest_d:
        if latest_d["persistent"] == 1:
            status = "Persistent Leak Detected"
        elif latest_d["anomaly"] == 1:
            status = "Anomaly / Warning"

    return {
        "status": status,
        "latest_reading": dict(latest_r) if latest_r else None,
        "latest_detection": dict(latest_d) if latest_d else None,
        "open_alerts": open_alerts,
        "high_alerts": high_alerts,
        "usage_24h": round(last_24["s"] or 0, 1) if last_24 else 0,
        "avg_flow_24h": round(last_24["f"] or 0, 2) if last_24 else 0,
        "avg_pressure_24h": round(last_24["p"] or 0, 2) if last_24 else 0,
        "usage_7d": round(last_7d["s"] or 0, 1) if last_7d else 0,
        "total_records": total_hours,
    }


def get_chart_data(hid=1, hours=72):
    """Returns aligned time-series of readings and detections for Chart.js visualization."""
    sql = """
    SELECT r.ts, r.consumption, r.flow_rate, r.pressure,
           d.predicted, d.residual, d.zscore, d.anomaly, d.persistent, d.if_flag
    FROM (
        SELECT * FROM readings WHERE household_id=? ORDER BY ts DESC LIMIT ?
    ) r
    LEFT JOIN detections d ON r.household_id=d.household_id AND r.ts=d.ts
    ORDER BY r.ts ASC
    """
    rows = q(sql, (hid, hours))
    data = {
        "labels": [],
        "consumption": [],
        "predicted": [],
        "residual": [],
        "zscore": [],
        "anomaly": [],
        "persistent": [],
        "if_flag": [],
        "flow_rate": [],
        "pressure": []
    }
    for r in rows:
        data["labels"].append(r["ts"])
        data["consumption"].append(round(r["consumption"], 1) if r["consumption"] is not None else 0)
        data["predicted"].append(round(r["predicted"], 1) if r["predicted"] is not None else None)
        data["residual"].append(round(r["residual"], 1) if r["residual"] is not None else None)
        data["zscore"].append(round(r["zscore"], 2) if r["zscore"] is not None else None)
        data["anomaly"].append(int(r["anomaly"]) if r["anomaly"] is not None else 0)
        data["persistent"].append(int(r["persistent"]) if r["persistent"] is not None else 0)
        data["if_flag"].append(int(r["if_flag"]) if r["if_flag"] is not None else 0)
        data["flow_rate"].append(round(r["flow_rate"], 2) if r["flow_rate"] is not None else 0)
        data["pressure"].append(round(r["pressure"], 2) if r["pressure"] is not None else 0)
    return data


def get_alerts_list(hid=1, status=None, limit=50):
    if status:
        return [dict(r) for r in q("SELECT * FROM alerts WHERE household_id=? AND status=? ORDER BY start_ts DESC LIMIT ?",
                                   (hid, status, limit))]
    return [dict(r) for r in q("SELECT * FROM alerts WHERE household_id=? ORDER BY start_ts DESC LIMIT ?",
                               (hid, limit))]


def update_alert_status(alert_id, new_status):
    with conn() as c:
        c.execute("UPDATE alerts SET status=? WHERE id=?", (new_status, alert_id))
    return True
