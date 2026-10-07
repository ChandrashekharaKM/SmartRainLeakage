"""
app.py  -  Web Application & API Server for Smart Water Monitoring
==================================================================
Modules integrated:
  Module 1: User Module (Auth & Sessions)
  Module 2: Data Collection Module (CSV Upload & Ingestion)
  Module 3: Data Preprocessing Module (Data Validation & Cleaning)
  Module 4: Machine Learning Module (Regression & Isolation Forest)
  Module 5: Leakage Detection Module (Residual z-score & Persistence)
  Module 6: Database Module (SQLite/MySQL via db.py)
  Module 7: Dashboard Module (Interactive UI & Visualizations)
  Module 8: Alert Module (Alerts management & Logs)
"""
import os
import json
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash, send_file)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd

import db
import service
import ml_pipeline as ml
import train as trainer

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE, "data")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")
ALERT_LOG = os.path.join(BASE, "logs", "alerts.log")

app = Flask(__name__)
app.secret_key = "smartrain_super_secure_key_water_ml_2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


@app.before_request
def ensure_db():
    # Guarantee tables exist on startup
    db.init_db(reset=False)
    if not os.path.exists(trainer.MODEL_PATH) and os.path.exists(trainer.DATA_PATH):
        try:
            trainer.train()
        except Exception:
            pass


@app.context_processor
def inject_global_data():
    if "user_id" in session:
        hid = session.get("household_id", 1)
        open_alerts = db.q("SELECT COUNT(*) c FROM alerts WHERE household_id=? AND status='open'", (hid,), one=True)
        return {
            "current_user": session.get("username", "Guest"),
            "current_hid": hid,
            "open_alerts_count": open_alerts["c"] if open_alerts else 0
        }
    return {"current_user": None, "current_hid": 1, "open_alerts_count": 0}


# ==============================================================================
# AUTHENTICATION (Module 1: User Module)
# ==============================================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = db.q("SELECT * FROM users WHERE username=?", (username,), one=True)
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["household_id"] = user["household_id"]
            flash("Welcome back, " + user["username"] + "!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Try demo / demo123 or admin / admin123", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Successfully logged out.", "info")
    return redirect(url_for("login"))


@app.route("/switch-household/<int:hid>")
@login_required
def switch_household(hid):
    session["household_id"] = hid
    flash(f"Switched active view to Household #{hid}", "success")
    return redirect(request.referrer or url_for("dashboard"))


# ==============================================================================
# MAIN DASHBOARD (Module 7: Dashboard Module)
# ==============================================================================
@app.route("/")
@login_required
def dashboard():
    hid = session.get("household_id", 1)
    summary = service.get_dashboard_summary(hid)
    metrics = load_metrics()
    recent_alerts = service.get_alerts_list(hid, limit=5)
    return render_template("index.html", summary=summary, metrics=metrics, recent_alerts=recent_alerts)


# ==============================================================================
# ANALYTICS & RESIDUALS (Module 5 & 7)
# ==============================================================================
@app.route("/analytics")
@login_required
def analytics():
    hid = session.get("household_id", 1)
    summary = service.get_dashboard_summary(hid)
    return render_template("analytics.html", summary=summary)


# ==============================================================================
# MACHINE LEARNING MODULE (Module 4 & 5)
# ==============================================================================
@app.route("/models")
@login_required
def models_view():
    metrics = load_metrics()
    return render_template("ml_models.html", metrics=metrics)


# ==============================================================================
# ALERTS & NOTIFICATIONS (Module 8: Alert Module)
# ==============================================================================
@app.route("/alerts")
@login_required
def alerts_view():
    hid = session.get("household_id", 1)
    status_filter = request.args.get("status")
    alerts = service.get_alerts_list(hid, status=status_filter, limit=100)

    log_lines = []
    if os.path.exists(ALERT_LOG):
        with open(ALERT_LOG, "r", encoding="utf-8") as f:
            log_lines = f.readlines()[-30:]
            log_lines.reverse()

    return render_template("alerts.html", alerts=alerts, log_lines=log_lines, active_filter=status_filter)


# ==============================================================================
# SIMULATION & IOT TESTING (Simulates ESP32 sensors)
# ==============================================================================
@app.route("/simulate")
@login_required
def simulate_view():
    hid = session.get("household_id", 1)
    summary = service.get_dashboard_summary(hid)
    return render_template("simulate.html", summary=summary)


# ==============================================================================
# DATASET & PREPROCESSING (Module 2 & 3)
# ==============================================================================
@app.route("/dataset")
@login_required
def dataset_view():
    hid = session.get("household_id", 1)
    metrics = load_metrics()
    readings_sample = db.q("SELECT * FROM readings WHERE household_id=? ORDER BY ts DESC LIMIT 25", (hid,))
    csv_exists = os.path.exists(trainer.DATA_PATH)
    file_size_kb = round(os.path.getsize(trainer.DATA_PATH) / 1024, 1) if csv_exists else 0

    return render_template("dataset.html",
                           metrics=metrics,
                           readings_sample=readings_sample,
                           csv_exists=csv_exists,
                           file_size_kb=file_size_kb)


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================
@app.route("/api/summary")
@login_required
def api_summary():
    hid = session.get("household_id", 1)
    summary = service.get_dashboard_summary(hid)
    return jsonify(summary)


@app.route("/api/chart-data")
@login_required
def api_chart_data():
    hid = session.get("household_id", 1)
    hours = int(request.args.get("hours", 72))
    hours = max(12, min(hours, 720))  # Between 12 hours and 30 days
    data = service.get_chart_data(hid=hid, hours=hours)
    return jsonify(data)


@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    hid = session.get("household_id", 1)
    payload = request.get_json(silent=True) or {}
    hours = int(payload.get("hours", 1))
    leak = bool(payload.get("leak", False))
    try:
        res = service.simulate(hid=hid, hours=hours, leak=leak)
        return jsonify({"success": True, "result": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/alerts/<int:alert_id>/status", methods=["POST"])
@login_required
def api_update_alert(alert_id):
    payload = request.get_json(silent=True) or {}
    new_status = payload.get("status")
    if new_status not in ["open", "acknowledged", "resolved"]:
        return jsonify({"error": "Invalid status"}), 400
    service.update_alert_status(alert_id, new_status)
    return jsonify({"success": True, "alert_id": alert_id, "status": new_status})


@app.route("/api/retrain", methods=["POST"])
@login_required
def api_retrain():
    payload = request.get_json(silent=True) or {}
    z_val = float(payload.get("z_threshold", 3.0))
    persist_val = int(payload.get("min_persist", 3))

    try:
        new_metrics = trainer.train(z=z_val, n_persist=persist_val)
        service.reload_bundle()
        return jsonify({"success": True, "metrics": new_metrics})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/upload-csv", methods=["POST"])
@login_required
def api_upload_csv():
    if "file" not in request.files:
        flash("No file part provided", "error")
        return redirect(url_for("dataset_view"))
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("dataset_view"))

    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        dest = os.path.join(app.config["UPLOAD_FOLDER"], "water_consumption.csv")
        file.save(dest)
        try:
            # Re-train models on the newly uploaded dataset
            trainer.train(csv_path=dest)
            service.reload_bundle()
            flash("Dataset successfully uploaded, preprocessed and models retrained!", "success")
        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "error")
        return redirect(url_for("dataset_view"))
    else:
        flash("Please upload a valid .csv file", "error")
        return redirect(url_for("dataset_view"))


@app.route("/download-sample-csv")
@login_required
def download_sample_csv():
    if os.path.exists(trainer.DATA_PATH):
        return send_file(trainer.DATA_PATH, as_attachment=True, download_name="water_consumption_sample.csv")
    flash("Sample dataset not found.", "error")
    return redirect(url_for("dataset_view"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
