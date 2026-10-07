"""
db.py  -  Module 6 (Database) of the synopsis
=============================================
SQLite is used so the project runs with zero setup. The SQL is deliberately plain, so
moving to MySQL only needs a different connection (e.g. mysql-connector / SQLAlchemy) and
swapping `INSERT OR REPLACE` for `INSERT ... ON DUPLICATE KEY UPDATE`.

Tables: users, readings, detections, alerts
"""
import os
import sqlite3
from werkzeug.security import generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "water.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    household_id INTEGER NOT NULL DEFAULT 1,
    created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS readings(
    household_id INTEGER, ts TEXT, consumption REAL, flow_rate REAL,
    duration REAL, pressure REAL, leak_label INTEGER,
    PRIMARY KEY(household_id, ts));
CREATE TABLE IF NOT EXISTS detections(
    household_id INTEGER, ts TEXT, predicted REAL, residual REAL, zscore REAL,
    anomaly INTEGER, persistent INTEGER, if_flag INTEGER,
    PRIMARY KEY(household_id, ts));
CREATE TABLE IF NOT EXISTS alerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    household_id INTEGER, start_ts TEXT, end_ts TEXT, hours INTEGER,
    excess_litres REAL, severity TEXT, status TEXT DEFAULT 'open',
    created TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(household_id, start_ts));
"""


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db(reset=False):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with conn() as c:
        c.executescript(SCHEMA)
        if not c.execute("SELECT 1 FROM users WHERE username='demo'").fetchone():
            c.execute("INSERT INTO users(username,password_hash,household_id) VALUES(?,?,1)",
                      ("demo", generate_password_hash("demo123")))
        if not c.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
            c.execute("INSERT INTO users(username,password_hash,household_id) VALUES(?,?,1)",
                      ("admin", generate_password_hash("admin123")))


def q(sql, args=(), one=False):
    with conn() as c:
        rows = c.execute(sql, args).fetchall()
    return (rows[0] if rows else None) if one else rows
