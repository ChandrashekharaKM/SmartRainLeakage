# 💧 Smart Water Consumption & Leakage Detection (SmartRain)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-SQLite%20%2F%20MySQL%20Ready-lightgrey.svg)](https://sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intelligent IoT and Machine Learning application designed to monitor household water usage, learn domestic diurnal profiles, and detect hidden or persistent water pipe leaks in real time before severe wastage occurs.

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [System Modules](#-system-modules)
3. [Machine Learning Methodology](#-machine-learning-methodology)
4. [Installation & Setup](#-installation--setup)
5. [Quick Start](#-quick-start)
6. [Interactive IoT Simulator](#-interactive-iot-simulator)
7. [Screenshots & UI Walkthrough](#-screenshots--ui-walkthrough)
8. [Project Structure](#-project-structure)
9. [REST API Documentation](#-rest-api-documentation)
10. [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 🌊 Overview
Traditional water leakage inspection relies on manual spot checks, acoustic rods, or monthly billing anomalies, frequently missing small, subterranean, or concealed plumbing leaks until thousands of litres are lost.

**SmartRain** combines **high-frequency time-series regression**, **heteroscedastic statistical noise modeling**, and **persistence-run anomaly filters** to deliver:
- Continuous hourly flow and pressure monitoring.
- Learning of normal household diurnal consumption cycles.
- Early detection of abnormal flow rates that indicate pipe or fixture leakage.
- Automated alert logging and real-time dashboard notifications.

---

## 🧩 System Modules

The project is structured according to the 8 core engineering modules:

| Module | Purpose | Implementation Details |
| :--- | :--- | :--- |
| **Module 1: User Module** | Access control & multi-tenancy | Session-based authentication (`demo`/`demo123`, `admin`/`admin123`) and multi-household switching. |
| **Module 2: Data Collection** | Sensor & CSV ingestion | Flexible schema normalizer mapping disparate columns (`Date`, `Time`, `FlowRate`, `Consumption`, `Duration`, `Pressure`, `Leakage`). |
| **Module 3: Preprocessing** | Data hygiene & standardization | Deduplication, negative entry-error filtering, time-based hourly resampling, forward/backward missing value interpolation. |
| **Module 4: Machine Learning** | Supervised consumption modeling | Benchmarking of **Linear Regression**, **Decision Trees**, **Random Forest**, and **XGBoost** trained strictly on normal domestic behavior. |
| **Module 5: Leakage Detection** | Residual & persistence analytics | Z-score anomaly thresholding ($Z > 3.0\sigma$), consecutive run persistence filtering ($\ge 3$ hrs), and unsupervised **Isolation Forest** comparison. |
| **Module 6: Database Module** | Persistent relational storage | SQLite database (`data/water.db`) storing `users`, `readings`, `detections`, and `alerts`. |
| **Module 7: Dashboard Module** | Visual analytics & monitoring | Modern glassmorphic dark-mode web interface with interactive Chart.js graphs, multi-window filters, and metric gauges. |
| **Module 8: Alert Module** | Notification dispatch & audit | Severity classification (Low, Medium, High), alert acknowledgment/resolution workflows, and notification logging (`logs/alerts.log`). |

---

## 🧠 Machine Learning Methodology

### 1. Feature Engineering
Models are trained with periodic temporal and historical consumption features:
- **Hour & Day of Week**: Cyclic encoding via $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$.
- **Weekend Indicator**: Captures increased Saturday/Sunday domestic consumption.
- **Historical Medians**:
  - `hist_day`: Median consumption for the same hour across the past 7 days.
  - `hist_week`: Median consumption for the same hour-of-week across the past 4 weeks.
  *(Medians are used instead of means so transient leaks do not corrupt the baseline).*

### 2. Residual Z-Score & Noise Model
Water usage is naturally more variable during peak daytime hours than at 3:00 AM. A linear noise model is fitted to prediction magnitude:
$$\sigma = (c_0 + c_1 \cdot \hat{y}) \cdot 1.2533$$
$$Z = \frac{y_{\text{actual}} - \hat{y}_{\text{predicted}}}{\sigma}$$

### 3. Persistence Filter
A transient spike (e.g., filling a garden tank or car wash) triggers $Z > 3.0$ for an isolated hour. A leak flows **non-stop**. The persistence engine only flags an official **Possible Leakage** when abnormal residual flow persists for $\ge 3$ consecutive hours.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Clone the Repository
```bash
git clone https://github.com/ChandrashekharaKM/SmartRainLeakage.git
cd SmartRainLeakage
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Train Models & Initialize Database
Generate the 120-day benchmark dataset, train and calibrate all regression models, and seed the SQLite database:
```bash
python train.py
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
**`http://127.0.0.1:5000`**

### 3. Login Credentials
| Role | Username | Password |
| :--- | :--- | :--- |
| **Standard User** | `demo` | `demo123` |
| **Administrator** | `admin` | `admin123` |

---

## ⚡ Interactive IoT Simulator
You can simulate live IoT hardware (ESP32 + flow & pressure sensors) directly through the browser without physical wiring:
1. Navigate to **`http://127.0.0.1:5000/simulate`** (or use the quick controls on the dashboard).
2. Choose a simulation scenario:
   - **Normal Flow**: Generates realistic diurnal usage and tap durations (1–8 min/h).
   - **Transient Surge**: Injects a single high-usage hour ($Z > 3.0\sigma$) — demonstrates that the model suppresses false alarms.
   - **Continuous Leakage**: Injects consecutive leaking hours ($55–60\text{ min/h}$ flow with pressure drops) — triggers persistence detection and generates an alert.

---

## 📂 Project Structure

```
SmartRainLeakage/
├── data/
│   └── water_consumption.csv   # 120-day benchmark time-series dataset
├── models/
│   ├── model.joblib            # Serialized best ML pipeline & noise calibrator
│   └── metrics.json            # Model evaluation & sensitivity metrics
├── static/
│   ├── css/
│   │   └── style.css           # Modern glassmorphic dark theme stylesheet
│   └── js/
│       └── dashboard.js        # Chart.js integration, live gauges & simulation AJAX
├── templates/
│   ├── base.html               # Base layout, navbar & session management
│   ├── login.html              # Authentication portal with quick demo buttons
│   ├── index.html              # Main monitoring dashboard & live charts
│   ├── analytics.html          # Dual-axis telemetry & residual distribution
│   ├── ml_models.html          # ML benchmarks, metrics & retraining controls
│   ├── alerts.html             # Incident audit table & notification dispatcher
│   ├── simulate.html           # Real-time IoT hardware simulator
│   └── dataset.html            # Preprocessing metrics & custom CSV uploader
├── app.py                      # Flask web application & REST API server
├── db.py                       # SQLite database manager & schema definitions
├── generate_dataset.py         # Realistic household dataset synthesizer
├── ml_pipeline.py              # Cleaning, feature extraction, ML training & scoring
├── service.py                  # Service layer connecting DB, ML scoring & alerts
├── train.py                    # CLI training pipeline & DB seeder
├── test_app.py                 # Automated test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/summary` | `GET` | Fetches current status, latest flow/pressure telemetry, 24h usage, and open alert counts. |
| `/api/chart-data?hours={N}` | `GET` | Returns aligned time-series of actual consumption, predicted consumption, residuals, and anomaly flags for the last `N` hours. |
| `/api/simulate` | `POST` | Injects synthetic sensor readings (`{"hours": 3, "leak": true}`). |
| `/api/alerts/{id}/status` | `POST` | Updates alert lifecycle status (`{"status": "acknowledged" \| "resolved"}`). |
| `/api/retrain` | `POST` | Re-fits ML models with custom Z-score threshold and persistence criteria. |
| `/api/upload-csv` | `POST` | Ingests and standardizes an external water dataset CSV and triggers pipeline retraining. |

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
