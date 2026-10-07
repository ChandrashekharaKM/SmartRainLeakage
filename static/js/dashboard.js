/* SmartRain - Interactive Dashboard & Real-Time Visualization */

let consumptionChart = null;
let telemetryChart = null;
let residualChart = null;
let currentHours = 72;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initTimeFilters();
  initSimulationButtons();
  initAlertActions();
  initRetrainForm();

  // If chart containers are present, load charts
  if (document.getElementById("consumptionChart")) {
    loadChartData(currentHours);
  }
});

function initTimeFilters() {
  const filterBtns = document.querySelectorAll(".btn-filter[data-hours]");
  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentHours = parseInt(btn.dataset.hours, 10);
      loadChartData(currentHours);
    });
  });
}

async function loadChartData(hours) {
  try {
    const res = await fetch(`/api/chart-data?hours=${hours}`);
    const data = await res.json();
    renderConsumptionChart(data);
    renderTelemetryChart(data);
    renderResidualChart(data);
  } catch (err) {
    console.error("Error loading chart data:", err);
  }
}

function renderConsumptionChart(data) {
  const ctx = document.getElementById("consumptionChart");
  if (!ctx) return;

  // Identify anomaly points for red marker display
  const pointColors = data.anomaly.map(a => a === 1 ? "#f43f5e" : "#06b6d4");
  const pointRadii = data.anomaly.map(a => a === 1 ? 6 : 2);

  const chartConfig = {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Actual Consumption (L/h)",
          data: data.consumption,
          borderColor: "#22d3ee",
          backgroundColor: "rgba(6, 182, 212, 0.12)",
          borderWidth: 2.2,
          fill: true,
          tension: 0.35,
          pointBackgroundColor: pointColors,
          pointBorderColor: pointColors,
          pointRadius: pointRadii,
        },
        {
          label: "Expected / Predicted (L/h)",
          data: data.predicted,
          borderColor: "rgba(148, 163, 184, 0.75)",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.35,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
        },
        tooltip: {
          backgroundColor: "rgba(12, 24, 47, 0.95)",
          titleColor: "#f8fafc",
          bodyColor: "#94a3b8",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            afterBody: function(tooltipItems) {
              const idx = tooltipItems[0].dataIndex;
              const isAnom = data.anomaly[idx] === 1;
              const isPersist = data.persistent[idx] === 1;
              if (isPersist) return ["⚠️ Persistent Leak Detected!"];
              if (isAnom) return ["⚠️ Consumption Anomaly"];
              return ["✓ Normal Consumption"];
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 12 },
          grid: { color: "rgba(255, 255, 255, 0.04)" }
        },
        y: {
          title: { display: true, text: "Litres / Hour", color: "#64748b" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          beginAtZero: true
        }
      }
    }
  };

  if (consumptionChart) {
    consumptionChart.destroy();
  }
  consumptionChart = new Chart(ctx, chartConfig);
}

function renderTelemetryChart(data) {
  const ctx = document.getElementById("telemetryChart");
  if (!ctx) return;

  const chartConfig = {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Flow Rate (L/min)",
          data: data.flow_rate,
          borderColor: "#3b82f6",
          backgroundColor: "transparent",
          borderWidth: 2,
          yAxisID: "yFlow",
          tension: 0.3,
          pointRadius: 1
        },
        {
          label: "Pressure (bar)",
          data: data.pressure,
          borderColor: "#10b981",
          backgroundColor: "transparent",
          borderWidth: 2,
          yAxisID: "yPress",
          tension: 0.3,
          pointRadius: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
        }
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 10 },
          grid: { color: "rgba(255, 255, 255, 0.04)" }
        },
        yFlow: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Flow Rate (L/min)", color: "#3b82f6" },
          ticks: { color: "#3b82f6" },
          grid: { color: "rgba(255, 255, 255, 0.04)" }
        },
        yPress: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Pressure (bar)", color: "#10b981" },
          ticks: { color: "#10b981" },
          grid: { drawOnChartArea: false }
        }
      }
    }
  };

  if (telemetryChart) {
    telemetryChart.destroy();
  }
  telemetryChart = new Chart(ctx, chartConfig);
}

function renderResidualChart(data) {
  const ctx = document.getElementById("residualChart");
  if (!ctx) return;

  const chartConfig = {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Residual Z-Score",
          data: data.zscore,
          backgroundColor: data.zscore.map(z => z >= 3.0 ? "rgba(244, 63, 94, 0.75)" : "rgba(59, 130, 246, 0.45)"),
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8" }
        }
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 10 },
          grid: { color: "rgba(255, 255, 255, 0.04)" }
        },
        y: {
          title: { display: true, text: "Z-Score (σ)", color: "#64748b" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(255, 255, 255, 0.04)" }
        }
      }
    }
  };

  if (residualChart) {
    residualChart.destroy();
  }
  residualChart = new Chart(ctx, chartConfig);
}

// -------------------------------------------------------------
// LIVE SIMULATION CONTROLS
// -------------------------------------------------------------
function initSimulationButtons() {
  const simBtns = document.querySelectorAll(".btn-sim");
  simBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      const hours = parseInt(btn.dataset.hours || "1", 10);
      const leak = btn.dataset.leak === "true";
      const originalText = btn.innerHTML;

      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span> Simulating...`;

      try {
        const res = await fetch("/api/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ hours, leak })
        });
        const result = await res.json();
        if (result.success) {
          showToast(`Injected ${hours}h ${leak ? "LEAK" : "normal"} sensor reading(s)!`, "success");
          // Refresh chart and dashboard metrics
          await loadChartData(currentHours);
          await refreshSummaryCards();
        } else {
          showToast(`Simulation error: ${result.error}`, "error");
        }
      } catch (e) {
        showToast(`Simulation request failed`, "error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    });
  });
}

async function refreshSummaryCards() {
  try {
    const res = await fetch("/api/summary");
    const s = await res.json();
    const heroTitle = document.getElementById("statusHeroTitle");
    const heroDesc = document.getElementById("statusHeroDesc");
    const heroBanner = document.getElementById("statusHeroBanner");

    if (heroBanner && heroTitle) {
      if (s.status.includes("Leak")) {
        heroBanner.className = "status-hero status-leak";
        heroTitle.innerHTML = "⚠️ " + s.status;
        heroDesc.innerText = "Abnormal continuous water flow detected across multiple hours!";
      } else if (s.status.includes("Anomaly")) {
        heroBanner.className = "status-hero status-warning";
        heroTitle.innerHTML = "⚡ " + s.status;
        heroDesc.innerText = "Short term flow surge detected. Monitoring for persistence...";
      } else {
        heroBanner.className = "status-hero status-normal";
        heroTitle.innerHTML = "✓ System Normal";
        heroDesc.innerText = "Consumption patterns match baseline machine learning profile.";
      }
    }

    // Update numbers if elements exist
    const valFlow = document.getElementById("valFlowRate");
    if (valFlow && s.latest_reading) valFlow.innerText = s.latest_reading.flow_rate;

    const valPress = document.getElementById("valPressure");
    if (valPress && s.latest_reading) valPress.innerText = s.latest_reading.pressure;

    const val24h = document.getElementById("val24hUsage");
    if (val24h) val24h.innerText = s.usage_24h;

    const valAlerts = document.getElementById("valOpenAlerts");
    if (valAlerts) valAlerts.innerText = s.open_alerts;

  } catch (err) {
    console.error("Error updating summary cards:", err);
  }
}

// -------------------------------------------------------------
// ALERT ACTIONS
// -------------------------------------------------------------
function initAlertActions() {
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-alert-action");
    if (!btn) return;

    const alertId = btn.dataset.id;
    const newStatus = btn.dataset.status;

    try {
      const res = await fetch(`/api/alerts/${alertId}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus })
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Alert marked as ${newStatus}`, "success");
        setTimeout(() => location.reload(), 600);
      }
    } catch (err) {
      showToast("Failed to update alert", "error");
    }
  });
}

// -------------------------------------------------------------
// MODEL RETRAIN FORM
// -------------------------------------------------------------
function initRetrainForm() {
  const form = document.getElementById("retrainModelForm");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = form.querySelector("button[type='submit']");
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = "Retraining ML Models...";

    const zVal = parseFloat(document.getElementById("retrainZ").value || 3.0);
    const persistVal = parseInt(document.getElementById("retrainPersist").value || 3, 10);

    try {
      const res = await fetch("/api/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ z_threshold: zVal, min_persist: persistVal })
      });
      const data = await res.json();
      if (data.success) {
        showToast("Models retrained and calibrated successfully!", "success");
        setTimeout(() => location.reload(), 1000);
      } else {
        showToast(`Retraining failed: ${data.error}`, "error");
      }
    } catch (err) {
      showToast("Retraining error", "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalText;
    }
  });
}

// -------------------------------------------------------------
// TOAST NOTIFICATIONS
// -------------------------------------------------------------
function showToast(message, type = "info") {
  let toastContainer = document.getElementById("toastContainer");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toastContainer";
    toastContainer.style.position = "fixed";
    toastContainer.style.bottom = "20px";
    toastContainer.style.right = "20px";
    toastContainer.style.zIndex = "9999";
    toastContainer.style.display = "flex";
    toastContainer.style.flexDirection = "column";
    toastContainer.style.gap = "10px";
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement("div");
  toast.className = `flash-msg flash-${type}`;
  toast.style.boxShadow = "0 8px 24px rgba(0,0,0,0.4)";
  toast.style.minWidth = "260px";
  toast.innerText = message;

  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = "opacity 0.4s ease";
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 400);
  }, 3500);
}
