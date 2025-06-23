# README: Drift Monitor Component

This README describes the drift monitoring component in your MLOps project. It covers the purpose, architecture, setup, and usage of the drift detection logic using Evidently, Prometheus, Pushgateway, and Grafana.

## What is this?

This module runs regularly to check if the incoming data is drifting away from your model’s original training distribution. It uses:

- Evidently AI to calculate text drift
- Pushgateway to send metrics
- Prometheus + Grafana to visualize and monitor the drift

## Structure

mlops-rakuten-project/
├── drift_monitor/
│   ├── drift_monitor.py            # main entry point
│   ├── generate_drift_report.py   # builds the HTML/JSON report using Evidently
│   ├── extract_utils.py           # extracts metrics from JSON
│   └── ...
├── shared_volume/data/           # input data (DVC-managed)
├── logs/                         # drift reports saved here
├── docker-compose.prometheus.yml
├── prometheus.yml
└── README-drift.md               # this file

## How it works

1. DVC ensures data is available (raw/, feedback/)

2. drift_monitor.py runs, calls generate_drift_report.py

3. A JSON and HTML report is created in /logs/

4. Drift metrics are extracted and pushed to the Pushgateway

5. Prometheus scrapes them

6. Grafana visualizes them


# Quickstart

## Run the full monitoring stack
```bash
docker compose -f docker-compose.prometheus.yml up -d
```

## Run drift check manually (if needed)
```bash
python drift_monitor/drift_monitor.py
```

## Metrics Pushed

- Metric Name
- Description
- text_drift_score
- Aggregated drift score (0..1)
- global_drift
- Binary indicator (0 or 1)
- column_drift_score_*
- Drift scores per feature column

# Notes

- This component is compatible with Evidently 0.2.8
- Requires Python 3.10+ and NumPy 2.x
- Use logs/ for persistent access to past reports.

