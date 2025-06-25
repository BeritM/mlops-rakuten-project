# Monitoring Setup

This document provides an end-to-end guide for setting up model monitoring using Evidently, Prometheus and Grafana. 
It covers everything from the existing exporter in the FastAPI service (the exporter lives in: plugins/cd4ml/inference/predict_service.py) 
to alerting and dashboard provisioning.


# Prerequisites
1. Python Exporter (FastAPI Service in predict_service.py)

2. Prometheus/Alertmanager Setup  
  2.1 prometheus.yml  
  2.2 alert_rules.yml  
  2.3 alertmanager.yml  

3. Grafana Setup
    3.1 dashboards.yml
    3.2 Dashboard JSON Files

# Project Structure

monitoring/
├── grafana/
│   └── provisioning/
│       └── dashboards/
│           ├── authentication_dashboard.json
│           ├── drift_and_f1_dashboard.json
│           ├── prediction_performance_dashboard.json
│           └── dashboards.yml
└── prometheus/
    ├── prometheus.yml
    ├── alert_rules.yml
    └── alertmanager.yml
plugins/
└── cd4ml/
    └── inference/
        └── predict_service.py  # FastAPI exporter

# Step-by-Step File Contents

## plugins/cd4ml/inference/predict_service.py

- Imported and defined Gauge, Counter, Histogram metrics.

- Exposed /metrics endpoint using generate_latest().

## monitoring/prometheus/prometheus.yml

- Configure global scrape interval and include rule_files.

- Add scrape_configs for prediction_service and api_probe (Blackbox Exporter).

## monitoring/prometheus/alert_rules.yml

- Define the ModelF1Drop alert rule (10% drop, 5m for).

## monitoring/prometheus/alertmanager.yml

- Set the global resolve_timeout and route to airflow_retrain.

- Configure webhook_configs to post to Airflow's DAG-run endpoint.

## monitoring/grafana/provisioning/dashboards/dashboards.yml

- Declare the file-based provider pointing to the dashboards/ folder.

# monitoring/grafana/provisioning/dashboards/*.json

- JSON dashboard files authentication, drift & F1, prediction performance are visible in Grafana as Dashboard panels.



# Running the Stack

- Blackbox Exporter:

```bash
docker run -d --name blackbox-exporter -p 9115:9115 prom/blackbox-exporter
```

- Prometheus:

```bash
docker run -d --name prometheus -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v $(pwd)/monitoring/prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml \
  prom/prometheus
```

- Alertmanager:

```bash
docker run -d --name alertmanager -p 9093:9093 \
  -v $(pwd)/monitoring/prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml \
  prom/alertmanager
```

- Grafana:

```bash
docker run -d --name grafana -p 3000:3000 \
  -v $(pwd)/monitoring/grafana/provisioning:/etc/grafana/provisioning \
  grafana/grafana
```

# Verify running services on localhost:

Auth metrics: http://localhost:8001/metrics  

Predict metrics: http://localhost:8002/metrics  

Prometheus UI: http://localhost:9090

Grafana UI: http://localhost:3000