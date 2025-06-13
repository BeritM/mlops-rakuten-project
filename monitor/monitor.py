#!/usr/bin/env python3
import os
import time
import requests
import numpy as np
from sklearn.metrics import f1_score
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Environment-Variablen
PREDICT_URL      = os.environ['PREDICT_URL']
VALIDATION_PATH  = os.environ['VALIDATION_DATA']
PUSHGATEWAY_URL  = os.environ['PUSHGATEWAY_URL']
BATCH_SIZE       = int(os.environ.get('BATCH_SIZE', 100))
INTERVAL_SEC     = int(os.environ.get('INTERVAL_SEC', 300))

# Prometheus Registry und Metrik
registry = CollectorRegistry()
f1_gauge = Gauge(
    'weighted_f1_score',
    'Weighted F1-Score der Prediction-API in Prozent',
    registry=registry
)

def compute_and_push():
    # Wahre Labels laden
    y_true = np.load(VALIDATION_PATH)

    # Vorhersagen von der API abrufen
    response = requests.post(
        PREDICT_URL,
        json={'batch_size': BATCH_SIZE},
        timeout=30
    )
    response.raise_for_status()
    y_pred = np.array(response.json()['predictions'])

    # F1-Score berechnen (percentage)
    score = f1_score(y_true, y_pred, average='weighted') * 100

    # Metrik setzen und an Pushgateway senden
    f1_gauge.set(score)
    push_to_gateway(PUSHGATEWAY_URL, job='model_monitor', registry=registry)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] pushed weighted_f1_score={score:.2f}%")

if __name__ == "__main__":
    while True:
        try:
            compute_and_push()
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fehler beim Metrik-Push: {e}")
        time.sleep(INTERVAL_SEC)
