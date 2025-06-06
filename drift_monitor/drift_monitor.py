# drift_monitor/drift_monitor.py

#!/usr/bin/env python3

# ─── NumPy-Monkey-Patch für NumPy 2.x & Evidently 0.2.8 ───
import numpy as np
np.float_ = np.float64
np.int_   = np.int64
np.bool_  = np.bool_
# ─────────────────────────────────────────────────────────

import os
import subprocess
import sys
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from generate_drift_report import generate_evidently_report
from extract_utils import extract_drift_metrics

# Projekt-Root (eine Ebene über diesem Ordner)
ROOT_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR          = os.path.join(ROOT_DIR, "shared_volume", "data", "raw")
FEEDBACK_DIR     = os.path.join(ROOT_DIR, "shared_volume", "data", "feedback")
DRIFT_DIR        = os.path.dirname(__file__)  # drift_monitor-Ordner selbst

TRAIN_X_FILENAME = "X_test_update.csv"  # Referenz-Features (Train-CSV)
NEW_FILENAME     = "feedback.csv"       # Feedback-CSV

HTML_REPORT_NAME = "drift_report.html"
JSON_REPORT_NAME = "drift_report.json"

PUSHGATEWAY_URL = "http://localhost:9091"
PUSHGATEWAY_JOB = "drift_monitor"


def dvc_pull_if_needed(csv_path: str):
    """
    Führt 'dvc pull <csv_path>.dvc' aus, falls eine .dvc-Datei existiert.
    Ansonsten wird nur eine Warnung angezeigt und das Skript geht davon aus,
    dass die CSV bereits lokal vorliegt.
    """
    abs_csv = os.path.abspath(csv_path)
    dvc_file = abs_csv + ".dvc"
    if os.path.exists(dvc_file):
        try:
            subprocess.run(
                ["dvc", "pull", dvc_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"[INFO] DVC-Pull erfolgreich: {dvc_file}")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] DVC-Pull fehlgeschlagen: {dvc_file} ({e})")
    else:
        print(f"[WARN] Keine .dvc-Datei gefunden für {abs_csv} → Verwende lokale Datei, falls vorhanden")


def main():
    print("=" * 60)
    print("STARTING DRIFT MONITORING")
    print("=" * 60)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    # Kein eigenes Processing-Verzeichnis mehr nötig – wir schreiben direkt ins drift_monitor-Verzeichnis

    train_x_csv = os.path.join(RAW_DIR, TRAIN_X_FILENAME)
    new_csv     = os.path.join(FEEDBACK_DIR, NEW_FILENAME)

    html_out = os.path.join(DRIFT_DIR, HTML_REPORT_NAME)
    json_out = os.path.join(DRIFT_DIR, JSON_REPORT_NAME)

    print(f"Raw Data Directory:       {RAW_DIR}")
    print(f"Feedback Data Directory:  {FEEDBACK_DIR}")
    print(f"Drift Report Directory:   {DRIFT_DIR}")
    print(f"TRAIN_X_CSV: {train_x_csv}")
    print(f"NEW_CSV:     {new_csv}")
    print(f"HTML Report: {html_out}")
    print(f"JSON Report: {json_out}")
    print(f"Pushgateway: {PUSHGATEWAY_URL}  Job: {PUSHGATEWAY_JOB}")

    print("\n[STEP] DVC-Pull der Dateien …")
    dvc_pull_if_needed(train_x_csv)
    dvc_pull_if_needed(new_csv)

    print("\n[STEP] Evidently-Report generieren …")
    try:
        generate_evidently_report(
            reference_x_csv=train_x_csv,
            current_csv=new_csv,
            html_path=html_out,
            json_path=json_out
        )
    except Exception as e:
        print(f"[ERROR] Evidently-Report-Erstellung fehlgeschlagen: {e}")
        return 1

    print("\n[STEP] Drift-Metriken extrahieren …")
    try:
        metrics = extract_drift_metrics(json_out)
    except Exception as e:
        print(f"[ERROR] Extrahieren der Metriken fehlgeschlagen: {e}")
        return 1

    # Beispielhaft nur Text-Drift ausgeben (Label-Drift entfällt)
    text_score   = metrics.get("text_drift_score", 0.0)
    global_drift = metrics.get("global_drift",     0)

    print("\n[STEP] Prometheus-Gauges befüllen & Pushgateway …")
    registry = CollectorRegistry()
    g_text   = Gauge("text_drift_score",   "Text Drift Score",    registry=registry)
    g_global = Gauge("global_drift",       "Globaler Drift-Flag", registry=registry)

    g_text.set(text_score)
    g_global.set(global_drift)

    try:
        push_to_gateway(PUSHGATEWAY_URL, job=PUSHGATEWAY_JOB, registry=registry)
        print(f"[INFO] Metriken gepusht → Pushgateway: {PUSHGATEWAY_URL}, Job: {PUSHGATEWAY_JOB}")
    except Exception as e:
        print(f"[ERROR] Push an Pushgateway fehlgeschlagen: {e}")
        return 1

    print("\n[FINISHED] Drift-Monitoring erfolgreich abgeschlossen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
