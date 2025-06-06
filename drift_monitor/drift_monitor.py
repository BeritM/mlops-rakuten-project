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
# shared_volume/data-Verzeichnis (Dort liegen raw.dvc, feedback.dvc usw.)
DVC_DATA_DIR     = os.path.join(ROOT_DIR, "shared_volume", "data")
RAW_DIR          = os.path.join(DVC_DATA_DIR, "raw")
FEEDBACK_DIR     = os.path.join(DVC_DATA_DIR, "feedback")
DRIFT_DIR        = os.path.dirname(__file__)  # drift_monitor-Ordner selbst

TRAIN_X_FILENAME = "X_test_update.csv"
NEW_FILENAME     = "feedback.csv"

HTML_REPORT_NAME = "drift_report.html"
JSON_REPORT_NAME = "drift_report.json"

PUSHGATEWAY_URL  = "http://localhost:9091"  # ggf. später auf "http://pushgateway:9091" ändern
PUSHGATEWAY_JOB  = "drift_monitor"


def dvc_pull_if_needed_for_folder(folder_name: str):
    """
    Wenn 'shared_volume/data/<folder_name>' per DVC versioniert ist,
    führt 'dvc pull shared_volume/data/<folder_name>.dvc' aus.
    Sonst Warnung.
    Beispiel: folder_name='raw' → 'shared_volume/data/raw.dvc'
    """
    # DVC-Datei liegt direkt unter shared_volume/data, nicht in raw/ oder feedback/
    dvc_file = os.path.join(DVC_DATA_DIR, f"{folder_name}.dvc")
    if os.path.exists(dvc_file):
        try:
            print(f"[INFO] Versuche DVC-Pull für: {dvc_file}")
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
        print(f"[WARN] Keine .dvc-Datei gefunden: {dvc_file} → Verwende lokale Dateien im Ordner '{folder_name}'")


def main():
    print("=" * 60)
    print("STARTING DRIFT MONITORING")
    print("=" * 60)

    # Verzeichnisse anlegen, falls sie noch nicht existieren
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    # Bericht wird direkt in drift_monitor-Ordner geschrieben, kein eigenes processed-Verzeichnis nötig

    train_x_csv = os.path.join(RAW_DIR, TRAIN_X_FILENAME)
    new_csv     = os.path.join(FEEDBACK_DIR, NEW_FILENAME)

    DRIFT_DIR = os.path.dirname(__file__) 
    html_out  = os.path.join(DRIFT_DIR, "drift_report.html")
    json_out  = os.path.join(DRIFT_DIR, "drift_report.json")

    print(f"Raw Data Directory:       {RAW_DIR}")
    print(f"Feedback Data Directory:  {FEEDBACK_DIR}")
    print(f"Drift Report Directory:   {DRIFT_DIR}")
    print(f"TRAIN_X_CSV: {train_x_csv}")
    print(f"NEW_CSV:     {new_csv}")
    print(f"HTML Report: {html_out}")
    print(f"JSON Report: {json_out}")
    print(f"Pushgateway: {PUSHGATEWAY_URL}  Job: {PUSHGATEWAY_JOB}")

    print("\n[STEP] DVC-Pull der Ordner …")
    # Statt jede CSV einzeln: Wir ziehen den gesamten raw-Ordner und feedback-Ordner per DVC
    dvc_pull_if_needed_for_folder("raw")
    dvc_pull_if_needed_for_folder("feedback")

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

    # Beispiel: Nur Text-Drift-Werte
    text_score   = metrics.get("text_drift_score", 0.0)
    global_drift = metrics.get("global_drift",     0)

    print("\n[STEP] Prometheus-Gauges befüllen & Pushgateway …")
    registry = CollectorRegistry()
    g_text   = Gauge("text_drift_score", "Text Drift Score", registry=registry)
    g_global = Gauge("global_drift",     "Globaler Drift-Flag", registry=registry)

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
