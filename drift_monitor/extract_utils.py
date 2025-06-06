# drift_monitor/extract_utils.py

import json

def extract_drift_metrics(json_path: str) -> dict:
    """
    Liest den JSON-Report aus Evidently (DataDriftPreset) ein und
    extrahiert die relevanten Metriken: 'text_drift_score' und 'global_drift'.

    Hinweis: Das JSON von Evidently 0.2.8 enthält typischerweise unter "metrics"
    ein Dictionary, in dem der Schlüssel "DatasetDriftMetric" (bzw. "data_drift")
    die Drift-Ergebnisse speichert. Je nach Version kann das leicht variieren.

    Diese Funktion versucht, folgende Felder zu lesen:
      • metrics.DatasetDriftMetric.drift_detected  → int (0 oder 1) als global_drift
      • metrics.DatasetDriftMetric.drift_score     → float  als text_drift_score

    Falls ein Eintrag fehlt, wird 0 bzw. 0.0 zurückgegeben.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # In Evidently 0.2.8 liegt das Data-Drift-Resultat meist unter payload["metrics"]["DatasetDriftMetric"]
    metrics = payload.get("metrics", {})

    # Versuche, DataDriftMetric zu finden (manchmal heißt es "DatasetDriftMetric")
    drift_metric = None
    for key in ("DatasetDriftMetric", "data_drift", "DataDriftMetric"):
        if key in metrics:
            drift_metric = metrics[key]
            break

    if drift_metric is None:
        # Falls wir nichts finden, geben wir Default-Werte zurück
        return {"text_drift_score": 0.0, "global_drift": 0}

    # 'drift_detected' ist üblicherweise ein bool oder 0/1
    global_drift = drift_metric.get("drift_detected")
    if isinstance(global_drift, bool):
        global_drift = int(global_drift)
    elif not isinstance(global_drift, int):
        try:
            global_drift = int(global_drift)
        except:
            global_drift = 0

    # 'drift_score' ist üblicherweise eine Zahl
    text_drift_score = drift_metric.get("drift_score", 0.0)
    try:
        text_drift_score = float(text_drift_score)
    except:
        text_drift_score = 0.0

    return {
        "text_drift_score": text_drift_score,
        "global_drift": global_drift
    }
