# drift_monitor/generate_drift_report.py

import os
import pandas as pd

# ─── NumPy-Monkey-Patch für NumPy 2.x & Evidently 0.2.8 ───
import numpy as np
np.float_ = np.float64
np.int_   = np.int64
np.bool_  = np.bool_
# ────────────────────────────────────────────────────────────

from evidently.report import Report
from evidently.metric_preset.data_drift import DataDriftPreset

def generate_evidently_report(
    reference_x_csv: str,
    current_csv: str,
    html_path: str,
    json_path: str
):
    """
    Erzeugt einen Data-Drift-Report (Evidently 0.2.8) basierend auf:
      • reference_x_csv – Feature-CSV (X_test_update.csv, enthält
        ggf. 'Unnamed: 0', 'productid', 'imageid', plus 'designation' & 'description')
      • current_csv     – Feedback-CSV  (feedback.csv, enthält
        'designation', 'description', 'correct_code')
    Speichert den HTML- und JSON-Report direkt im drift_monitor-Ordner.
    """

    # 1) Existenz prüfen
    if not os.path.exists(reference_x_csv):
        raise FileNotFoundError(f"Feature-CSV nicht gefunden: {reference_x_csv}")
    if not os.path.exists(current_csv):
        raise FileNotFoundError(f"Feedback-CSV nicht gefunden: {current_csv}")

    # 2) Feature-CSV einlesen und überflüssige Spalten entfernen
    df_x = pd.read_csv(reference_x_csv)
    for drop_col in ("Unnamed: 0", "productid", "imageid", "correct_code"):
        if drop_col in df_x.columns:
            df_x = df_x.drop(columns=[drop_col])

    # Sicherstellen, dass mindestens 'designation' und 'description' da sind
    required = {"designation", "description"}
    if not required.issubset(df_x.columns):
        missing = required - set(df_x.columns)
        raise KeyError(f"Fehlende Spalten in Feature-CSV: {missing}")

    # 3) Feedback-CSV einlesen
    df_feedback = pd.read_csv(current_csv)
    if not required.issubset(df_feedback.columns):
        missing = required - set(df_feedback.columns)
        raise KeyError(f"Fehlende Spalten in Feedback-CSV: {missing}")
    if "correct_code" not in df_feedback.columns:
        raise KeyError("'correct_code' fehlt in Feedback-CSV")

    # 4) Nur Text-Spalten extrahieren (Drift auf Textbasis)
    df_ref_text = df_x[["designation", "description"]].copy()
    df_new_text = df_feedback[["designation", "description"]].copy()

    # 5) Evidently-Report auf Text-Drift erstellen
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_ref_text, current_data=df_new_text)

    # 6) Report speichern
    report.save_html(html_path)
    report.save_json(json_path)
