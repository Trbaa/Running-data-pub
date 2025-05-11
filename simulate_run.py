import time
import pandas as pd
import numpy as np
from joblib import load

DATA_PATH   = "output/combined_data_with_predictions.csv"
SCALER_PATH = "scaler.pkl"
RF_MODEL    = "model/trained_model_rf.pkl"

FEATURES = [
    "speed_kmh",
    "ele_diff",
    "grade_percent",
    "delta_km",
    "delta_time_s",
    "distance_km"
]

def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    hours   = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs    = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def simulate_run(file_id: str, sleep_secs: float = 1.0):
    #Load and filter one activity
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df[df["file"] == file_id].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No data for file '{file_id}' in {DATA_PATH}")

    total_km = df["distance_km"].iloc[-1]
    print(f" Starting simulation for {file_id}:")
    print(f" Total distance: {total_km:.2f} km \n")

    #Load models and scaler
    scaler   = load(SCALER_PATH)
    rf_model = load(RF_MODEL)

    for idx, row in df.iterrows():
        # skip zero‐speed rows
        if row["speed_kmh"] <= 1e-6: #1 * 10^-6
            continue

        # build a single‐row DataFrame so scaler.transform won't warn
        X_df = pd.DataFrame([row[FEATURES]], columns=FEATURES)

        Xs      = scaler.transform(X_df)
        rf_pred = rf_model.predict(Xs)[0]

        remaining_km = total_km - row["distance_km"]
        actual       = (remaining_km / row["speed_kmh"]) *3600
        actual_hms = seconds_to_hms(actual)


        # errors
        err_rf = rf_pred - actual

        print(
            f"{idx:04d} | {row['time'].time()} | "
            f"Dist {row['distance_km']:.3f} km | "
            f"Speed {row['speed_kmh']:.3f} km/h | "
            f"Rem_Dist {remaining_km:.4f} km | "
            f"Actual ETA {actual_hms} ({actual:6.2f}s) | "
            f"RF pred {rf_pred:6.2f}s (err {err_rf:6.2f})"
        )

        time.sleep(sleep_secs)
