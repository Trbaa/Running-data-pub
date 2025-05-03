from parser import load_all_gpx
from metrics import compute_metrics
from utils import export_to_csv
from eda_running import perform_eda
from training import train_and_evaluate
from predict import predict_finish_time
from graphs import make_graph
import os

if __name__ == "__main__":


    REQUIRED_FOLDERS = ["model", "diagrams", "predictions", "data"]

    for folder in REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)

    combined_df = load_all_gpx("data/")
    combined_df = compute_metrics(combined_df)

    train_and_evaluate(combined_df)
    combined_df = predict_finish_time(combined_df)

    export_to_csv(combined_df, "output/combined_data_with_predictions.csv")
    perform_eda(combined_df)
    
    make_graph()