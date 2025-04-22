from parser import load_all_gpx
from metrics import compute_metrics
from utils import export_to_csv
from eda_running import perform_eda
from training import train_and_evaluate
from predict import predict_finish_time

if __name__ == "__main__":
    combined_df = load_all_gpx("data/")
    combined_df = compute_metrics(combined_df)

    train_and_evaluate(combined_df)
    combined_df = predict_finish_time(combined_df)

    export_to_csv(combined_df, "output/combined_data_with_predictions.csv")
    perform_eda(combined_df)
    