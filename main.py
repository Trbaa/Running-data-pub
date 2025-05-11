from simulate_run import simulate_run
import pandas as pd

if __name__ == "__main__":
   # pick first file automatically, or replace with specific filename
    all_files = pd.read_csv("output/combined_data_with_predictions.csv")["file"].unique()
    simulate_run(all_files[0], sleep_secs=1.0)