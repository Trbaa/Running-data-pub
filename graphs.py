import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_graph():

    df = pd.read_csv("output/combined_data_with_predictions.csv", parse_dates=["time"])

    mae_lr_list, rmse_lr_list = [], []
    mae_rf_list, rmse_rf_list = [], []
    activity_ids = []

    # Grupisanje po aktivnosti
    for activity_id, group in df.groupby("file"):
        y_true = group["time_to_finish"]
        y_lr = group["lr_predicted_time"]
        y_rf = group["rf_predicted_time"]

        mae_lr = mean_absolute_error(y_true, y_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_true, y_lr))
        mae_rf = mean_absolute_error(y_true, y_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_true, y_rf))

        print(f"\nEvaluacija za {activity_id}:")
        print(f"  LR MAE: {mae_lr:.3f}  RMSE: {rmse_lr:.3f}")
        print(f"  RF MAE: {mae_rf:.3f}  RMSE: {rmse_rf:.3f}")

        mae_lr_list.append(mae_lr)
        rmse_lr_list.append(rmse_lr)
        mae_rf_list.append(mae_rf)
        rmse_rf_list.append(rmse_rf)
        activity_ids.append(activity_id)

    # Prosečne vrednosti
    print("\n--- Proseci ---")
    print(f"Linear Regression:\n  MAE: {np.mean(mae_lr_list):.3f}  RMSE: {np.mean(rmse_lr_list):.3f}")
    print(f"Random Forest:\n  MAE: {np.mean(mae_rf_list):.3f}  RMSE: {np.mean(rmse_rf_list):.3f}")

    x = np.arange(len(activity_ids))
    width = 0.35

    # RMSE
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, rmse_lr_list, width, label="LR RMSE", color="cornflowerblue")
    plt.bar(x + width/2, rmse_rf_list, width, label="RF RMSE", color="darkorange")
    plt.xticks(x, activity_ids, rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title("RMSE po aktivnosti")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("diagrams/rmse_comparison.png")
    plt.show()

    # MAE
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, mae_lr_list, width, label="LR MAE", color="lightgreen")
    plt.bar(x + width/2, mae_rf_list, width, label="RF MAE", color="tomato")
    plt.xticks(x, activity_ids, rotation=45, ha="right")
    plt.ylabel("MAE")
    plt.title("MAE po aktivnosti")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("diagrams/mae_comparison.png")
    plt.show()
