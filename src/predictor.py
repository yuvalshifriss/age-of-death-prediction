import pandas as pd
import os
import joblib
import json
import plotly.graph_objs as go
from time import time
from scipy.stats import ks_2samp
from typing import Tuple, Dict, Any, List, Optional, Union
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common_utils import load_data

P_VALUE_TH = 0.05


def load_data_for_prediction() -> Tuple[pd.DataFrame, pd.Series]:
    file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'test_data.csv'))
    dead_df = load_data(file_path, True)
    true_age = dead_df['age'].where(dead_df['DEATH_EVENT'] == 1, other=None)
    X = dead_df.drop(columns=['DEATH_EVENT', 'age', 'time'])
    return X.reset_index(drop=True), true_age.reset_index(drop=True)


def load_model_and_metadata(output_dir: str) -> Tuple[RegressorMixin, Dict[str, Any]]:
    model_path = os.path.join(output_dir, 'best_model.joblib')
    meta_path = os.path.join(output_dir, 'best_model_meta.json')

    model = joblib.load(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"model_name": "Unknown", "metric": "unknown", "score": float('nan'), "features": None}

    return model, metadata


def predict(model: RegressorMixin, X: pd.DataFrame, feature_list: Optional[List[str]]) -> pd.Series:
    if feature_list:
        missing = set(feature_list) - set(X.columns)
        if missing:
            raise ValueError(f"Missing expected features: {missing}")
        X = X[feature_list]
    return model.predict(X)


def visualize_predictions(results_df: pd.DataFrame, output_dir: str) -> None:
    results_df = results_df.reset_index().rename(columns={"index": "Patient_Index"})

    # Compute metrics
    y_true = results_df["True_Age"]
    y_pred = results_df["Predicted_Age"]
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # One scatter trace: True Age on x, Predicted Age on y
    trace = go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(color="blue"),
        name="Prediction",
        customdata=results_df[["Patient_Index", "Predicted_Age", "True_Age"]].round(2),
        hovertemplate=(
            "Patient Index: %{customdata[0]}<br>"
            "True Age: %{customdata[2]}<br>"
            "Predicted Age: %{customdata[1]}<extra></extra>"
        )
    )

    min_age = min(y_true.min(), y_pred.min())
    max_age = max(y_true.max(), y_pred.max())
    ideal_line = go.Scatter(
        x=[min_age, max_age],
        y=[min_age, max_age],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Ideal (y = x)"
    )

    # Add metrics to the title
    fig = go.Figure([trace, ideal_line])
    fig.update_layout(
        title=f"Test Data Predicted vs Actual Age of Death<br>"
              f"MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}",
        xaxis_title="True Age",
        yaxis_title="Predicted Age",
        width=900,
        height=600
    )

    fig.show()
    plot_path = os.path.join(output_dir, "test_data_predicted_vs_true_by_index.html")
    fig.write_html(str(plot_path))
    print(f"📊 Plot saved to: {plot_path}")


def compare_distributions(y_true: Union[pd.Series, List[float]], y_pred: Union[pd.Series, List[float]]) -> None:
    statistic, p_value = ks_2samp(y_true, y_pred)
    print(f"KS Statistic: {statistic}, p-value: {p_value}")
    if p_value > P_VALUE_TH:
        print("There’s no statistically significant difference between the distributions "
              "of the predictions and actual target values (ages).")
    else:
        print("The predicted age-at-death distribution differs significantly from the true one, suggesting the model "
              "may be biased in some age ranges (e.g., underestimating younger ages at death or overestimating older ones).")


def main() -> None:
    output_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'output'))
    os.makedirs(output_dir, exist_ok=True)

    X, y_true = load_data_for_prediction()
    model, metadata = load_model_and_metadata(output_dir)
    features = metadata.get("features")

    print(f"✅ Loaded model: {metadata.get('model_name', 'Unknown')} "
          f"(optimized for {metadata.get('metric', 'unknown').upper()}, "
          f"score = {metadata.get('score', float('nan')):.4f})")

    start_time = time()
    y_pred = predict(model, X, features)
    total_time = time() - start_time
    avg_time_per_patient = total_time / len(X)

    print(f"⏱️ Inference time: {total_time:.4f} sec total, {avg_time_per_patient:.6f} sec/patient")

    results_df = X.copy()
    results_df["True_Age"] = y_true
    results_df["Predicted_Age"] = y_pred

    save_path = os.path.join(output_dir, "test_data_predicted_ages.csv")
    results_df.to_csv(save_path, index=False)
    print(f"📄 Predictions saved to: {save_path}")
    visualize_predictions(results_df, output_dir)
    compare_distributions(y_true, y_pred)


if __name__ == '__main__':
    main()
