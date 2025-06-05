import pandas as pd
import os
import joblib
import json
import plotly.graph_objs as go
from time import time

from common_utils import load_data


def load_data_for_prediction():
    file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'test_data.csv'))
    dead_df = load_data(file_path, True)
    true_age = dead_df['age'].where(dead_df['DEATH_EVENT'] == 1, other=None)
    X = dead_df.drop(columns=['DEATH_EVENT', 'age', 'time'])
    return X.reset_index(drop=True), true_age.reset_index(drop=True)


def load_model_and_metadata(output_dir):
    model_path = os.path.join(output_dir, 'best_model.joblib')
    meta_path = os.path.join(output_dir, 'best_model_meta.json')

    model = joblib.load(model_path)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"model_name": "Unknown", "metric": "unknown", "score": float('nan'), "features": None}

    return model, metadata


def predict(model, X, feature_list):
    if feature_list:
        missing = set(feature_list) - set(X.columns)
        if missing:
            raise ValueError(f"Missing expected features: {missing}")
        X = X[feature_list]
    return model.predict(X)


def visualize_predictions(results_df: pd.DataFrame, output_dir: str) -> None:
    results_df = results_df.reset_index().rename(columns={"index": "Patient_Index"})
    # One scatter trace: True Age on x, Predicted Age on y
    trace = go.Scatter(
        x=results_df["True_Age"],
        y=results_df["Predicted_Age"],
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
    # Ideal line (y = x)
    min_age = results_df[["True_Age", "Predicted_Age"]].min().min()
    max_age = results_df[["True_Age", "Predicted_Age"]].max().max()
    ideal_line = go.Scatter(
        x=[min_age, max_age],
        y=[min_age, max_age],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Ideal (y = x)"
    )
    fig = go.Figure([trace, ideal_line])
    fig.update_layout(
        title="Test Data Predicted vs Actual Age of Death",
        xaxis_title="True Age",
        yaxis_title="Predicted Age",
        width=900,
        height=600
    )
    fig.show()
    plot_path = os.path.join(output_dir, "test_data_predicted_vs_true_by_index.html")
    fig.write_html(str(plot_path))
    print(f"📊 Plot saved to: {plot_path}")


def main():
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


if __name__ == '__main__':
    main()
