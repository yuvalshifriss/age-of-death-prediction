import pandas as pd
import os
import joblib
import json
import plotly.graph_objs as go
from time import time


def load_data_for_prediction():
    file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'heart_failure_clinical_records.csv'))
    df = pd.read_csv(file_path)

    X = df.drop(columns=['DEATH_EVENT', 'time'], errors='ignore')
    death_event = df['DEATH_EVENT']
    true_age = df['age'].where(death_event == 1, other=None)

    return X.reset_index(drop=True), true_age.reset_index(drop=True), death_event.reset_index(drop=True)


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


def visualize_predictions(results_df, output_dir):
    results_df = results_df.reset_index().rename(columns={"index": "Patient_Index"})

    # Predicted Age (red)
    pred_trace = go.Scatter(
        x=results_df["Patient_Index"],
        y=results_df["Predicted_Age"],
        mode="markers",
        marker=dict(color="red"),
        name="Predicted Age of Death",
        customdata=results_df[["Predicted_Age", "True_Age"]].round(2).fillna("N/A"),
        hovertemplate=(
            "Patient Index: %{x}<br>"
            "Predicted Age: %{customdata[0]}<br>"
            "True Age: %{customdata[1]}<extra></extra>"
        )
    )

    # True Age (green)
    deceased_df = results_df[results_df["True_Age"].notna()]
    true_trace = go.Scatter(
        x=deceased_df["Patient_Index"],
        y=deceased_df["True_Age"],
        mode="markers",
        marker=dict(color="green"),
        name="True Age of Death (Deceased)",
        customdata=deceased_df[["True_Age", "Predicted_Age"]].round(2),
        hovertemplate=(
            "Patient Index: %{x}<br>"
            "True Age: %{customdata[0]}<br>"
            "Predicted Age: %{customdata[1]}<extra></extra>"
        )
    )

    fig = go.Figure([pred_trace, true_trace])
    fig.update_layout(
        title="Predicted vs True Age of Death by Patient Index",
        xaxis_title="Patient Index",
        yaxis_title="Age",
        width=900,
        height=600
    )
    fig.show()
    plot_path = os.path.join(output_dir, "predicted_vs_true_by_index.html")
    fig.write_html(str(plot_path))
    print(f"📊 Plot saved to: {plot_path}")


def main():
    output_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'output'))
    os.makedirs(output_dir, exist_ok=True)

    X, y_true, death_event = load_data_for_prediction()
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
    results_df["Death_Event"] = death_event

    save_path = os.path.join(output_dir, "predicted_ages.csv")
    results_df.to_csv(save_path, index=False)
    print(f"📄 Predictions saved to: {save_path}")

    visualize_predictions(results_df, output_dir)


if __name__ == '__main__':
    main()
