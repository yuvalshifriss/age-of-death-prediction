import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from time import time
from typing import Tuple, Dict, List, Optional, Any

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from common_utils import load_data

# Config
SEEDS = [0, 42, 77, 123, 999]
DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'train_val_data.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_split_data() -> Tuple[pd.DataFrame, pd.Series]:
    dead_df = load_data(DATA_PATH, True)
    X = dead_df.drop(columns=['DEATH_EVENT', 'age', 'time'])
    y = dead_df['age']
    return X, y


def define_models() -> Dict[str, RegressorMixin]:
    return {
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }


def evaluate_model(model: RegressorMixin, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.DataFrame,
                   y_valid: pd.DataFrame) -> Dict[str, Any]:
    pipeline = make_pipeline(StandardScaler(), model)
    start = time()
    pipeline.fit(X_train, y_train)  # scaling / normalization is done inside
    duration = time() - start
    y_pred = pipeline.predict(X_valid)
    return {
        "pipeline": pipeline,
        "r2": r2_score(y_valid, y_pred),
        "mse": mean_squared_error(y_valid, y_pred),
        "mae": mean_absolute_error(y_valid, y_pred),
        "y_pred": y_pred,
        "time": duration
    }


def plot_learning_curve(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str, output_dir: str):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training MSE'))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Validation MSE'))
    fig.update_layout(title=f'SKlearn Learning Curve – {model_name}',
                      xaxis_title='Training examples', yaxis_title='MSE')
    fig.show()
    path = os.path.join(output_dir, f'sklearn_learning_curve_{model_name.replace(" ", "_")}.html')
    fig.write_html(path)
    return path


def plot_predictions(results_sorted: List[Dict[str, Any]], y_valid: pd.Series, metric_to_optimize: str, output_dir: str):
    traces = []
    for res in results_sorted:
        traces.append(go.Scatter(
            x=y_valid,
            y=res['y_pred'],
            mode='markers',
            name=f"{res['name']} (R²={res['r2']:.2f})",
            hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))

    traces.append(go.Scatter(
        x=y_valid,
        y=y_valid,
        mode='lines',
        name='Ideal (y = x)',
        line=dict(color='red', dash='dash')
    ))

    fig = go.Figure(data=traces, layout=go.Layout(
        title=f'Validation Data Predicted vs Actual (All Models) – Sorted by {metric_to_optimize.upper()}',
        xaxis=dict(title='Actual Age'),
        yaxis=dict(title='Predicted Age'),
        width=900, height=700
    ))

    fig.show()
    plot_path = os.path.join(output_dir, 'validation_data_predicted_vs_actual_all_models.html')
    fig.write_html(plot_path)
    return plot_path


def plot_feature_importance(best_result: Dict[str, Any], feature_names: List[str], best_name: str, output_dir: str) \
        -> Optional[str]:
    model_name = best_result['name']
    model = best_result['pipeline'].named_steps[list(best_result['pipeline'].named_steps)[-1]]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        print("\n📌 Feature Importances:")
        sorted_idx = np.argsort(importances)[::-1]
        for i in sorted_idx:
            print(f"{feature_names[i]}: {importances[i]:.4f}")

        fig_imp = go.Figure(go.Bar(
            x=importances,
            y=feature_names,
            orientation='h',
            marker=dict(color=importances, colorscale='Viridis')
        ))
        fig_imp.update_layout(
            title=f"Feature Importances – {model_name}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600, width=800
        )
        fig_imp.show()
        path = os.path.join(output_dir, f'feature_importances_{best_name}.html')
        fig_imp.write_html(path)
        return path
    return None


def run_full_pipeline(metric_to_optimize: str = "mse") -> None:
    assert metric_to_optimize in ["mae", "mse", "r2"]
    print("📥 Loading data...")
    X, y = load_and_split_data()
    feature_names = X.columns.tolist()

    print("🔧 Defining models...")
    models = define_models()
    seed_results = {}

    print("🚀 Running evaluations grouped by model...")
    for name, model in models.items():
        print(f"\n📌 Results for {name}:")
        seed_results[name] = []
        for i, seed in enumerate(SEEDS):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=seed)
            res = evaluate_model(model, X_train, X_valid, y_train, y_valid)
            res["name"] = name
            res["y_valid"] = y_valid
            seed_results[name].append(res)
            print(f"   Seed {seed} ▶ MSE: {res['mse']:.4f}, MAE: {res['mae']:.4f}, R²: {res['r2']:.4f}")

    print("\n📊 Aggregating average results across seeds...")
    summary = []
    for name, runs in seed_results.items():
        metric_vals = [r[metric_to_optimize] for r in runs]
        avg_metric = np.mean(metric_vals)
        std_metric = np.std(metric_vals)
        best_run = min(runs, key=lambda r: r[metric_to_optimize])
        summary.append({
            "name": name,
            "avg_metric": avg_metric,
            "std_metric": std_metric,
            "result": best_run
        })

    results_sorted = sorted(summary, key=lambda x: x["avg_metric"])
    best_result = results_sorted[0]["result"]
    best_name = results_sorted[0]["name"]

    print(f"\n🎯 Best model by {metric_to_optimize.upper()}: {best_name} ({results_sorted[0]['avg_metric']:.4f} {metric_to_optimize.upper()})")

    joblib.dump(best_result['pipeline'], os.path.join(OUTPUT_DIR, 'best_model.joblib'))
    with open(os.path.join(OUTPUT_DIR, 'best_model_meta.json'), 'w') as f:
        json.dump({
            "model_name": best_name,
            "metric": metric_to_optimize,
            "score": results_sorted[0]["avg_metric"],
            "features": feature_names
        }, f, indent=2)

    print(f"💾 Model + metadata saved to: {OUTPUT_DIR}")

    _ = plot_predictions([r["result"] for r in results_sorted], best_result['y_valid'], metric_to_optimize, OUTPUT_DIR)
    _ = plot_learning_curve(best_result['pipeline'], X, y, best_name, OUTPUT_DIR)
    _ = plot_feature_importance(best_result, feature_names, best_name, OUTPUT_DIR)


if __name__ == '__main__':
    # 🎯 Goal: predict age of death of patients who had heart failure
    run_full_pipeline(metric_to_optimize="mse")  # or "mae" or "r2"
