# 💀 Age of Death Predictor

This project predicts the **age of death** for patients who experienced **heart failure**, using clinical features and various regression models. It includes data exploration, training, evaluation, and prediction capabilities with visualizations.

---

## 📁 Project Structure
```
death-age-predictor/
├── data/
│   └── heart_failure_clinical_records.csv
├── output/
│   ├── best_model.joblib
│   ├── best_model_meta.json
│   └── *.html  (visualizations)
├── src/
│   ├── train_and_test.py
│   ├── predictor.py
│   └── explore_data_app.py
```

# 🧰 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

# 📊 Data Source
Input file: data/heart_failure_clinical_records.csv
Download from: [Kaggle – Heart Failure Clinical Records](https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records)
It contains clinical features of patients, with DEATH_EVENT = 1 indicating
the patient died during follow-up and age at death. The time column represents the duration (in days) until the follow-up, with a maximum value of 285 days.
Given this, we assume that for patients with DEATH_EVENT = 1, the age column reflects their age at death.


# 🚀 Scripts Overview
## 1. `explore_data_app.py`

An interactive **Streamlit** web app for exploring the clinical data of deceased heart failure patients.

### 🔍 Features
- Summary statistics
- Condition prevalence (e.g. anaemia, diabetes, high blood pressure, smoking)
- Feature distributions (e.g. age, ejection fraction)
- Correlation heatmap of numeric features

### ▶️ How to Launch
```bash
python src/train_and_test.py
```
or
```bash
streamlit run src/explore_data_app.py
```

## 2. `train_and_test.py`

Trains multiple regression models to predict the **age of death** for patients who died from heart failure.

### 📋 Key Steps

- Filters only deceased patients (`DEATH_EVENT == 1`)
- Trains multiple models:
  - Linear Regression, Ridge, Lasso, ElasticNet
  - Decision Trees, Random Forest, Gradient Boosting
  - SVR, XGBoost, KNN
- Evaluates each using multiple random seeds
- Ranks models based on a user-defined metric: `mse`, `mae`, or `r2`
- Saves the **best model** and its training metadata

### 📈 Output Artifacts

- Learning curve plots
- Prediction vs actual scatter plots
- Feature importance visualizations (if available)
- `output/best_model.joblib` and `output/best_model_meta.json`

### ▶️ How to Run

```bash
python src/train_and_test.py
```

## 3. `predictor.py`

Uses the trained model to make predictions on **all patients**, including 
those who survived (for the time being 😈) 

### 🔍 What it does:
- Loads the **trained model** and its **training metadata** (including the features used).
- Applies the **same preprocessing** used during training.
- Predicts the **age of death** for all patients in the dataset.
- Saves the **predictions to CSV**.
- Generates an **interactive Plotly chart**:
  - 🔴 **Red**: Predicted age of death (all patients)
  - 🟢 **Green**: True age (for patients who died)

### ▶️ Run it with:

```bash
python src/predictor.py
```


# 🔍 Overfitting

This project includes several strategies to detect and reduce overfitting:

### ✅ Multiple Random Seeds
- Models are trained and evaluated using several random seeds: `[0, 42, 77, 123, 999]`.
- Results (MSE, MAE, R²) are aggregated across seeds.
- Low variation between runs indicates stable model performance.

### 📉 Learning Curve Plots
- Learning curves compare training and validation errors across increasing dataset sizes.
- In this project, validation error is typically ~15% higher than training error.
- This gap suggests **moderate overfitting**, understandable given the small amount of samples ~1500, but within an acceptable range.

### 📊 Feature Importance Visualization
- Tree-based models (e.g. Random Forest, XGBoost) provide interpretable feature importances.
- Helps identify:
  - Over-reliance on a few features
  - Potential for dimensionality reduction or regularization
