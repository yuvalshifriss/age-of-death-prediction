# 💀 Age of Death Predictor

This project predicts the **age at death** for heart failure patients who died during the study period, using clinical features and multiple regression models. It includes data exploration, model training, evaluation, and prediction — along with visualizations to highlight patterns associated with earlier or later mortality among the deceased patients.

---

## 📁 Project Structure
```
death-age-predictor/
├── data/
│   └── heart_failure_clinical_records.csv
│   └── train_val_data.csv
│   └── test_data.csv
├── output/
│   ├── best_model.joblib
│   ├── best_model_meta.json
│   └── *.html  (visualizations)
│   └── *.csv  (test data predictions)
├── src/
│   ├── common_utils.py
│   ├── split_data_train_test.py
│   ├── data_exploration_app.py
│   ├── run_data_exploration_app.py
│   ├── train_and_validate.py
│   └── predictor.py
├── README.md
├── requirements.txt
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
Given this, we assume that for patients with DEATH_EVENT = 1, the age column reflects their age at death. There are about 1500 such patients, on which the different models train, which is clearly not a lot.


# 🚀 Scripts Overview
## 1. `split_data_train_test.py`

Splits the data to 15% / 85% test / train data and save to csv file. 

### ▶️ How to Run

```bash
python src/split_data_train_test.py
```

## 2. `explore_data_app.py`

An interactive **Streamlit** web app for exploring the clinical data of deceased heart failure patients.

### 🔍 Features
- Summary statistics
- Condition prevalence (e.g. anaemia, diabetes, high blood pressure, smoking)
- Feature distributions (e.g. age, ejection fraction)
- Correlation heatmap of numeric features

### ▶️ How to Launch
```bash
python src/run_data_exploration_app.py
```
or
```bash
streamlit run src/explore_data_app.py
```
![data_explorer](https://github.com/user-attachments/assets/84402dbb-5dc3-49ed-b2f3-2d88596e332a)

## 3. `train_and_validate.py`

Trains multiple regression models to predict the **age at death** for heart failure patients who **died during the study period**, based on their clinical characteristics. This model is limited to the observed deaths and is intended to explore patterns associated with earlier or later mortality among these patients.

### 📋 Key Steps

- Filters only deceased patients (`DEATH_EVENT == 1`)
- Scales input features (doesn't scale target values as the regression models used don’t require target normalization)
- Trains multiple models:
  - Linear Regression, Ridge, Lasso, ElasticNet
  - Decision Trees, Random Forest, Gradient Boosting
  - SVR, XGBoost, KNN
- Evaluates each using multiple random seeds
- Ranks models based on a user-defined metric: `mse`, `mae`, or `r2`
- Saves the **best model** and its training metadata
- Predicts the **age of death** for all patients in the validation data.
- Generates an **interactive Plotly chart** with the predicted age and the actual age of death

### 📈 Output Artifacts

- Learning curve plots (see **Overfitting** bellow)
- Validation data prediction vs actual scatter plots
![valid](https://github.com/user-attachments/assets/15b2a0c9-abb5-45be-8d76-b609a146a442)
- Feature importance visualizations (if available, see **Overfitting** bellow)
- `output/best_model.joblib` and `output/best_model_meta.json`

### ▶️ How to Run

```bash
python src/train_and_validate.py
```

## 4. `predictor.py`

Uses the trained model to make predictions on **test data** that did not take part in the training stage

### 🔍 What it does:
- Loads the **trained model** and its **training metadata** (including the features used).
- Applies the **same preprocessing** used during training.
- Predicts the **age of death** for all patients in the **test data** dataset.
- Saves the **predictions to CSV**.
- Generates an **interactive Plotly chart** with the predicted age and the actual age of death
![test_prediction](https://github.com/user-attachments/assets/9229dae0-e1f2-4ed0-af92-4f2cb10ef840)

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
```
  📌 Results for Random Forest:
   Seed 0 ▶ MSE: 23.5239, MAE: 1.4986, R²: 0.8638
   Seed 42 ▶ MSE: 9.3703, MAE: 1.0943, R²: 0.9424
   Seed 77 ▶ MSE: 9.5878, MAE: 0.9732, R²: 0.9452
   Seed 123 ▶ MSE: 13.1486, MAE: 1.0241, R²: 0.9121
   Seed 999 ▶ MSE: 14.2135, MAE: 1.2069, R²: 0.9182
```
### 📉 SKlearn Learning Curve Plots
- Learning curves compare training and validation errors across increasing dataset sizes.
![learning_curve](https://github.com/user-attachments/assets/7afcef73-7c4c-4e3d-a862-2e229d485ce0)
- Here, the best model was random forest, it seems to be learning well: As training size increases, the validation MSE decreases.
- No underfitting: Training error is low, which suggests the model can represent the data well.
- Validation error plateaus: Adding more data helped up to a point (~600–800 samples), but gains diminish after that.
- **Overfitting**: The training error is much lower than the validation error — a common trait of Random Forests. The model may be too tightly fitting the training data. But the gap is stable and not extreme.

### 📊 Feature Importance Visualization
- Tree-based models (e.g. Random Forest, XGBoost) provide interpretable feature importances.
- Helps identify:
  - Over-reliance on a few features
  - Potential for dimensionality reduction or regularization
![importance](https://github.com/user-attachments/assets/3fd3dec8-cfd3-456f-a4f7-f4026a1ece6d)
