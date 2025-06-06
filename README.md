# 💀 Age of Death Predictor

This project predicts the **age at death** for heart failure patients who **died during the study period** (i.e., it does not predict **whether** a patient will die or when death might occur, but rather estimates the age at death for those already known to have died), using clinical features (such as ```smoking``` and ```serum_creatinine```, some continuous and others binary) and multiple regression models (Random Forest, XGBoost,  ElasticNet). It includes data exploration, train (85%) / test (15%) splitting, model training, evaluation via cross-validation, and prediction on unseen test data.

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
data/heart_failure_clinical_records.csv downloaded from: [Kaggle – Heart Failure Clinical Records](https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records)
The dataset contains clinical data from heart failure patients, with the time column indicating the duration of follow-up in days. Patients with ```DEATH_EVENT``` = 1 are those who died during this follow-up period, and for them, the ```age``` column is interpreted as their age at death. The models are trained exclusively on this subset of deceased patients.

# 🚀 Scripts Overview
## 1. `split_data_train_test.py`

Splits the data into training (85%) and test (15%) sets, and saves them as CSV files.

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
  - Random Forest
  - XGBoost
  - ElasticNet
- Evaluates each using Repeated K-Fold Cross-Validation
- Ranks models based on a user-defined metric: `mse`, `mae`, or `r2`
- Saves the **best model** and its training metadata
- Predicts the **age at death** for patients in the **validation dataset**, which consists only of individuals who died during the study period.
- Generates an **interactive Plotly chart** with the predicted age and the actual age of death

### 📈 Output Artifacts

- Learning curve plots (see **Discussion** below)
- Validation data prediction vs actual scatter plots
![valid](https://github.com/user-attachments/assets/15b2a0c9-abb5-45be-8d76-b609a146a442)
- Feature importance visualizations (if available, see **Discussion** bellow)
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

# 🔍 Discussion

This project includes multiple strategies to evaluate model performance and stability:

### ✅ Repeated K-Fold Cross-Validation
- Each model is evaluated using repeated k-fold splits (e.g. 5 folds × 3 repeats = 15 folds total)
- Each fold provides a separate estimate of performance (MSE, MAE, R²)
- This reduces bias from a single train/test split and better captures variance in performance

```
📌 Results for Random Forest:
 Fold  1 ▶ MSE: 9.1715, MAE: 0.9503, R²: 0.9444
 Fold  2 ▶ MSE: 30.5263, MAE: 1.8680, R²: 0.8150
 Fold  3 ▶ MSE: 16.0632, MAE: 1.3539, R²: 0.9172
 ...
 Fold 15 ▶ MSE: 21.2127, MAE: 1.5120, R²: 0.8781
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
