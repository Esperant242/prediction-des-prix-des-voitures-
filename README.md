# Car Price Prediction Project

## 1. Introduction
This project aims to predict car prices using a machine learning pipeline applied to a dataset containing detailed information about cars (brand, engine volume, mileage, etc.). The project follows a complete data science approach including data cleaning, preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## 2. Data Exploration and Cleaning
- **Missing Values**: We explored and imputed missing values, especially for the 'Levy' column using the median.
- **Duplicates**: Removed duplicate entries to ensure data quality.

## 3. Feature Engineering
- **Type Conversion**: Converted categorical variables to string type for proper encoding.
- **Feature Extraction**: Extracted 'Turbo' from the 'Engine volume' and cleaned various features.
- **Imputation**: Categorical variables with missing values were imputed using the mode.

## 4. Preprocessing
### 4.1 Remove Useless Columns
Dropped irrelevant columns: `Model`, `ID`.

### 4.2 Outliers Treatment
Applied Tukey's method to remove outliers from continuous variables: `Levy`, `Mileage`, `Price`, `Prod. year`, and `Engine volume`. Additionally, removed cars with price below €500 as these are likely errors.

### 4.3 Standardize Variables
Standardized continuous variables: `Levy`, `Mileage`, `Prod. year`, `Engine volume`, `Cylinders` using `StandardScaler`.

## 5. Modeling
### 5.1 Detect Multicollinearity
Used VIF (Variance Inflation Factor) to detect multicollinearity in continuous variables. All VIF values were acceptable, so no variables were dropped.

### 5.2 Encoding and Dataset Splitting
Applied one-hot encoding to categorical variables. Split data into training and testing sets (80/20 split).

### 5.3 Model Comparison
Trained and evaluated four models:
- Linear Regression: RMSE = 8049, R2 = 44.2%
- Random Forest: RMSE = 5429, R2 = 74.6%
- XGBoost: RMSE = 5007, R2 = 78.4%
- Decision Tree: RMSE = 6134, R2 = 67.6%

**XGBoost** performed the best in terms of RMSE and R2.

### 5.4 Hyperparameter Tuning
Used GridSearchCV to optimize XGBoost hyperparameters:
- Best parameters: `n_estimators=700`, `max_depth=9`, `learning_rate=0.05`
- Final model RMSE: 4813, R2: 80.1%

### Feature Importance
Top influential features:
- `Fuel type_Diesel`
- `Gear box type_Tiptronic`
- `Manufacturer_SSANGYONG`
- Continuous: `Prod. year`, `Cylinders`

## 6. Conclusion
- XGBoost achieved the best performance with R2 = 80.1%, making it suitable for car price prediction.
- Results can be improved by enhancing the dataset and hardware capabilities.
- Missing values in 'Levy' and limited computational resources slightly constrained model performance.

## Tools and Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Seaborn, Matplotlib

---
