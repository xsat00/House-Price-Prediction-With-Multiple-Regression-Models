# 🏠 Multi-Model House Price Prediction System

# Project DIscription
This project predicts house prices using multiple Machine Learning regression algorithms and automatically identifies the best-performing model based on evaluation metrics.

## Machine Learning Workflow

1. Data Cleaning
2. Missing Value Treatment
3. Feature Engineering
4. Exploratory Data Analysis
5. Feature Scaling
6. Model Training
7. Hyperparameter Tuning
8. Multi-Model Evaluation
9. Best Model Selection
10. Price Prediction

## Machine Learning Models

The following regression algorithms were trained and compared:

1. Linear Regression
2. ANN
3. Random Forest Regressor
4. Huber  Regressor
5. XGBoost Regressor
6. KNN Regressor
7. Lasso Regressor
8. LGBM
9. Polynomial Regression
10. Ridge Regression
11. SGD Regressor
12. SVR

The best model was selected based on evaluation metrics.

## Key Findings

A total of 13 regression models were trained and evaluated.

Surprisingly, simple linear models outperformed more complex ensemble and neural network approaches on this dataset.

Best Performing Model:

🥇 Lasso Regression

R² Score: 0.9146

MAE: 82,657

MSE: 10.54 Billion

The results suggest that the relationship between house features and target prices in this dataset is largely linear, making regularized linear models highly effective.

# Performance Table

## Model Performance Comparison

| Model | MAE | R² Score |
|---------|---------|---------|
| Lasso Regression | 82,657 | 0.9146 |
| Linear Regression | 82,657 | 0.9146 |
| Ridge Regression | 82,659 | 0.9146 |
| Polynomial Regression | 83,413 | 0.9137 |
| LGBM | 92,133 | 0.8940 |
| Random Forest | 97,925 | 0.8781 |
| ElasticNet | 99,126 | 0.8780 |
| XGBoost | 101,565 | 0.8694 |
| KNN | 198,086 | 0.5113 |
| ANN | 199,284 | 0.5020 |
| Huber Regressor | 199,465 | 0.5011 |
| SVR | 282,947 | 0.0004 |
| SGDRegressor | Failed to Converge |

## Conclusion

This project demonstrates a complete end-to-end machine learning workflow for house price prediction.

Thirteen different regression algorithms were trained and evaluated using MAE, MSE, and R² metrics.

Although advanced ensemble methods such as Random Forest, LightGBM, and XGBoost were tested, regularized linear models achieved the highest predictive performance on the dataset.

Lasso Regression was selected as the final model with an R² score of 91.46%.

