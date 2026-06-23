# 🏠 Multi-Model House Price Prediction System

## 📌 Project Overview

This project predicts house prices using multiple Machine Learning regression algorithms and automatically identifies the best-performing model based on evaluation metrics.

The project follows a complete end-to-end Machine Learning workflow, from data preprocessing and feature engineering to model evaluation and prediction.

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

## 🔄 Machine Learning Workflow

1. Data Cleaning
2. Missing Value Handling
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Scaling
6. Model Training
7. Hyperparameter Tuning
8. Multi-Model Evaluation
9. Best Model Selection
10. House Price Prediction

---

## 🤖 Models Evaluated

The following regression models were trained and compared:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Random Forest Regressor
7. XGBoost Regressor
8. LightGBM Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor (KNN)
11. Huber Regressor
12. SGD Regressor
13. Artificial Neural Network (ANN)

---

## 📊 Key Findings

A total of 13 regression models were trained and evaluated.

Despite testing advanced ensemble and neural network models, regularized linear models achieved the best overall performance on this dataset.

### 🏆 Best Performing Model: Lasso Regression

| Metric | Value |
|----------|----------|
| R² Score | 0.9146 |
| MAE | 82,657 |
| MSE | 10.54 Billion |

These results suggest that the relationship between house features and target prices is largely linear, making regularized linear models highly effective.

---

## 📈 Model Performance Comparison

| Model | MAE | R² Score |
|---------|---------|---------|
| Lasso Regression | 82,657 | 0.9146 |
| Linear Regression | 82,657 | 0.9146 |
| Ridge Regression | 82,659 | 0.9146 |
| Polynomial Regression | 83,413 | 0.9137 |
| LightGBM | 92,133 | 0.8940 |
| Random Forest | 97,925 | 0.8781 |
| ElasticNet | 99,126 | 0.8780 |
| XGBoost | 101,565 | 0.8694 |
| KNN | 198,086 | 0.5113 |
| ANN | 199,284 | 0.5020 |
| Huber Regressor | 199,465 | 0.5011 |
| SVR | 282,947 | 0.0004 |
| SGD Regressor | Failed to Converge |

---

## 📷 Visual Results

### Model Leaderboard

![Model Leaderboard](results/model_leaderboard.png)

### R² Score Comparison

![R2 Comparison](results/r2_comparison.png)

### MAE Comparison

![MAE Comparison](results/mae_comparison.png)

### Actual vs Predicted (Lasso Regression)

![Actual vs Predicted](results/actual_vs_predicted_lasso.png)

---

## 💼 Business Use Case

This project can assist:

- Real Estate Companies
- Property Buyers
- Property Sellers
- Housing Market Analysts
- Investment Firms

by providing data-driven property price estimates.

---

## ⚙️ Installation

```bash
git clone <repository-url>
cd house-price-prediction
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
python train.py
```

or

```bash
python streamlit run STapp.py
```

---

## 🎯 Conclusion

This project demonstrates a complete end-to-end Machine Learning pipeline for house price prediction.

A total of 13 regression algorithms were trained and evaluated using MAE, MSE, and R² metrics. While advanced models such as Random Forest, LightGBM, XGBoost, and ANN were explored, Lasso Regression achieved the best predictive performance with an R² score of 91.46%.

The project highlights the importance of benchmarking multiple algorithms rather than assuming that more complex models will always perform better.

---

## 👨‍💻 Author

Sathwik

B.Tech Computer Science Engineering

Aspiring AI / ML Engineer
