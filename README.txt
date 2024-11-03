#AeroDelay: Intelligent Flight Delay Prediction
Problem
Airline delays can significantly impact passenger satisfaction and operational efficiency. Predicting flight delays in advance helps airlines to optimize scheduling, improve customer service, and manage resources effectively.

Solution
This project implements a machine learning model to predict whether a flight will be delayed by more than 15 minutes based on various flight-related features. The solution leverages multiple classification algorithms to achieve accurate predictions.

Dataset
Source: The dataset used for this project is from the Kaggle Bank Churn Dataset, 2024, and the Flight Delays dataset, 2018.
The dataset contains historical flight information, including features such as departure time, arrival time, airline, flight distance, and weather conditions.
Model
The project employs three different models:

Logistic Regression

Hyperparameters:
max_iter: 1000
random_state: 42
Random Forest Classifier

Hyperparameter Tuning:
n_estimators: [100, 200]
max_depth: [None, 10, 20]
min_samples_split: [2, 5]
class_weight: ['balanced', None]
XGBoost Classifier

Hyperparameter Tuning:
n_estimators: [100, 200]
max_depth: [3, 5]
learning_rate: [0.01, 0.1]
subsample: [0.6, 0.8]
colsample_bytree: [0.6, 0.8]
gamma: [0, 0.1]
scale_pos_weight: [1]
Evaluation Score
The models were evaluated based on ROC-AUC scores, with the following results:

Logistic Regression: 0.91
Random Forest: 0.86
XGBoost: 0.92 (Best Model)
Citation
Walter Reade and Ashley Chow. Binary Classification with a Bank Churn Dataset. Kaggle, 2024.
Yury Kashnitsky. mlcourse.ai: Flight delays. Kaggle, 2018.
