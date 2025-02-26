# Speed Prediction for Traffic Light Scheduling

This project uses machine learning techniques to predict the speed of cars based on data collected from a fleet of taxi vehicles. The goal is to enhance traffic light scheduling systems by predicting car speeds in real-time, allowing for better optimization and improved traffic flow.

## Overview

This repository contains a Python script that:
- Loads and processes taxi fleet data from CSV files.
- Performs data preprocessing, including date-time conversion, feature engineering, and label encoding.
- Uses the LightGBM (LGBMRegressor) model to predict car speed.
- Evaluates the model's performance and visualizes key aspects of the model using various interpretability tools like SHAP and Partial Dependence Plots.
- Provides insights into feature importance and model performance using permutation importance and feature correlation heatmaps.

## Features

- **Data Preprocessing**: Handles missing or invalid datetime formats, adds time-based features (Month, Day, Hour, Minute), and drops irrelevant rows.
- **Machine Learning Model**: Uses the LGBMRegressor model for speed prediction.
- **Model Evaluation**: Evaluates the model using metrics such as Mean Squared Error (MSE) and R² score.
- **Model Interpretability**: Uses SHAP, partial dependence plots, and permutation importance to explain model predictions.

## Requirements

Before running the code, ensure you have the following dependencies installed:

- `pandas`
- `matplotlib`
- `seaborn`
- `lightgbm`
- `scikit-learn`
- `eli5`
- `shap`

You can install the required packages by running the following command:

```bash
pip install -r requirements.txt