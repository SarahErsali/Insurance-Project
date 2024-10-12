import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error)
import shap
from components.data import X_train, y_train, X_val, y_val, X_blind_test, y_blind_test, property_data_model
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt
import os





 #xgb_model = XGBRegressor()
#xgb_model.fit(X_train, y_train)

# # Get feature importance
# feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_model.feature_importances_})
# feature_importance = feature_importance.sort_values(by='Importance', ascending=False)


# # xgb_model = XGBRegressor()
# # xgb_model.fit(X_train, y_train)

# xgb_val_preds = xgb_model.predict(X_val)

# xgb_mse = mean_squared_error(y_val, xgb_val_preds)
# xgb_rmse = np.sqrt(xgb_mse)
# xgb_mae = mean_absolute_error(y_val, xgb_val_preds)
# xgb_r2 = r2_score(y_val, xgb_val_preds)
# xgb_mape = mean_absolute_percentage_error(y_val, xgb_val_preds)
# xgb_bias = np.mean(xgb_val_preds - y_val)
# xgb_accuracy = 100 - (xgb_mape * 100)



# lgb_model = LGBMRegressor()
# lgb_model.fit(X_train, y_train)

# # Get feature importance
# feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': lgb_model.feature_importances_})
# feature_importance = feature_importance.sort_values(by='Importance', ascending=False)



# lgb_val_preds = lgb_model.predict(X_val)

# lgb_mse = mean_squared_error(y_val, lgb_val_preds)
# lgb_rmse = np.sqrt(lgb_mse)
# lgb_r2 = r2_score(y_val, lgb_val_preds)
# lgb_mae = mean_absolute_error(y_val, lgb_val_preds)
# lgb_mape = mean_absolute_percentage_error(y_val, lgb_val_preds)
# lgb_bias = np.mean(lgb_val_preds - y_val)
# lgb_accuracy = 100 - (lgb_mape * 100)


#xgb_blind_test_preds = xgb_model.predict(X_blind_test)

# xgb_blind_test_mse = mean_squared_error(y_blind_test, xgb_blind_test_preds)
# xgb_blind_test_rmse = np.sqrt(xgb_blind_test_mse)
# xgb_blind_test_mae = mean_absolute_error(y_blind_test, xgb_blind_test_preds)
# xgb_blind_test_mape = mean_absolute_percentage_error(y_blind_test, xgb_blind_test_preds) * 100  # MAPE in percentage
# xgb_blind_test_r2 = r2_score(y_blind_test, xgb_blind_test_preds)
# xgb_blind_test_bias = np.mean(xgb_blind_test_preds - y_blind_test)
# xgb_blind_test_accuracy = 100 - xgb_blind_test_mape




# lgb_blind_test_preds = lgb_model.predict(X_blind_test)

# lgb_blind_test_mse = mean_squared_error(y_blind_test, lgb_blind_test_preds)
# lgb_blind_test_rmse = np.sqrt(lgb_blind_test_mse)
# lgb_blind_test_mae = mean_absolute_error(y_blind_test, lgb_blind_test_preds)
# lgb_blind_test_mape = mean_absolute_percentage_error(y_blind_test, lgb_blind_test_preds) * 100  # MAPE in percentage
# lgb_blind_test_r2 = r2_score(y_blind_test, lgb_blind_test_preds)
# lgb_blind_test_bias = np.mean(lgb_blind_test_preds - y_blind_test)
# lgb_blind_test_accuracy = 100 - lgb_blind_test_mape



# plt.plot(y_blind_test.values, label='Actual Values (Blind Test)', marker='o', linestyle='-', alpha=0.6)
# plt.plot(xgb_blind_test_preds, label='XGBoost Predictions (Blind Test)', marker='x', linestyle='--', alpha=0.6)
# plt.plot(lgb_blind_test_preds, label='LightGBM Predictions (Blind Test)', marker='s', linestyle='--', alpha=0.6)

# plt.title('Actual vs model predictions on validation and blind test sets')
# plt.xlabel('Observations')
# plt.ylabel('Claims Incurred')
# plt.legend()

# plt.tight_layout()



# Combine Training and Validation Sets
X_combined = pd.concat([X_train, X_val], ignore_index=True)
y_combined = pd.concat([y_train, y_val], ignore_index=True)

# XGBoost best parameters
xgb_best_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'gamma': 0,
}

# # Retrain the XGBoost model with best hyperparameters
re_xgb_model = XGBRegressor(**xgb_best_params)
re_xgb_model.fit(X_combined, y_combined)

# LightGBM best parameters
lgb_best_params = {
    'n_estimators': 200,
    'max_depth': 5,  
    'learning_rate': 0.1,
    'num_leaves': 10
}

# # Retrain the XGBoost model with best hyperparameters
re_lgb_model = LGBMRegressor(**lgb_best_params)
re_lgb_model.fit(X_combined, y_combined)

# # Predict on the blind test set
# xgb_blind_test_preds = re_xgb_model.predict(X_blind_test)
# lgb_blind_test_preds = re_lgb_model.predict(X_blind_test)

# # Calculate metrics for XGBoost
# re_xgb_blind_test_mse = mean_squared_error(y_blind_test, xgb_blind_test_preds)
# re_xgb_blind_test_rmse = np.sqrt(mean_squared_error(y_blind_test, xgb_blind_test_preds))
# re_xgb_blind_test_mae = mean_absolute_error(y_blind_test, xgb_blind_test_preds)
# re_xgb_blind_test_mape = mean_absolute_percentage_error(y_blind_test, xgb_blind_test_preds) * 100
# re_xgb_blind_test_r2 = r2_score(y_blind_test, xgb_blind_test_preds)
# re_xgb_blind_test_bias = np.mean(xgb_blind_test_preds - y_blind_test)  
# re_xgb_blind_test_accuracy = 100 - re_xgb_blind_test_mape

# # Calculate metrics for LightGBM
# re_lgb_blind_test_mse = mean_squared_error(y_blind_test, lgb_blind_test_preds)
# re_lgb_blind_test_rmse = np.sqrt(mean_squared_error(y_blind_test, lgb_blind_test_preds))
# re_lgb_blind_test_mae = mean_absolute_error(y_blind_test, lgb_blind_test_preds)
# re_lgb_blind_test_mape = mean_absolute_percentage_error(y_blind_test, lgb_blind_test_preds) * 100
# re_lgb_blind_test_r2 = r2_score(y_blind_test, lgb_blind_test_preds)
# re_lgb_blind_test_bias = np.mean(lgb_blind_test_preds - y_blind_test)  
# re_lgb_blind_test_accuracy = 100 - re_lgb_blind_test_mape


# # Initialize the SHAP explainer for retrained  XGBoost
# explainer_re_xgb = shap.Explainer(re_xgb_model, X_blind_test)
# shap_values_re_xgb = explainer_re_xgb(X_blind_test)

# # Initialize the SHAP explainer for retrained LightGBM
# explainer_re_lgb = shap.Explainer(re_lgb_model, X_blind_test)
# shap_values_re_lgb = explainer_re_lgb(X_blind_test)

# shap.summary_plot(shap_values_re_xgb, X_blind_test, plot_type="bar")  
# shap.summary_plot(shap_values_re_xgb, X_blind_test)

# shap.summary_plot(shap_values_re_lgb, X_blind_test, plot_type="bar")  
# shap.summary_plot(shap_values_re_lgb, X_blind_test)


# plt.figure(figsize=(12, 6))


# plt.plot(y_blind_test.values, label='Actual Values (Blind Test)', color='green', marker='o', linestyle='-', alpha=0.6)
# plt.plot(xgb_blind_test_preds, label='XGBoost Predictions (Blind Test)', color='orange', marker='x', linestyle='--', alpha=0.6)
# plt.plot(lgb_blind_test_preds, label='LightGBM Predictions (Blind Test)', color='blue', marker='s', linestyle='--', alpha=0.6)

# plt.title('Actual vs model predictions on blind test sets')
# plt.xlabel('Observations')
# plt.ylabel('Claims Incurred')
# plt.legend()

# plt.tight_layout()

# Define a separate dataset for the ARIMA model
arima_data = property_data_model.copy()

# Set 'Date' as index for time series modeling
arima_data.set_index('Date', inplace=True)

# Explicitly set the frequency to monthly (fixes the frequency warning)
arima_data.index = pd.date_range(start=arima_data.index[0], periods=len(arima_data), freq='ME')

# Split data for ARIMA
arima_train_data = arima_data[(arima_data.index.year >= 2008) & (arima_data.index.year <= 2022)]
arima_test_data = arima_data[(arima_data.index.year >= 2023) & (arima_data.index.year <= 2024)]

# Define target
#arima_y_train = arima_train_data['Claims_Incurred']
#arima_y_test = arima_test_data['Claims_Incurred']

# # Apply differencing to remove non-stationarity (optional based on your data)
# #arima_y_train_diff = arima_y_train.diff().dropna()


#### Best SARIMA Order: (1, 1, 3) with Seasonal Order: (1, 1, 1, 12) - AIC: 4988.444255903043
#### Best SARIMA Order: (3, 2, 3) with Seasonal Order: (1, 1, 1, 12) - AIC: 5523.916629307649
# Manually set the best parameters
best_pdq = (3, 2, 3)  # ARIMA order
best_seasonal_pdq = (1, 1, 1, 12)  # Seasonal order

# Fit the final ARIMA model with the manually set best order and seasonal order
#final_arima_model = sm.tsa.SARIMAX(arima_y_train, 
                                #    order=best_pdq, 
                                #    seasonal_order=best_seasonal_pdq,
                                #    )  # Relax invertibility constraint

# # Fit the model with more iterations and a stricter tolerance
#final_arima_model_fit = final_arima_model.fit(disp=False, maxiter=1000, tol=1e-6)

#arima_forecast = final_arima_model_fit.forecast(steps=len(arima_y_test))

# # Forecast 24 months AHEAD
# future_arima_forecast = final_arima_model_fit.forecast(steps=24)

# arima_mse = mean_squared_error(arima_y_test, arima_forecast)
# arima_rmse = np.sqrt(arima_mse)
# arima_mae = mean_absolute_error(arima_y_test, arima_forecast)
# arima_mape = mean_absolute_percentage_error(arima_y_test, arima_forecast) * 100  
# arima_r2 = r2_score(arima_y_test, arima_forecast)
# arima_bias = np.mean(arima_forecast - arima_y_test)  
# arima_accuracy = 100 - arima_mape


# plt.figure(figsize=(12, 6))
# plt.plot(arima_y_train.index, arima_y_train, label='Train')
# plt.plot(arima_y_test.index, arima_y_test, label='Test')
# plt.plot(arima_forecast.index, arima_forecast, label='ARIMA Forecast', color='black')
# plt.plot(pd.date_range(start=arima_y_test.index[-1], periods=24, freq='ME'), future_arima_forecast, label='Future Forecast', color='green')
# plt.xlabel('Date')
# plt.ylabel('Claims Incurred')
# plt.title('ARIMA forecast')
# plt.legend();


ma_data = property_data_model.copy()


# ma_train_data = ma_data[(ma_data['Date'].dt.year >= 2008) & (ma_data['Date'].dt.year <= 2022)]
ma_test_data = ma_data[(ma_data['Date'].dt.year >= 2023) & (ma_data['Date'].dt.year <= 2024)]

# # Ensure 'Date' is the index of your datasets
# ma_train_data.set_index('Date', inplace=True)
# ma_test_data.set_index('Date', inplace=True)

# ma_y_train = ma_train_data['Claims_Incurred']
ma_y_test = ma_test_data['Claims_Incurred']

# window_size = 3


# # Shift series to align predictions with actuals
# ma_train = ma_y_train.rolling(window=window_size).mean().shift(1)
# ma_test = ma_y_test.rolling(window=window_size).mean().shift(1)

# # Fill any NaN values generated at the beginning of the series
# ma_train.fillna(ma_y_train.mean(), inplace=True)
# ma_test.fillna(ma_y_test.mean(), inplace=True)

# ma_mse = mean_squared_error(ma_y_test, ma_test)
# ma_rmse = np.sqrt(ma_mse)
# ma_mae = mean_absolute_error(ma_y_test, ma_test)
# ma_mape = mean_absolute_percentage_error(ma_y_test, ma_test) * 100
# ma_r2 = r2_score(ma_y_test, ma_test)
# ma_bias = np.mean(ma_test - ma_y_test)
# ma_accuracy = 100 - ma_mape


# plt.figure(figsize=(13, 6))
# plt.plot(ma_y_train.index, ma_y_train, label='Train', color='blue')
# plt.plot(ma_y_test.index, ma_y_test, label='Test', color='orange')
# plt.plot(ma_y_test.index, ma_test, label=f'Moving Average (Window size {window_size})', color='green')
# plt.xlabel('Date')
# plt.ylabel('Claims Incurred')
# plt.title(f'Moving average (Window size {window_size}) vs actual claims')
# plt.legend()

#-------------------------------------------------------------------------------------

# Function to get XGBoost predictions
def get_xgboost_predictions(X_combined, y_combined, X_blind_test):
    re_xgb_model = XGBRegressor(**xgb_best_params)
    re_xgb_model.fit(X_combined, y_combined)
    return re_xgb_model.predict(X_blind_test)


#-------------------------------------------------------------------------------------

# Function to get LightGBM predictions
def get_lightgbm_predictions(X_combined, y_combined, X_blind_test):
    re_lgb_model = LGBMRegressor(**lgb_best_params)
    re_lgb_model.fit(X_combined, y_combined)
    return re_lgb_model.predict(X_blind_test)


#-------------------------------------------------------------------------------------


# Function to get ARIMA predictions
def get_arima_predictions(arima_train_data, arima_test_data, best_pdq, best_seasonal_pdq):
    # Define target for training
    arima_y_train = arima_train_data['Claims_Incurred']
    
    # Fit the final ARIMA model with the manually set best order and seasonal order
    final_arima_model = sm.tsa.SARIMAX(arima_y_train, 
                                       order=best_pdq, 
                                       seasonal_order=best_seasonal_pdq,
                                       enforce_stationarity=False,  
                                       enforce_invertibility=False)
    
    # Fit the model with more iterations and a stricter tolerance
    final_arima_model_fit = final_arima_model.fit(disp=False, maxiter=1000, tol=1e-6)
    
    # Forecast for the test period
    arima_forecast = final_arima_model_fit.forecast(steps=len(arima_test_data))
    
    return arima_forecast


#-------------------------------------------------------------------------------------

# Function to get Moving Average predictions
def get_moving_average_predictions(ma_y_test, window_size):
    return ma_y_test.rolling(window=window_size).mean().shift(1).fillna(ma_y_test.mean())


#-------------------------------------------------------------------------------------


# Function to get performance metrics
def calculate_model_metrics(y_true, y_pred):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle cases where y_true contains zeros to avoid division by zero in MAPE
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Replace 0 with a small epsilon value

    bias = np.mean(y_pred - y_true)

    # using the safe version of y_true
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    accuracy = 100 - mape

    return {'Bias': bias, 'MAPE': mape, 'Accuracy': accuracy}

#-------------------------------------------------------------------------------------
# Ensure your project root and assets folder is correctly targeted
assets_folder_path = os.path.join(os.getcwd(), 'assets')

# Function to get SHAP plots
def generate_shap_plot_xgboost(X_combined, y_combined, X_blind_test):
    re_xgb_model = XGBRegressor(**xgb_best_params)
    re_xgb_model.fit(X_combined, y_combined)
    explainer = shap.Explainer(re_xgb_model, X_blind_test)
    shap_values = explainer(X_blind_test)

    plt.figure()
    shap.summary_plot(shap_values, X_blind_test, show=False)  # Prevent it from showing directly
    save_path_xgb = os.path.join(assets_folder_path, 'shap_summary_xgboost.png')
    plt.savefig(save_path_xgb)  # Save the plot to assets folder
    plt.close()  # Close the plot to avoid memory issues

def generate_shap_plot_lightgbm(X_combined, y_combined, X_blind_test):
    re_lgb_model = LGBMRegressor(**lgb_best_params)
    re_lgb_model.fit(X_combined, y_combined)
    explainer = shap.Explainer(re_lgb_model, X_blind_test)
    shap_values = explainer(X_blind_test)

    plt.figure()
    shap.summary_plot(shap_values, X_blind_test, show=False)  # Prevent it from showing directly
    save_path_lgb = os.path.join(assets_folder_path, 'shap_summary_lightgbm.png')
    plt.savefig(save_path_lgb)  # Save the plot to assets folder
    plt.close()  # Close the plot to avoid memory issues

# Generate SHAP plots by calling the functions
generate_shap_plot_xgboost(X_combined, y_combined, X_blind_test)
generate_shap_plot_lightgbm(X_combined, y_combined, X_blind_test)


# Export the variables
#__all__ = ['arima_train_data', 'arima_test_data', 'ma_y_test', 'arima_test_data', 'arima_y_train']