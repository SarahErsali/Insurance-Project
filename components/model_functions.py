import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm
from sklearn.metrics import (mean_absolute_percentage_error)
import shap
from components.data import X_train, y_train, X_val, y_val, X_blind_test, property_data_model, property_data_feature_selected
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt
import os




X_combined = pd.concat([X_train, X_val], ignore_index=True)
y_combined = pd.concat([y_train, y_val], ignore_index=True)
xgb_best_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'gamma': 0,
}
re_xgb_model = XGBRegressor(**xgb_best_params)
re_xgb_model.fit(X_combined, y_combined)
lgb_best_params = {
    'n_estimators': 200,
    'max_depth': 5,  
    'learning_rate': 0.1,
    'num_leaves': 10
}
re_lgb_model = LGBMRegressor(**lgb_best_params)
re_lgb_model.fit(X_combined, y_combined)
# Define a separate dataset for the ARIMA model
arima_data = property_data_model.copy()
# Set 'Date' as index for time series modeling
arima_data.set_index('Date', inplace=True)
# Explicitly set the frequency to monthly (fixes the frequency warning)
arima_data.index = pd.date_range(start=arima_data.index[0], periods=len(arima_data), freq='ME')
# Split data for ARIMA
arima_train_data = arima_data[(arima_data.index.year >= 2008) & (arima_data.index.year <= 2022)]
arima_test_data = arima_data[(arima_data.index.year >= 2023) & (arima_data.index.year <= 2024)]
# Manually set the best parameters
best_pdq = (3, 2, 3)  # ARIMA order
best_seasonal_pdq = (1, 1, 1, 12)  # Seasonal order
ma_data = property_data_model.copy()
# ma_train_data = ma_data[(ma_data['Date'].dt.year >= 2008) & (ma_data['Date'].dt.year <= 2022)]
ma_test_data = ma_data[(ma_data['Date'].dt.year >= 2023) & (ma_data['Date'].dt.year <= 2024)]
# ma_y_train = ma_train_data['Claims_Incurred']
ma_y_test = ma_test_data['Claims_Incurred']




#-------- Function to get XGBoost predictions --------------------------------------------------



def get_xgboost_predictions(X_combined, y_combined, X_blind_test):
    re_xgb_model = XGBRegressor(**xgb_best_params)
    re_xgb_model.fit(X_combined, y_combined)
    return re_xgb_model.predict(X_blind_test)



#--------- Function to get LightGBM predictions --------------------------------------------------



def get_lightgbm_predictions(X_combined, y_combined, X_blind_test):
    re_lgb_model = LGBMRegressor(**lgb_best_params)
    re_lgb_model.fit(X_combined, y_combined)
    return re_lgb_model.predict(X_blind_test)



#---------- Function to get ARIMA predictions -----------------------------------------------------




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



#-------- Function to get Moving Average predictions -------------------------------------------------


def get_moving_average_predictions(ma_y_test, window_size):
    return ma_y_test.rolling(window=window_size).mean().shift(1).fillna(ma_y_test.mean())



#----------- Function to get performance metrics -------------------------------------------------



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



#----------- Function to get SHAP Analysis -------------------------------------------------------------------------------



# Ensure your project root and assets folder is correctly targeted
assets_folder_path = os.path.join(os.getcwd(), 'assets')

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
    shap.summary_plot(shap_values, X_blind_test, show=False) 
    save_path_lgb = os.path.join(assets_folder_path, 'shap_summary_lightgbm.png')
    plt.savefig(save_path_lgb)  
    plt.close()  

# Generate SHAP plots by calling the functions
generate_shap_plot_xgboost(X_combined, y_combined, X_blind_test)
generate_shap_plot_lightgbm(X_combined, y_combined, X_blind_test)



#-------- Function to get feature importance -------------------------------------------------------------



def get_xgboost_feature_importance(X_combined, y_combined):
    re_xgb_model = XGBRegressor(**xgb_best_params)
    re_xgb_model.fit(X_combined, y_combined)
    
    # Get feature importance and return as a DataFrame
    re_feature_importance_xgb = pd.DataFrame({
        'Feature': X_combined.columns,
        'Importance': re_xgb_model.feature_importances_
    })

    re_feature_importance_xgb = re_feature_importance_xgb.sort_values(by='Importance', ascending=False)
    return re_feature_importance_xgb

def get_lightgbm_feature_importance(X_combined, y_combined):
    re_lgb_model = LGBMRegressor(**lgb_best_params)
    re_lgb_model.fit(X_combined, y_combined)
    
    # Get feature importance and return as a DataFrame
    re_feature_importance_lgb = pd.DataFrame({
        'Feature': X_combined.columns,
        'Importance': re_lgb_model.feature_importances_
    })

    re_feature_importance_lgb = re_feature_importance_lgb.sort_values(by='Importance', ascending=False)
    return re_feature_importance_lgb



#------------- Function to get Stress Testing ---------------------------------------------------------



window_size = 3

# Prepare data for stress testing
property_data_feature_selected_s = property_data_feature_selected.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
property_data_model_s = property_data_model.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')

# Actual data for ML and ts models
ml_actual_y = property_data_feature_selected['Claims_Incurred']
ts_actual_y = property_data_model['Claims_Incurred']

def get_xgboost_predictions_storm():
    return re_xgb_model.predict(property_data_feature_selected_s)

def get_lightgbm_predictions_storm():
    return re_lgb_model.predict(property_data_feature_selected_s)

def get_arima_predictions_storm():
    arima_y_train = arima_train_data['Claims_Incurred']
    final_arima_model = sm.tsa.SARIMAX(arima_y_train, 
                                       order=best_pdq, 
                                       seasonal_order=best_seasonal_pdq,
                                       enforce_stationarity=False,  
                                       enforce_invertibility=False)
    
    final_arima_model_fit = final_arima_model.fit(disp=False, maxiter=1000, tol=1e-6)
    return final_arima_model_fit.forecast(steps=len(property_data_model))

def get_moving_average_predictions_storm():
    storm_ma_prediction = ts_actual_y.rolling(window=window_size).mean().shift(1)
    storm_ma_prediction.fillna(ts_actual_y.mean(), inplace=True)
    return storm_ma_prediction




#------------- Function to get Back Testing -------------------------------------------------------------------

# Parameters:
#     - data: DataFrame containing the data to backtest.
#     - model_type: String specifying the model ('xgb', 'lgb', 'arima', 'ma').
#     - window: Integer specifying the size of the rolling window.

#     Returns:
#     - DataFrame with statistical summary of backtesting results.



def backtest(data, model_type, window=24, test_size=12, num_cycles=3):
    metrics = {'bias': [], 'accuracy': [], 'mape': []}

    for i in range(num_cycles):
        train, test = data.iloc[i:i + window], data.iloc[i + window:i + window + test_size]
        b_X_train = train.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
        b_y_train = train['Claims_Incurred']
        b_X_test = test.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
        b_y_test = test['Claims_Incurred']

        if model_type == 'xgb':
            preds = XGBRegressor().fit(b_X_train, b_y_train).predict(b_X_test)
        elif model_type == 'lgb':
            preds = LGBMRegressor().fit(b_X_train, b_y_train).predict(b_X_test)
        elif model_type == 'arima':
            arima_model = sm.tsa.SARIMAX(b_y_train, order=(3, 2, 3), seasonal_order=(1, 1, 1, 12))
            arima_fitted = arima_model.fit(disp=False)
            preds = arima_fitted.forecast(test_size)
                                
        elif model_type == 'ma':
            preds = [b_y_train.rolling(window=3).mean().iloc[-1]] * test_size

        
        bias = np.mean(preds - b_y_test)
        accuracy = 100 - (mean_absolute_percentage_error(b_y_test, preds) * 100)
        mape = mean_absolute_percentage_error(b_y_test, preds) * 100

        metrics['bias'].append(bias)
        metrics['accuracy'].append(accuracy)
        metrics['mape'].append(mape)

    return pd.DataFrame(metrics).describe()


def get_xgb_backtest_results(data):
    return backtest(data, 'xgb', window=66, test_size=12, num_cycles=3)

def get_lgb_backtest_results(data):
    return backtest(data, 'lgb', window=66, test_size=12, num_cycles=3)

def get_arima_backtest_results(data):
    return backtest(data, 'arima', window=66, test_size=12, num_cycles=3)

def get_ma_backtest_results(data):
    return backtest(data, 'ma', window=66, test_size=12, num_cycles=3)




#------------- Function to get Business Solution -------------------------------------------------------------------




def generate_forecast_tables(final_arima_model_fit, re_lgb_model, X_combined):
    # ARIMA Forecast (Nov 2024 - Oct 2025)
    forecast_dates = pd.date_range(start='2024-11-01', periods=12, freq='ME')
    arima_future_predictions = final_arima_model_fit.forecast(steps=12)

    arima_forecast_table = pd.DataFrame({
        'LOB': 'Property',
        'Date': forecast_dates.strftime('%Y-%b'),  # Format YYYYMon
        'Prediction': arima_future_predictions,
        'Model': 'ARIMA'
    })

    arima_forecast_table.reset_index(drop=True, inplace=True)

    # LightGBM Future Forecast (Nov 2024 - Oct 2025)
    # Generate future months and quarters
    lgb_future_dates = pd.date_range(start='2024-11-01', periods=12, freq='ME')
    lgb_future_months = lgb_future_dates.month
    lgb_future_quarters = lgb_future_dates.quarter

    # Static features from X_combined
    latest_gdp_growth_rate = X_combined['GDP_Growth_Rate'].iloc[-1]
    latest_inflation_rate = X_combined['Inflation_Rate'].iloc[-1]
    latest_unemployment_rate = X_combined['Unemployment_Rate'].iloc[-1]
    latest_interest_rate = X_combined['Interest_Rate'].iloc[-1]
    latest_equity_return = X_combined['Equity_Return'].iloc[-1]
    latest_expenses = X_combined['Expenses'].iloc[-1]
    latest_scr = X_combined['SCR'].iloc[-1]

    # Assuming no crisis in the future
    future_is_crisis = [0] * 12

    # Create the future feature set for LightGBM
    lgb_future_features = pd.DataFrame({
        'GDP_Growth_Rate': [latest_gdp_growth_rate] * 12,
        'Inflation_Rate': [latest_inflation_rate] * 12,
        'Unemployment_Rate': [latest_unemployment_rate] * 12,
        'Interest_Rate': [latest_interest_rate] * 12,
        'Equity_Return': [latest_equity_return] * 12,
        'Expenses': [latest_expenses] * 12,
        'Month': lgb_future_months,
        'Quarter': lgb_future_quarters,
        'SCR': [latest_scr] * 12,
        'Is_Crisis': future_is_crisis
    })

    # Predict future claims with LightGBM
    lgb_future_predictions = re_lgb_model.predict(lgb_future_features)

    lgb_forecast_table = pd.DataFrame({
        'LOB': 'Property',
        'Date': lgb_future_dates.strftime('%Y-%b'),
        'Prediction': lgb_future_predictions.round(2),
        'Model': 'LightGBM'
    })

    lgb_forecast_table.reset_index(drop=True, inplace=True)

    # Combine ARIMA and LightGBM Predictions
    combined_forecast_table = pd.DataFrame({
        'LOB': 'Property',
        'Date': arima_forecast_table['Date'],
        'ARIMA_Prediction': arima_forecast_table['Prediction'],
        'LightGBM_Prediction': lgb_forecast_table['Prediction']
    })

    # Select the best prediction (can customize the selection criteria)
    combined_forecast_table['Best_Prediction'] = combined_forecast_table[['ARIMA_Prediction', 'LightGBM_Prediction']].min(axis=1)

    # Assign the best model
    combined_forecast_table['Model'] = np.where(
        combined_forecast_table['Best_Prediction'] == combined_forecast_table['ARIMA_Prediction'], 'ARIMA', 'LightGBM'
    )

    # Create the final forecast table
    final_forecast_table = combined_forecast_table[['LOB', 'Date', 'Best_Prediction', 'Model']].copy()
    final_forecast_table.rename(columns={'Best_Prediction': 'Prediction'}, inplace=True)

    # Modify the 'LOB' column to only show 'Property' in the middle row
    middle_row = len(final_forecast_table) // 2
    final_forecast_table['LOB'] = [''] * len(final_forecast_table)  # Empty the LOB column
    final_forecast_table.loc[middle_row, 'LOB'] = 'Property'  # Set 'Property' in the middle row
    
    return final_forecast_table
