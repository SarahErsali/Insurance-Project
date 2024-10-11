import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from components.tabs.tab2 import render_tab2
from components.tabs.tab1 import render_tab1
from components.tabs.tab3 import render_tab3
from components.data import X_combined, y_combined, X_blind_test, y_blind_test
from components.model_functions import (
    get_xgboost_predictions,
    get_lightgbm_predictions,
    get_arima_predictions,
    get_moving_average_predictions,
    calculate_model_metrics,
    arima_test_data,
    ma_y_test,
    arima_train_data
)


# Initialize the app
app = dash.Dash(__name__)

# Set the app title
app.title = "Insurance Consultant Service"

# Define the layout of the app
app.layout = html.Div([
    # Header with company name
    html.Header([
        html.H1("Welcome to BaNex Consulting", style={'textAlign': 'center', 'fontSize': '48px', 'marginTop': '10px'}),
    ], style={'backgroundColor': '#f0f0f0', 'padding': '50px'}),

    # Navigation tabs
    dcc.Tabs(id='tabs-example', value='home', children=[
        dcc.Tab(label='Home', value='home'),
        dcc.Tab(label='Business Objectives', value='tab-1'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2'),
        dcc.Tab(label='Model Performance', value='tab-3'),
        dcc.Tab(label='Model Robustness', value='tab-4'),
    ]),

    # Content section that changes with each tab
    html.Div(id='tabs-content', style={'textAlign': 'center', 'padding': '0px', 'height': '50vh'})
])

# Callback to update the page content based on the selected tab
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'home':
        return html.Div([
            html.P("At BaNex Consulting, we empower the insurance industry to excel in today’s fast-paced world through cutting-edge, data-driven solutions that drive innovation, efficiency, and growth.",
                   style={'textAlign': 'center', 'fontSize': '28px', 'marginTop': '2vh', 'lineHeight': '1.9',
                          'maxWidth': '80%', 'marginLeft': 'auto', 'marginRight': 'auto', 'paddingLeft': '3cm',
                          'paddingRight': '3cm'}),
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'height': '60vh'})

    elif tab == 'tab-1':
        return render_tab1()

    elif tab == 'tab-2':
        return render_tab2()

    elif tab == 'tab-3':
        return render_tab3()
    
    elif tab == 'tab-4':
        return html.Div([
            html.H2("Model Robustness", style={'textAlign': 'center'}),
            html.P("This section will showcase the results of stress testing.", style={'textAlign': 'center'}),
        ])


# Callback for Model Predictions Plot
@app.callback(
    Output('model-comparison-graph', 'figure'),
    Input('model-dropdown-prediction', 'value')
)
def update_model_predictions(models_selected):
    fig = go.Figure()

    # Fetch predictions for each selected model
    for model in models_selected:
        if model == 'xgboost':
            preds = get_xgboost_predictions(X_combined, y_combined, X_blind_test)
            print(f"XGBoost Predictions: {preds}")
            preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
            fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='XGBoost', line=dict(color='purple')))

        if model == 'lightgbm':
            preds = get_lightgbm_predictions(X_combined, y_combined, X_blind_test)
            print(f"LightGBM Predictions: {preds}")
            preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
            fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='LightGBM', line=dict(color='blue')))

        if model == 'arima':
            preds = get_arima_predictions(arima_train_data, arima_test_data, (3, 2, 3), (1, 1, 1, 12))
            print(f"ARIMA Predictions: {preds}")
            preds = pd.Series(preds, index=arima_test_data.index)
            fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='ARIMA', line=dict(color='green')))

        if model == 'moving_average':
            preds = get_moving_average_predictions(ma_y_test, window_size=3)
            print(f"Moving Average Predictions: {preds}")
            preds = pd.Series(preds, index=ma_y_test.index)
            fig.add_trace(go.Scatter(x=y_blind_test.index, y=preds, mode='lines', name='Moving Average', line=dict(color='red')))

    # Add the actual values to the original graph
    fig.add_trace(go.Scatter(x=y_blind_test.index, y=y_blind_test, mode='lines', name='Actual', line=dict(color='black', dash='dot')))
    fig.update_layout(xaxis_title='Date', yaxis_title='Claims Incurred', xaxis_showgrid=False, yaxis_showgrid=False)

    return fig


# Callback for Metrics Bar Chart
@app.callback(
    Output('model-metrics-bar-chart', 'figure'),
    Input('model-dropdown-metrics', 'value')
)
def update_metrics_chart(models_selected):
    # New figure for the metrics bar chart
    metrics_fig = go.Figure()

    # Metrics storage for bias, accuracy, and MAPE
    metrics = {'Bias': [], 'Accuracy': [], 'MAPE': []}
    model_names = []


    # Fetch metrics for each selected model
    for model in models_selected:
        if model == 'xgboost':
            preds = get_xgboost_predictions(X_combined, y_combined, X_blind_test)
            preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
            model_metrics = calculate_model_metrics(y_blind_test, preds)
            model_names.append('XGBoost')
            metrics['Bias'].append(model_metrics['Bias'] / 1000)
            metrics['Accuracy'].append(model_metrics['Accuracy'])
            metrics['MAPE'].append(model_metrics['MAPE'])

        if model == 'lightgbm':
            preds = get_lightgbm_predictions(X_combined, y_combined, X_blind_test)
            preds = pd.Series(preds, index=y_blind_test.reset_index(drop=True))
            model_metrics = calculate_model_metrics(y_blind_test, preds)
            model_names.append('LightGBM')
            metrics['Bias'].append(model_metrics['Bias'] / 1000)
            metrics['Accuracy'].append(model_metrics['Accuracy'])
            metrics['MAPE'].append(model_metrics['MAPE'])

        if model == 'arima':
            preds = get_arima_predictions(arima_train_data, arima_test_data, (3, 2, 3), (1, 1, 1, 12))
            preds = pd.Series(preds, index=arima_test_data.index)
            model_metrics = calculate_model_metrics(y_blind_test, preds)
            model_names.append('ARIMA')
            metrics['Bias'].append(model_metrics['Bias'] / 1000)
            metrics['Accuracy'].append(model_metrics['Accuracy'])
            metrics['MAPE'].append(model_metrics['MAPE'])

        if model == 'moving_average':
            preds = get_moving_average_predictions(ma_y_test, window_size=3)
            preds = pd.Series(preds, index=ma_y_test.index)
            model_metrics = calculate_model_metrics(y_blind_test, preds)
            model_names.append('Moving Average')
            metrics['Bias'].append(model_metrics['Bias'] / 1000)
            metrics['Accuracy'].append(model_metrics['Accuracy'])
            metrics['MAPE'].append(model_metrics['MAPE'])

    
    # Debug output to track what is happening
    print("Metrics before plotting:")
    print("Model Names:", model_names)
    print("Bias Values:", metrics['Bias'])
    print("Accuracy Values:", metrics['Accuracy'])
    print("MAPE Values:", metrics['MAPE'])


    # keep the Debug: Convert NaNs to 0 and ensure all values are floats
    for key in metrics:
        metrics[key] = [0.0 if np.isnan(value) else float(value) for value in metrics[key]]
    


    # Create a horizontal bar chart for metrics (Bias, Accuracy, MAPE)
    metrics_fig.add_trace(go.Bar(y=model_names, x=metrics['Bias'], name='Bias', marker_color='orange', opacity=0.7, orientation='h'))
    metrics_fig.add_trace(go.Bar(y=model_names, x=metrics['Accuracy'], name='Accuracy', marker_color='green', opacity=0.5, orientation='h'))
    metrics_fig.add_trace(go.Bar(y=model_names, x=metrics['MAPE'], name='MAPE', marker_color='blue', opacity=0.5, orientation='h'))

    metrics_fig.update_layout(
        barmode='group',
        #title="Model Performance Metrics",
        #yaxis_title="Model",
        xaxis_title="Metric Value",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        bargap=0.15,  # Increase space between bars
        bargroupgap=0.1  # Increase space between groups
    )

    return metrics_fig



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
