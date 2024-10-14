from dash import dcc, html
from dash.dependencies import Input, Output
import shap
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt


def render_tab3():
    return html.Div([
        html.H2("Model Performance", style={'textAlign': 'left', 'fontSize': '28px'}),

        # Instruction line for dropdown (model selection)
        html.P("Select the models you would like to evaluate", style={
            'textAlign': 'left', 
            'fontSize': '16px', 
            'marginTop': '10px', 
            'marginBottom': '15px'  # Increase space between text and dropdown
        }),

        # Dropdown for model selection (for model predictions)
        dcc.Dropdown(
            id='model-dropdown-prediction',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LightGBM', 'value': 'lightgbm'},
                {'label': 'ARIMA', 'value': 'arima'},
                {'label': 'Moving Average', 'value': 'moving_average'}
            ],
            value=['xgboost'],  # Default values
            multi=True,  # Allow multiple selections
            style={
                'width': '500px',  # Make the dropdown smaller
                'display': 'inline-block',  # Align inline
                'marginTop': '20px',  # Add space above dropdown
                'marginLeft': '-10cm',  # Align left
                'marginBottom': '15px'  # Increase space below dropdown
            }
        ),

        # Graph for comparing models' predictions
        dcc.Graph(id='model-comparison-graph'),

        html.H2("Model Performance Metrics", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '50px'}),

        # Instruction line for dropdown (metrics selection)
        html.P("Select the models for which you want to analyze performance metrics", style={
            'textAlign': 'left', 
            'fontSize': '16px', 
            'marginTop': '10px', 
            'marginBottom': '15px'  # Increase space between text and dropdown
        }),

        # Dropdown for model selection (for metrics chart)
        dcc.Dropdown(
            id='model-dropdown-metrics',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LightGBM', 'value': 'lightgbm'},
                {'label': 'ARIMA', 'value': 'arima'},
                {'label': 'Moving Average', 'value': 'moving_average'}
            ],
            value=['xgboost'],  # Default values
            multi=True,  # Allow multiple selections
            style={
                'width': '500px',  # Make the dropdown smaller
                'display': 'inline-block',  # Align inline
                'marginTop': '20px',  # Add space above dropdown
                'marginLeft': '-10cm',  # Align left
                'marginBottom': '15px'  # Increase space below dropdown
            }
        ),

        # Graph for comparing models' metrics (Bias, Accuracy, MAPE)
        dcc.Graph(id='model-metrics-bar-chart'),

        
        
        # SHAP Plots (after performance metrics)
        html.H2("SHAP Summary Plots", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '50px'}),

        # XGBoost SHAP Plot Section
        html.Div([
            html.Div([
                html.H3("XGBoost Model", style={'textAlign': 'left', 'fontSize': '20px', 'marginRight': '20px'}),
                html.Img(src='/assets/shap_summary_xgboost.png', style={'height':'60%', 'width':'60%', 'display': 'block', 'marginLeft': '4cm'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'}),
        ], style={'marginBottom': '60px'}),

        # LightGBM SHAP Plot Section
        html.Div([
            html.Div([
                html.H3("LightGBM Model", style={'textAlign': 'left', 'fontSize': '20px', 'marginRight': '20px'}),
                html.Img(src='/assets/shap_summary_lightgbm.png', style={'height':'60%', 'width':'60%', 'display': 'block', 'marginLeft': '4cm'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'}),
        ], style={'marginTop': '50px'}),


        # Feature Importance Section
        html.H2("Feature Importance Analysis", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '50px'}),

        # Instruction line for dropdown (model selection)
        html.P("Select the models you would like to evaluate the feature importance", style={
            'textAlign': 'left', 
            'fontSize': '16px', 
            'marginTop': '10px', 
            'marginBottom': '15px'  # Increase space between text and dropdown
        }),

        # Dropdown for feature importance selection (XGBoost, LightGBM)
        dcc.Dropdown(
            id='feature-importance-dropdown',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LightGBM', 'value': 'lightgbm'}
            ],
            value='xgboost',  # Default to XGBoost
            style={
                'width': '500px',  # Make the dropdown smaller
                'display': 'inline-block',  # Align inline
                'marginTop': '20px',  # Add space above dropdown
                'marginLeft': '-10cm',  # Align left
                'marginBottom': '15px'  # Increase space below dropdown
            }
        ),

        # Bar chart for feature importance
        dcc.Graph(id='feature-importance-bar-chart'),

    ], style={'padding': '20px', 'paddingLeft': '3cm', 'paddingRight': '3cm'})
