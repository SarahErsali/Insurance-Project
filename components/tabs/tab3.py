from dash import dcc, html
from dash.dependencies import Input, Output
import shap
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt


def render_tab3():
    return html.Div([
        html.H3("Models Evaluation", style={
            'fontWeight': 'bold', 
            'textAlign': 'left', 
            'fontSize': '30px', 
            'marginTop': '3cm'  # 3 cm space to the top of the page
        }),

        html.P("The line plot shows the predictions of four different models—XGBoost, ARIMA, LightGBM, and Moving Average—against the actual values for 'Claims Incurred' over a period. The dotted black line represents the actual observed values, while the colored lines show the predictions of the models.",
               style={
                   'fontSize': '16px', 
                   'textAlign': 'left', 
                   'lineHeight': '2.0',
                   #'marginTop': '3cm'  # Space from the top of the page
               }),

        # Bullet points for model interpretation
        html.Ul([
            html.Li([
                html.Span("XGBoost: ", style={'fontWeight': 'bold'}),
                "In the flatter regions or moderate fluctuations, XGBoost tends to smooth out the predictions but still follows the overall pattern. The deviation from actual values is relatively small."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),  # Increased line spacing
            
            html.Li([
                html.Span("LightGBM: ", style={'fontWeight': 'bold'}),
                "LightGBM seems less stable and has larger prediction errors in regions with moderate variations in claims."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),  # Increased line spacing
            
            html.Li([
                html.Span("ARIMA: ", style={'fontWeight': 'bold'}),
                "ARIMA performs better in flatter regions, where the actual values are relatively stable."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),  # Increased line spacing
            
            html.Li([
                html.Span("Moving Average: ", style={'fontWeight': 'bold'}),
                "The Moving Average model shows a smoother pattern compared to the others, which is expected due to its nature. However, it struggles to capture sudden shifts and spikes in the data."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'})  # Increased line spacing
        ], style={'textAlign': 'left', 'marginBottom': '2.5cm'}),


        #html.H2("Model Performance", style={'textAlign': 'left', 'fontSize': '28px'}),


        # Instruction line for dropdown (model selection)
        html.P("Select the models you would like to evaluate", style={
            'textAlign': 'left', 
            'fontSize': '20px', 
            'marginTop': '4px', 
            'marginBottom': '5px'  # Increase space between text and dropdown
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

        html.H3("Models Performance Metrics", style={
            'fontWeight': 'bold', 
            'textAlign': 'left', 
            'fontSize': '30px', 
            'marginTop': '3cm'  # 3 cm space to the top of the page
        }),
        #html.H2("Model Performance ", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '50px'}),
        html.P("The bar chart evaluates each model based on three key metrics: Bias, Accuracy, and MAPE.",
               style={
                   'fontSize': '16px', 
                   'textAlign': 'left', 
                   'lineHeight': '2.0',
                   #'marginTop': '3cm'  # Space from the top of the page
               }),
        
        # Text explanation for the performance metrics in bullet points
        html.Ul([
            html.Li([
                html.Span("ARIMA: ", style={'fontWeight': 'bold'}),
                "ARIMA achieves moderate accuracy, lower than XGBoost but better than LightGBM and Moving Average. In addition, it has a slightly negative bias but is closer to zero than the other models, which means it provides relatively unbiased predictions on average. Moreover, the MAPE for ARIMA is also reasonable, indicating decent predictive performance."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),

            html.Li([
                html.Span("LightGBM: ", style={'fontWeight': 'bold'}),
                "LightGBM has a more substantial negative and high bias and error, suggesting it consistently underpredicts the actual values, although its accuracy value is high. It shows more volatility in predictions, making it less reliable than other models."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2'}),   
        ], style={'textAlign': 'left', 'marginBottom': '2.5cm'}),


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
