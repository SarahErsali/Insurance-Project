from dash import dcc, html

def render_tab3():
    return html.Div([
        html.H2("Model Performance", style={'textAlign': 'left'}),
       
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
                'marginLeft': '-10.3cm',  # Align left
                'marginBottom': '20px'  # Add some space below
            }
        ),

        # Graph for comparing models' predictions
        dcc.Graph(id='model-comparison-graph'),

        html.H2("Model Performance Metrics", style={'textAlign': 'left', 'marginTop': '40px'}),

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
                'marginLeft': '-10.3cm',  # Align left
                'marginBottom': '20px'  # Add some space below
            }
        ),

        # Graph for comparing models' metrics (Bias, Accuracy, MAPE)
        dcc.Graph(id='model-metrics-bar-chart')

    ], style={'padding': '20px', 'paddingLeft': '3cm', 'paddingRight': '3cm'})
