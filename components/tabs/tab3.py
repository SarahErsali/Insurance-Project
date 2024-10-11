from dash import dcc, html

def render_tab3():
    return html.Div([
        html.H2("Model Performance", style={'textAlign': 'left', 'fontSize': '28px'}),

        # Instruction line for dropdown (model selection)
        html.P("Select the models you would like to evaluate:", style={
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
        html.P("Select the models for which you want to analyze performance metrics:", style={
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
        dcc.Graph(id='model-metrics-bar-chart')

    ], style={'padding': '20px', 'paddingLeft': '3cm', 'paddingRight': '3cm'})
