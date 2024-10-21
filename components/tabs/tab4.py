from dash import html, dcc

def render_tab4():
    return html.Div([
        # Title for Storm Testing
        html.H2("Models Stress Testing on Storm Periods", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '3cm'}),

        # Description for Storm Testing
        html.P("The storm period plot presents a clear difference in how the models perform during storm events.",
               style={
                   'fontSize': '16px', 
                   'textAlign': 'left', 
                   'lineHeight': '2.0',
               }),

        # Bullet points explaining each model's performance during storm periods
        html.Ul([
            html.Li([
                html.Span("XGBoost: ", style={'fontWeight': 'bold'}),
                "XGBoost predictions closely follow the actual claims during storm periods, with some deviations but generally capturing the trend well."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2.0'}),

            html.Li([
                html.Span("LightGBM: ", style={'fontWeight': 'bold'}),
                "LightGBM shows larger deviations from the actual claims, particularly during the later storm periods, where it underestimates the claims significantly."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2.0'}),

            html.Li([
                html.Span("LightGBM (2023-05): ", style={'fontWeight': 'bold'}),
                "The drop in predicted claims for LightGBM during the storm period (2023-05) is particularly notable."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2.0'})
        ], style={'textAlign': 'left', 'marginBottom': '2cm'}),
        

        # Instruction for storm model selection
        html.P("Select the models you want to test during storm periods:", style={
            'textAlign': 'left', 
            'fontSize': '16px', 
            'marginTop': '10px', 
            'marginBottom': '15px'
        }),

        # Dropdown for storm model selection
        dcc.Dropdown(
            id='model-dropdown-storm',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LightGBM', 'value': 'lightgbm'},
                {'label': 'ARIMA', 'value': 'arima'},
                {'label': 'Moving Average', 'value': 'moving_average'}
            ],
            value=['xgboost'],  # Default value
            multi=True,  # Allow selecting multiple models
            style={'width': '50%', 'margin': 'auto'}
        ),

        # Graph for storm testing
        dcc.Graph(id='storm-testing-graph'),



#----------------- Backtesting Section -------------------------------------------

        # Title for Backtesting
        html.H2("Backtesting Metrics", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '4cm'}),

        # Instruction for backtesting dropdown
        html.P("Select the models you want to see backtesting metrics for:", style={
            'textAlign': 'left', 
            'fontSize': '16px', 
            'marginTop': '10px', 
            'marginBottom': '15px'
        }),

        # Dropdown for backtesting model selection
        dcc.Dropdown(
            id='model-dropdown-backtest',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LightGBM', 'value': 'lightgbm'},
                {'label': 'ARIMA', 'value': 'arima'},
                {'label': 'Moving Average', 'value': 'moving_average'}
            ],
            value=['xgboost'],  # Default value
            multi=True,  # Allow selecting multiple models
            style={'width': '50%', 'margin': 'auto'}
        ),

        # Stacked bar charts for Bias, Accuracy, and MAPE (now vertically aligned)
        html.Div([
            # Bias Bar Chart
            html.Div([
                html.H3("Bias", style={'textAlign': 'center', 'fontSize': '20px'}),
                dcc.Graph(id='backtest-bias-chart')
            ], style={'padding': '10px'}),  # No flex needed for stacking

            # Accuracy Bar Chart
            html.Div([
                html.H3("Accuracy %", style={'textAlign': 'center', 'fontSize': '20px'}),
                dcc.Graph(id='backtest-accuracy-chart')
            ], style={'padding': '10px'}),  # No flex needed for stacking

            # MAPE Bar Chart
            html.Div([
                html.H3("MAPE %", style={'textAlign': 'center', 'fontSize': '20px'}),
                dcc.Graph(id='backtest-mape-chart')
            ], style={'padding': '10px'})  # No flex needed for stacking
        ], style={'display': 'block'}),  # Use block display to stack elements

        ], style={'padding': '20px', 'paddingLeft': '3cm', 'paddingRight': '3cm'})