from dash import html, dcc

def render_tab4():
    return html.Div([
        html.H2("Models Stress Testing on Storm Periods", style={'textAlign': 'left', 'fontSize': '28px', 'marginTop': '3cm'}),

        html.P("The storm period plot presents a clear difference in how the models perform during storm events.",
               style={
                   'fontSize': '16px', 
                   'textAlign': 'left', 
                   'lineHeight': '2.0',
                   #'marginTop': '3cm'  # Space from the top of the page
               }),

        # Bullet points with bold titles
        html.Ul([
            html.Li([
                html.Span("XGBoost: ", style={'fontWeight': 'bold'}),
                "XGBoost predictions closely follow the actual claims during storm periods, with some deviations but generally capturing the trend well."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2.0'}),

            html.Li([
                html.Span("LightGBM: ", style={'fontWeight': 'bold'}),
                "LightGBM shows larger deviations from the actual claims, particularly during the later storm periods, where it underestimates the claims significantly. This suggests that XGBoost is better at handling these extreme conditions and LightGBM struggles more with the volatility introduced by storm scenarios, resulting in more significant errors."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2.0'}),

            html.Li([
                html.Span("LightGBM (2023-05): ", style={'fontWeight': 'bold'}),
                "The drop in predicted claims for LightGBM during the storm period (2023-05) is particularly notable, where it predicts significantly lower values than the actual observed claims."
            ], style={'fontSize': '16px', 'textAlign': 'left', 'lineHeight': '2.0'})
        ], style={'textAlign': 'left', 'marginBottom': '2cm'}),
        
        #html.H2("Model Stress Testing on Storm Periods", style={'textAlign': 'left', 'fontSize': '28px'}),

        # Instruction for dropdown
        html.P("Select the models you want to test during storm periods", style={
            'textAlign': 'left', 
            'fontSize': '16px', 
            'marginTop': '10px', 
            'marginBottom': '15px'
        }),

        # Dropdown for storm testing
        dcc.Dropdown(
            id='model-dropdown-storm',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LightGBM', 'value': 'lightgbm'},
            ],
            value=['xgboost'],  # Default value
            multi=True,  # Allow selecting multiple models
            style={'width': '500px', 'display': 'inline-block', 'marginLeft': '-10cm'}
        ),

        # Graph to show storm testing results
        dcc.Graph(id='storm-testing-graph'),

    ], style={'padding': '20px', 'paddingLeft': '3cm', 'paddingRight': '3cm'})

