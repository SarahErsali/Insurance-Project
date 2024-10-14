from dash import html, dcc

def render_tab4():
    return html.Div([
        html.H2("Model Stress Testing on Storm Periods", style={'textAlign': 'left', 'fontSize': '28px'}),

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

