import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from components.tabs.tab2 import render_tab2  # Import render function from tab2.py

# Initialize the app
app = dash.Dash(__name__)

# Set the app title
app.title = "Business Consultant Service"

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
        # Content for the Home tab (text larger, bold, and centered, with minimal margin and no scrolling)
        return html.Div([
            html.P("At BaNex Consulting, we empower the insurance companies to excel in today’s fast-paced world through cutting-edge, data-driven solutions that drive innovation, efficiency, and growth.",
                   style={
                       'textAlign': 'center', 
                       'fontSize': '28px', 
                       'marginTop': '2vh', 
                       'lineHeight': '1.9',
                       'maxWidth': '80%',  # Constrains the width of the paragraph
                       'marginLeft': 'auto', 
                       'marginRight': 'auto',  # Centers the text horizontally
                       'padding': '0 2cm'  # Adds 2cm padding on left and right
                   }),
        ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'height': '60vh'})
    
    elif tab == 'tab-1':
        # Business Objectives tab with summary of project and data
        return html.Div([

            # Business Problem
            html.H3("Business Problem", style={'marginBottom': '20px', 'fontSize': '30px'}),
            html.P("The project focuses on the property line of business within the insurance industry. "
                   "The key goal is to predict Solvency capital requirement (SCR) through developing predictive models for insurance claims incurred, factoring in various "
                   "economic indicators, crisis periods, and natural disasters' impact to improve business decisions.",
                   style={'lineHeight': '2', 'marginBottom': '30px'}),

            # Model Developed
            html.H3("Model Developed", style={'marginBottom': '20px', 'fontSize': '30px'}),
            html.P("For this project, advanced machine learning models such as XGBoost and LightGBM were developed. "
                   "These models help predict claims incurred based on a variety of features, including economic data, "
                   "insurance data, and specific factors affecting the property insurance line.",
                   style={'lineHeight': '2', 'marginBottom': '30px'}),

            # Data Summary
            html.H3("Data Summary", style={'marginBottom': '20px', 'fontSize': '30px'}),
            html.Ul([
                html.Li([html.B("Features:"), " A wide range of features were used, including underwriting risk, number of policies, expenses, etc."]),
                html.Li([html.B("Economic Data:"), " This includes key macroeconomic indicators that impact claims behavior, "
                        "such as GDP growth rate, inflation rate, unemployment rate, interest rate, and equity return."]),
                html.Li([html.B("Crisis Periods:"), " The dataset includes multiple crisis periods like the 2008 Financial "
                        "Crisis, the European Debt Crisis, and the COVID-19 pandemic. Each of these periods has a "
                        "significant impact on the model's predictions, as they affect economic indicators and insurance "
                        "risk."]),
                html.Li([html.B("Natural Disaster:"), " Several storm periods in the dataset typically cause a spike in property insurance claims, impacting the risk evaluation."])
            ], style={'lineHeight': '2', 'marginBottom': '30px'}),

            # Impact on Business Model
            html.H3("Impact on Analysis and Business Model", style={'marginBottom': '20px', 'fontSize': '30px'}),
            html.P("The features and factors used in the models play a critical role in understanding how external "
                   "factors, such as the economy and catastrophic events, influence the number of claims. This aids in "
                   "refining the insurance pricing strategy and improving risk management practices.",
                   style={'lineHeight': '2', 'marginBottom': '30px'})
        ], style={
            'padding': '10px',  # Less padding for left and right margins
            'textAlign': 'left',
            'maxWidth': '90%',  # More space for content width
            'marginLeft': 'auto',  # Center the div horizontally
            'marginRight': 'auto',
            'lineHeight': '1.8',  # Increase line height for readability
        })

    elif tab == 'tab-2':
        return render_tab2()  # Rendering the content from tab2.py with plots

    elif tab == 'tab-3':
        # Placeholder content for Model Performance tab
        return html.Div([
            html.H2("Model Performance", style={'textAlign': 'center'}),
            html.P("This section will display the performance of models.", style={'textAlign': 'center'}),
        ])
    
    elif tab == 'tab-4':
        # Placeholder content for Stress Testing tab
        return html.Div([
            html.H2("Model Robustness", style={'textAlign': 'center'}),
            html.P("This section will showcase the results of stress testing.", style={'textAlign': 'center'}),
        ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
