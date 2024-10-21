from dash import html, dash_table

def render_tab5():
    return html.Div([
        html.H3("Future predictions for the property line of business with the best two models", 
                style={'textAlign': 'center', 'fontSize': '24px', 'lineHeight': '1.6'}),  # Increased line height
        
        html.P("The table shows the future predictions for each month throughout a year ahead, and highlights the model with the best prediction.",
               style={'textAlign': 'center', 'lineHeight': '1.8', 'marginBottom': '40px'}),  # More spacing
        
        # Table with adjusted size and appearance
        dash_table.DataTable(
            id='future-prediction-table',
            columns=[#{'name': 'LOB', 'id': 'LOB'}, 
                     {'name': 'Date', 'id': 'Date'},
                     {'name': 'Prediction of Property LOB', 'id': 'Prediction'},
                     {'name': 'Model', 'id': 'Model'}],
            style_cell={
                'textAlign': 'center',  # Center align text
                'padding': '5px',  # Reduce padding for smaller table cells
                'fontSize': '12px',  # Set smaller font size
            },
            style_table={
                'overflowY': 'auto',  # Enable vertical scroll if needed
                'width': '50%',  # Adjust the width of the table
                'margin': 'auto',  # Center the table in the layout
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold',
                'fontSize': '14px',  # Set a slightly larger font for the headers
            },
        )  
    ], style={'paddingTop': '3cm', 'paddingBottom': '3cm', 'paddingLeft': '20px', 'paddingRight': '20px'}) 