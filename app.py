import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from components.tabs import tab1, tab2, tab3, tab4
from components import navbar

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar.create_navbar(),  # Navbar component
    html.Div(id='page-content')  # Where the tabs' content will be displayed
])

# Callback to update the content based on the tab selected
@app.callback(Output('page-content', 'children'), 
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/business-objectives':
        return tab1.layout
    elif pathname == '/exploratory-data-analysis':
        return tab2.layout
    elif pathname == '/model-performance':
        return tab3.layout
    elif pathname == '/model-robustness':
        return tab4.layout
    else:
        return tab1.layout  # Default to Tab 1

if __name__ == '__main__':
    app.run_server(debug=True)
