from dash import html

def render_home_tab():
    return html.Div([
        html.P("At BaNex Consulting, we empower the insurance industry to excel in today’s fast-paced world through cutting-edge, data-driven solutions that drive innovation, efficiency, and growth.",
               style={'textAlign': 'center', 'fontSize': '28px', 'marginTop': '2vh', 'lineHeight': '1.9',
                      'maxWidth': '80%', 'marginLeft': 'auto', 'marginRight': 'auto', 'paddingLeft': '3cm',
                      'paddingRight': '3cm'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'height': '60vh'})
