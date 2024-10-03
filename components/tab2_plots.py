import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_economic_histograms(df, columns):
    """
    Plot histograms of economic variables using Plotly Subplots.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - columns (list): List of column names to plot.

    Returns:
    - fig (Plotly figure)
    """
    # Create a subplot figure with one row and multiple columns
    num_columns = len(columns)
    fig = make_subplots(rows=1, cols=num_columns, subplot_titles=columns, horizontal_spacing=0.06)  # Adjust the horizontal spacing

    # Add histograms for each economic-related variable in its own subplot
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=df[col], nbinsx=30, showlegend=False),
            row=1, col=i+1
        )

        # Update each subplot's layout: removing grids and setting titles
        fig.update_xaxes(title_text='', showgrid=False, row=1, col=i+1)
        fig.update_yaxes(title_text='', showgrid=False, row=1, col=i+1)

    # Update the layout for the entire figure
    fig.update_layout(
        title_text='✔️ <b>Distribution of Economic Indicators</b>',
        height=600,  # Adjusted for better spacing
        width=1300,  # Adjust width for better fitting of multiple subplots
        showlegend=False
    )
    
    return fig




def plot_insurance_histograms(df, columns):
    """
    Plot histograms of insurance-related variables using Plotly Subplots.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - columns (list): List of column names to plot.

    Returns:
    - fig (Plotly figure)
    """
    # Create a subplot figure with one row and multiple columns
    num_columns = len(columns)
    fig = make_subplots(rows=1, cols=num_columns, subplot_titles=columns, horizontal_spacing=0.06)  # Adjust the horizontal spacing

    # Add histograms for each insurance-related variable in its own subplot
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=df[col], nbinsx=30, showlegend=False),
            row=1, col=i+1
        )

        # Update each subplot's layout: removing grids and setting titles
        fig.update_xaxes(title_text='', showgrid=False, row=1, col=i+1)
        fig.update_yaxes(title_text='', showgrid=False, row=1, col=i+1)

    # Update the layout for the entire figure
    fig.update_layout(
        title_text='✔️ <b>Distribution of Insurance Variables</b>',
        height=600,  # Adjusted for better spacing
        width=1300,  # Adjust width for better fitting of multiple subplots
        showlegend=False
    )
    
    return fig




def plot_scr_ratio(final_data):
    fig = px.histogram(final_data, x='SCR_Ratio', title='Distribution of SCR Ratio', nbins=30)
    fig.update_layout(
        xaxis_title='SCR Ratio (%)',
        yaxis_title='Frequency',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig

def plot_time_series(df, kpi, lob=None):
    """
    Plot a time series for a given KPI using Plotly. Optionally filter by Line of Business (LOB).

    Parameters:
    - df (pd.DataFrame): Preprocessed DataFrame.
    - kpi (str): KPI column name to plot.
    - lob (str, optional): Specific Line of Business to filter. Defaults to None (all LOBs).

    Returns:
    - fig (Plotly figure)
    """
    if lob:
        lob_df = df[df['Line_of_Business'] == lob]
        fig = px.line(lob_df, x='Date', y=kpi, title=f'{kpi.replace("_", " ")} Over Time for {lob} Insurance')
    else:
        fig = px.line(df, x='Date', y=kpi, color='Line_of_Business', title=f'{kpi.replace("_", " ")} Over Time for All Lines of Business')

    # Update axes labels and title
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=kpi.replace('_', ' '),
        legend_title='Line of Business',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    
    return fig

def plot_combined_ratio(final_data):
    fig = px.histogram(final_data, x='Combined_Ratio', color='Line_of_Business',
                       title='✔️ <b>Distribution of Combined Ratios by Line of Business</b>',
                       nbins=50)
    fig.update_layout(
        xaxis_title='Combined Ratio (%)',
        yaxis_title='Frequency',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig

def plot_loss_ratio_vs_nep(final_data):
    fig = px.scatter(final_data, x='NEP', y='Loss_Ratio', color='Line_of_Business',
                     title='✔️ <b>Loss Ratio vs NEP</b>')
    fig.update_layout(
        xaxis_title='Net Earned Premium (NEP)',
        yaxis_title='Loss Ratio (%)',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig

def plot_nep_vs_gdp(final_data):
    fig = px.scatter(final_data, x='GDP_Growth_Rate', y='NEP', color='Line_of_Business',
                     title='✔️ <b>NEP vs GDP Growth Rate</b>')
    fig.update_layout(
        xaxis_title='GDP Growth Rate',
        yaxis_title='Net Earned Premium (NEP)',
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig

def plot_policy_count(final_data):
    fig = px.line(final_data, x='Date', y='Policy_Count', color='Line_of_Business',
                  title='✔️ <b>Policy Count Over Time by Line of Business</b>')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Policy Count',
        xaxis_showgrid=False,  # Remove x-axis grid
        yaxis_showgrid=False   # Remove y-axis grid
    )
    return fig
