from dash import dcc, html
from components.data import final_data
from components.tab2_plots import (plot_policy_count, plot_nep_vs_gdp, plot_loss_ratio_vs_nep,
                                   plot_combined_ratio, plot_time_series,
                                   plot_economic_histograms, plot_insurance_histograms)


def calculate_statistics(final_data):
    # List to hold HTML elements for stats
    stats = []
    
    # Correlation values and averages for each Line of Business (LOB)
    lobs = final_data['Line_of_Business'].unique()
    for lob in lobs:
        lob_data = final_data[final_data['Line_of_Business'] == lob]
        
        # Correlation between NEP and GDP Growth Rate
        corr_nep_gdp = lob_data['NEP'].corr(lob_data['GDP_Growth_Rate'])
        corr_nep_loss_ratio = lob_data['NEP'].corr(lob_data['Loss_Ratio'])
        
        # Averages of ratios
        avg_loss_ratio = lob_data['Loss_Ratio'].mean()
        avg_expense_ratio = lob_data['Expense_Ratio'].mean()
        avg_combined_ratio = lob_data['Combined_Ratio'].mean()
        over_100_percent = (lob_data['Combined_Ratio'] > 100).mean() * 100
        
        # Append the statistics to the list
        stats.append(html.Div([
            html.H4(f"{lob} Insurance:"),
            html.P(f"Correlation between NEP and GDP Growth Rate: {corr_nep_gdp:.2f}"),
            html.P(f"Correlation between NEP and Loss Ratio: {corr_nep_loss_ratio:.2f}"),
            html.P(f"Average Loss Ratio: {avg_loss_ratio:.2f}%"),
            html.P(f"Average Expense Ratio: {avg_expense_ratio:.2f}%"),
            html.P(f"Average Combined Ratio: {avg_combined_ratio:.2f}%"),
            html.P(f"Percentage of periods with Combined Ratio > 100%: {over_100_percent:.2f}%")
        ], style={'padding': '10px', 'margin': '10px', 'width': '45%'}))  # Removed the border style
    
    return stats


def render_tab2():
    # Reordered Plots
    fig_economic_histograms = plot_economic_histograms(final_data, ['GDP_Growth_Rate', 'Inflation_Rate', 'Unemployment_Rate', 'Interest_Rate', 'Equity_Return'])
    fig_insurance_histograms = plot_insurance_histograms(final_data, ['Policy_Count', 'Avg_Premium', 'Claims_Incurred', 'Expenses', 'SCR_Ratio'])
    fig_claims_incurred = plot_time_series(final_data, 'Claims_Incurred')
    fig_nep = plot_time_series(final_data, 'NEP')
    fig_combined_ratio = plot_combined_ratio(final_data)
    fig_loss_ratio_vs_nep = plot_loss_ratio_vs_nep(final_data)
    fig_nep_vs_gdp = plot_nep_vs_gdp(final_data)
    fig_policy_count = plot_policy_count(final_data)

    # Calculate statistics (correlations and averages)
    stats_elements = calculate_statistics(final_data)

    return html.Div([
        html.H2("Key metrics and correlation analysis of the dataset based on line of business", style={'textAlign': 'left','marginBottom': '20px', 'fontSize': '30px'}),
        
        # Key insights
        html.Ul([
            html.Li("Across all lines of insurance, there is a consistent moderate positive correlation between NEP and GDP Growth Rate (around 0.31 - 0.34). This suggests that economic growth generally benefits the insurance industry by increasing earned premiums."),
            html.Li("The negative correlation between NEP and Loss Ratio is present but weak for all lines, with the most notable relationship in life insurance (- 0.27). This implies that while higher NEP tends to slightly reduce the loss ratio, profitability is also influenced by other factors specific to each insurance line."),
            html.Li("Life and Health Insurance show the strongest profitability and stability, with no periods exceeding a combined ratio of 100%."),
            html.Li("Motor and Property Insurance have marginal profitability, with some periods showing losses. This could indicate higher risk and susceptibility to fluctuations, possibly due to unpredictable claims like accidents (for motor) and natural disasters (for property).")
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),

        # Display the calculated statistics before the plots, arranged 2 by 2
        html.Div(stats_elements, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),  # Flexbox for 2-by-2 layout

        # Reordered Plots
        dcc.Graph(figure=fig_economic_histograms),
        html.Ul([
            html.Li([html.B("GDP Growth Rate: "), 
                    "The GDP growth rate mostly hovers around 0.005 (0.5% growth), with a spike showing over 120 months having a growth rate near this value. This distribution is skewed to the left, indicating that economic recessions are less frequent but do occur, with significant GDP drops during those periods."]),
            html.Li([html.B("Inflation Rate: "), 
                    "The inflation rate represents a normal distribution and is mostly centered around 0.0012 (0.12%) per month, indicating low and stable inflation."]),
            html.Li([html.B("Unemployment Rate: "), 
                    "The unemployment rate is clustered between 0.004 (0.4%) and 0.006 (0.6%), with a peak around 0.004 (4%). The distribution is skewed to the right, indicating that higher unemployment rates occur less frequently, though significant jumps happen during economic downturns."]),
            html.Li([html.B("Interest Rate: "), 
                    "The interest rate is mostly centered around 0.001 (1%), with a nearly normal distribution. There are some slight fluctuations, but the rates are relatively stable and closely follow a normal distribution, suggesting that the interest rates are managed or adjusted in response to the economic conditions in the simulation."]),
            html.Li([html.B("Equity Return: "), 
                    "The equity return is mostly centered around 0. The distribution suggests a mix of positive and negative returns, typical of equity markets, with both positive and negative shocks."])
        ], style={'textAlign': 'left', 'lineHeight': '2'}),


        dcc.Graph(figure=fig_insurance_histograms),
        html.Ul([
            html.Li([
                html.B("Policy Count: "),
                "There are distinct spikes in the data, especially around 60,000 and 120,000. Policy count distribution is uneven, with specific clusters of policies being issued at certain months."
            ]),
            html.Li([
                html.B("Average Premium: "),
                "The multimodal distribution suggests that the insurance company may offer various product lines with differing premium rates."
            ]),
            html.Li([
                html.B("Claims Incurred: "),
                "The distribution suggests that most months have moderately incurred claims, but some periods saw exceptionally high claim amounts, which could be related to external factors like natural disasters."
            ]),
            html.Li([
                html.B("Expenses: "),
                "Expenses are clustered around specific values, but there are periods where expenses spiked, possibly due to crises or increased claims. This suggests that expenses are somewhat variable but show predictable patterns."
            ]),
            html.Li([
                html.B("SCR Ratio: "),
                "The company appears to have a stable SCR ratio, which indicates strong financial health. There are no extreme outliers, meaning the company likely has good capital management practices in place."
            ]),
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),



        dcc.Graph(figure=fig_claims_incurred),
        html.Ul([
            html.Li([
                html.B("Life, motor, and property insurance: "),
                "All show growth over time, but the patterns and trends differ."
            ]),
            html.Li([
                html.B("Property insurance: "),
                "The sharp increases in claims incurred, show considerable volatility, especially around 2020 and 2023, due to natural disasters."
            ]),
            html.Li([
                html.B("Life insurance: "),
                "Exhibits the highest and most consistent growth in claims incurred, reflecting its long-term nature and larger payouts. The upward trend is steady, with no significant disruptions or spikes."
            ]),
            html.Li([
                html.B("Conclusion: "),
                "The analysis suggests that property insurance carries higher exposure to event-driven risks in this dataset. Property insurance's susceptibility to external events underscores the need for robust risk management strategies to mitigate such spikes in claims. For this reason, this line of business has been investigated in this project."
            ]),
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),


        dcc.Graph(figure=fig_nep),
        html.Ul([
            html.Li([
                html.B("Life, Motor, and Property Insurance: "),
                "Show a consistent upward trend, indicating growth in premiums over time. The sharp increases in life insurance suggest a growing customer base or larger premiums being paid by policyholders."
            ]),
            html.Li([
                html.B("Health Insurance: "),
                "The significant increase in NEP, especially post-2016, indicates a growing importance in the health insurance market. This could be due to increased demand for healthcare coverage or changes in regulations affecting health insurance."
            ]),
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),


        dcc.Graph(figure=fig_combined_ratio),
        html.Ul([
            html.Li([
                html.B("The Combined Ratio: "),
                "is a key metric in insurance that represents the sum of the loss ratio and expense ratio. A combined ratio under 100% indicates profitability, while a ratio above 100% indicates losses."
            ]),
            html.Li([
                html.B("Motor Insurance: "),
                "The combined ratio for motor insurance is distributed between 90% and 100%, indicating tight margins in profitability. There's little to no combined ratio data beyond 100%, suggesting motor insurance generally avoids losses."
            ]),
            html.Li([
                html.B("Property Insurance: "),
                "The combined ratio for property insurance has a wider range, from 80% to slightly above 130%. Most of the distribution is centered around 90% to 100%, which indicates a decent level of profitability, but there are also instances of losses with combined ratios exceeding 100%. The presence of combined ratios above 120% suggests some challenging periods for property insurance, possibly due to high claims in certain crisis periods or natural disasters."
            ]),
            html.Li([
                html.B("Health Insurance: "),
                "Health insurance shows a distribution mainly between 85% and 95%. There’s a clear concentration of combined ratios around 90%, indicating consistent profitability."
            ]),
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),


        dcc.Graph(figure=fig_loss_ratio_vs_nep),
        html.Ul([
            html.Li([
                html.B("Scatter Plot: "),
                "The scatter plot shows the relationship between Loss Ratio and Net Earned Premium (NEP) across different lines of business."
            ]),
            html.Li([
                html.B("Property Insurance: "),
                "Property insurance shows a wider range of loss ratios compared to other lines of business. NEP varies between 30M and 60M, but some outliers in the loss ratio exceed 100% (above 120% at times), indicating significant losses during certain periods. This may suggest that property insurance faces more volatility, due to unpredictable natural disasters, which cause sudden spikes in claims. While many points cluster around a loss ratio of 80% to 90%, indicating profitability, the higher loss ratios suggest some challenging periods."
            ]),
            html.Li([
                html.B("Life Insurance: "),
                "Life insurance has relatively low loss ratios, mainly concentrated between 60% and 75%, indicating strong profitability. As NEP increases from 30M to over 60M, loss ratios stay consistently low. This suggests life insurance is efficiently managing claims, with consistent profitability even as the business grows (higher NEP)."
            ]),
            html.Li([
                html.B("Motor Insurance: "),
                "Motor insurance shows more fluctuation in the loss ratio, mostly ranging from 70% to 90%. The NEP for motor insurance hovers between 30M and 45M, and as NEP increases, the loss ratio slightly decreases, indicating some efficiency improvements with scale. Despite some fluctuations, motor insurance appears to operate within a manageable loss ratio range, suggesting moderate profitability."
            ]),
            html.Li([
                html.B("Health Insurance: "),
                "Health insurance shows a consistently low loss ratio between 60% and 80%, suggesting solid profitability. NEP for health insurance extends from 60M to over 100M, and as NEP increases, the loss ratio remains stable. This indicates that health insurance is highly efficient in managing claims and maintains profitability across various business scales."
            ]),
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),

        dcc.Graph(figure=fig_nep_vs_gdp),
        html.Ul([
            html.Li([
                html.B("Scatter Plot: "),
                "The scatter plot shows the relationship between Net Earned Premium (NEP) and GDP Growth Rate for different lines of business."
            ]),
            html.Li([
                html.B("Property Insurance: "),
                "Property insurance shows a greater spread in NEP, ranging between 35M and 60M. During periods of negative GDP growth (left side of the plot), there are some extreme outliers indicating potentially high variability in premiums during economic downturns. Positive GDP growth shows more stability for NEP, suggesting that the property line of business is more sensitive to economic downturns but recovers during growth periods."
            ]),
            html.Li([
                html.B("Life Insurance: "),
                "Life insurance shows a wide spread in NEP ranging from 40M to 60M across all GDP growth rates, indicating that NEP is relatively stable despite fluctuations in GDP. The overall distribution of NEP suggests that life insurance is not heavily impacted by changes in GDP, as the premiums remain steady even during economic downturns (negative GDP growth)."
            ]),
            html.Li([
                html.B("Motor Insurance: "),
                "Motor insurance generally has lower NEP (around 30M to 45M), and there is a slight clustering around negative GDP growth. As the GDP growth improves (toward positive territory), NEP for motor insurance appears to stabilize, indicating that economic recovery helps maintain stable premiums."
            ]),
            html.Li([
                html.B("Health Insurance: "),
                "Health insurance has the highest NEP compared to other lines of business, ranging from 80M to 100M. NEP remains relatively stable across both negative and positive GDP growth, indicating strong resilience to economic changes. Even during sharp downturns in GDP growth, health insurance maintains consistent premiums, likely due to the essential nature of health coverage."
            ]),
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),

        dcc.Graph(figure=fig_policy_count),
        html.Ul([
            html.Li([
                html.B("Policy Count Growth: "),
                "The policy count growth shows that motor and health insurance are the most dynamic sectors in terms of acquiring new policies, while life and property insurance have slower but steady growth."
            ])
        ], style={'textAlign': 'left', 'lineHeight': '2', 'marginBottom': '30px'}),

        ], style={'padding': '20px', 'paddingLeft': '3cm', 'paddingRight': '3cm'})  # Added 2cm padding on left and right

