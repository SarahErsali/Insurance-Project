import pandas as pd
import numpy as np



# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------
# Function to simulate economic indicators with crises on a monthly basis
# ----------------------------

def simulate_economic_indicators(start_date='2008-01-01', end_date='2024-12-31'):
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')  # Monthly frequency
    n = len(dates)
    data = pd.DataFrame({'Date': dates})
    
    # Base economic indicators
    gdp_growth_base = 0.005  # 0.5% per month (~6% annually)
    inflation_base = 0.0012   # 0.12% per month (~1.44% annually)
    unemployment_base = 0.004  # 0.4% per month (~4.8% annually)
    interest_rate_base = 0.001  # 0.1% per month (~1.2% annually)
    equity_return_base = 0.004  # Approximate monthly return of ~4.8% annually
    
    # Initialize arrays
    gdp_growth = np.full(n, gdp_growth_base)
    inflation_rate = np.full(n, inflation_base)
    unemployment_rate = np.full(n, unemployment_base)
    interest_rate = np.full(n, interest_rate_base)
    equity_return = np.full(n, equity_return_base)
    
    # Define crisis periods with monthly granularity
    crisis_periods = {
        'financial_crisis': {
            'start': '2008-01-01',
            'end': '2009-12-31',
            'gdp_shock': -0.015,  # -1.5% per month
            'equity_shock': -0.05,  # -5% per month
            'unemployment_shock': 0.005,  # +0.5% per month
        },
        'european_debt_crisis': {
            'start': '2011-01-01',
            'end': '2012-12-31',
            'gdp_shock': -0.005,  # -0.5% per month
            'equity_shock': -0.03,  # -3% per month
            'unemployment_shock': 0.002,  # +0.2% per month
        },
        'covid_pandemic': {
            'start': '2020-01-01',
            'end': '2021-12-31',
            'gdp_shock': -0.02,  # -2% per month
            'equity_shock': -0.04,  # -4% per month
            'unemployment_shock': 0.006,  # +0.6% per month
        },
    }
    
    # Apply shocks during crisis periods
    for crisis in crisis_periods.values():
        mask = (data['Date'] >= crisis['start']) & (data['Date'] <= crisis['end'])
        idx_range = data[mask].index
        
        gdp_growth[idx_range] += crisis['gdp_shock']
        equity_return[idx_range] += crisis['equity_shock']
        unemployment_rate[idx_range] += crisis['unemployment_shock']
    
    # Add randomness
    gdp_growth += np.random.normal(0, 0.002, n)  # Reduced volatility for monthly data
    inflation_rate += np.random.normal(0, 0.0005, n)
    unemployment_rate += np.random.normal(0, 0.001, n)
    interest_rate += np.random.normal(0, 0.001, n)
    equity_return += np.random.normal(0, 0.02, n)
    
    # Assign to dataframe with clipping to ensure realistic ranges
    data['GDP_Growth_Rate'] = gdp_growth.clip(min=-0.05, max=0.05)  # -5% to +5%
    data['Inflation_Rate'] = inflation_rate.clip(min=-0.01, max=0.05)  # -1% to +5%
    data['Unemployment_Rate'] = unemployment_rate.clip(min=0, max=0.15)  # 0% to 15%
    data['Interest_Rate'] = interest_rate.clip(min=-0.01, max=0.05)  # -1% to +5%
    data['Equity_Return'] = equity_return.clip(min=-0.5, max=0.5)  # -50% to +50%
    
    return data

# ----------------------------
# Function to simulate insurance data by line of business on a monthly basis
# ----------------------------

def simulate_insurance_data(data, line_of_business):
    n = len(data)
    
    # Parameters by line of business
    params = {
        'life': {
            'avg_premium': 1200,
            'expense_ratio': 0.18,      # 18%
            'asset_mix': {'bonds': 0.75, 'equities': 0.1, 'real_estate': 0.1, 'cash': 0.05},
            'cession_rate': 0.10,
            'underwriting_risk_factor': 0.12,
            'operational_risk_factor': 0.03,
            'base_policy_count': 30000,
            'investment_return_adjustment': 0.05,  # Additional investment return
            'per_policy_claim': 680,  # Calibrated per-policy claim amount
        },
        'motor': {
            'avg_premium': 500,
            'expense_ratio': 0.23,      # 23%
            'asset_mix': {'bonds': 0.6, 'equities': 0.2, 'real_estate': 0.1, 'cash': 0.1},
            'cession_rate': 0.30,
            'underwriting_risk_factor': 0.15,
            'operational_risk_factor': 0.025,
            'base_policy_count': 60000,
            'per_policy_claim': 328,  # Calibrated per-policy claim amount
        },
        'property': {
            'avg_premium': 800,
            'expense_ratio': 0.18,      # 18%
            'asset_mix': {'bonds': 0.5, 'equities': 0.25, 'real_estate': 0.15, 'cash': 0.1},
            'cession_rate': 0.30,
            'underwriting_risk_factor': 0.18,
            'operational_risk_factor': 0.025,
            'base_policy_count': 40000,
            'per_policy_claim': 560,  # Calibrated per-policy claim amount
            'storm_impact_factor': 1.5,  #### New: Impact multiplier during STORM scenarios

        },
        'health': {
            'avg_premium': 1000,
            'expense_ratio': 0.20,      # 20%
            'asset_mix': {'bonds': 0.65, 'equities': 0.15, 'real_estate': 0.1, 'cash': 0.1},
            'cession_rate': 0.15,
            'underwriting_risk_factor': 0.14,
            'operational_risk_factor': 0.028,
            'base_policy_count': 55000,
            'investment_return_adjustment': 0.07,  # Additional investment return
            'per_policy_claim': 635,  # Calibrated per-policy claim amount
        },
    }
    
    p = params[line_of_business]
    
    # --------------------------------------
    # Policy Count and Premiums
    # --------------------------------------
    
    # Initialize Policy_Count as a list to store dynamic values
    policy_counts = []
    
    # Initialize the first month's Policy_Count
    initial_policy_count = p['base_policy_count']
    policy_counts.append(initial_policy_count)
    
    # Define sensitivity parameters
    base_lapse_rate = 0.02  # 2%
    lapse_rate_sensitivity = 0.18  # 18% - increased sensitivity
    base_acquisition_rate = 0.03  # 3%
    acquisition_rate_sensitivity = 0.25  # 25% - increased sensitivity
    
    # Define min and max policy counts to prevent extreme fluctuations
    min_policy_count = p['base_policy_count'] * 0.5  # 50% of base
    max_policy_count = p['base_policy_count'] * 2.0  # 200% of base
    
    for t in range(1, n):
        previous_policy_count = policy_counts[-1]
        
        # Use previous month's GDP Growth Rate
        gdp_growth = data.at[t-1, 'GDP_Growth_Rate']
        
        # Calculate Adjusted Lapse Rate
        adjusted_lapse_rate = base_lapse_rate - (lapse_rate_sensitivity * gdp_growth)
        adjusted_lapse_rate = max(adjusted_lapse_rate, 0.005)  # Set a floor to prevent lapses from being too low
        
        # Calculate Policy_Lapses (Deterministic)
        policy_lapses = previous_policy_count * adjusted_lapse_rate
        
        # Calculate New_Policies (Deterministic)
        new_policies = previous_policy_count * (base_acquisition_rate + (acquisition_rate_sensitivity * gdp_growth))
        new_policies = max(new_policies, 0)  # Prevent negative new policies
        
        # Update Policy_Count (Deterministic)
        updated_policy_count = previous_policy_count * (1 - adjusted_lapse_rate) + new_policies
        
        # Implement lower and upper bounds
        updated_policy_count = np.clip(updated_policy_count, min_policy_count, max_policy_count)
        
        policy_counts.append(updated_policy_count)
    
    # Assign Policy_Count to data
    data['Policy_Count'] = policy_counts
    
    # Adjusted Average Premium with variability
    avg_premium_variation = np.random.uniform(0.95, 1.05, n)
    data['Avg_Premium'] = p['avg_premium'] * avg_premium_variation
    
    # Gross Written Premiums
    data['GWP'] = data['Policy_Count'] * data['Avg_Premium']
    
    # Net Earned Premiums
    data['NEP'] = data['GWP'] * 0.9  # Assuming 90% earned
    
    # --------------------------------------
    # Claims Incurred
    # --------------------------------------
    
    # Introduce an independent Claims_Variability_Factor
    data['Claims_Variability_Factor'] = np.random.uniform(0.95, 1.05, n)  # Reduced variability
    
    # Introduce an independent factor based on Inflation Rate
    data['Independent_Claims_Factor'] = 1 + (data['Inflation_Rate'] * 2)  # Example influence
    
    # Calculate Claims_Incurred based on Policy_Count and per_policy_claim
    data['Claims_Incurred'] = data['Policy_Count'] * p['per_policy_claim'] * data['Claims_Variability_Factor'] * data['Independent_Claims_Factor']

    ##### New: Storm scenario implementation for property line of business
    if line_of_business == 'property':
        # Define storm periods and impact factor
        storm_periods = {
            'storm_event_1': {
                'start': '2020-06-15',
                'end': '2020-09-13',
                'impact_factor': p['storm_impact_factor']  # Increase in claims during storm event
            },
            'storm_event_2': {
                'start': '2022-07-20',
                'end': '2022-09-02',
                'impact_factor': p['storm_impact_factor']  # Increase in claims during storm event
            },
            'storm_event_3': {
                'start': '2023-05-08',
                'end': '2023-07-30',
                'impact_factor': p['storm_impact_factor']  # Increase in claims during storm event
            }
        }
    
        # Apply storm impact
        for storm in storm_periods.values():
            storm_mask = (data['Date'] >= storm['start']) & (data['Date'] <= storm['end'])
            data.loc[storm_mask, 'Claims_Incurred'] *= storm['impact_factor']

    
    # Limit Claims_Incurred to a maximum of 120% of NEP to prevent extreme Loss Ratios
    data['Claims_Incurred'] = data['Claims_Incurred'].clip(upper=1.2 * data['NEP'])
    
    # --------------------------------------
    # Expenses
    # --------------------------------------
    
    # Expenses with slight variation and economic influence
    expense_variation = 1 + (data['GDP_Growth_Rate'] * 0.1)  # Slight variation based on GDP
    expense_random_variation = np.random.uniform(0.98, 1.02, n)
    data['Expenses'] = data['NEP'] * p['expense_ratio'] * expense_random_variation * expense_variation
    
    # --------------------------------------
    # Reinsurance Recoveries
    # --------------------------------------
    
    data['Reinsurance_Recoveries'] = data['Claims_Incurred'] * p['cession_rate']
    
    # Net Claims
    data['Net_Claims'] = data['Claims_Incurred'] - data['Reinsurance_Recoveries']
    
    # --------------------------------------
    # Total Assets and Asset Allocation
    # --------------------------------------
    
    # Assuming Total Assets are a multiple of GWP with some variation
    data['Total_Assets'] = data['GWP'] * np.random.uniform(1.3, 1.7, n)
    
    # Asset Allocation based on asset_mix
    data['Bonds'] = data['Total_Assets'] * p['asset_mix']['bonds']
    data['Equities'] = data['Total_Assets'] * p['asset_mix']['equities']
    data['Real_Estate'] = data['Total_Assets'] * p['asset_mix']['real_estate']
    data['Cash'] = data['Total_Assets'] * p['asset_mix']['cash']
    
    # --------------------------------------
    # Investment Returns
    # --------------------------------------
    
    # Adjust investment returns for Life and Health Insurance
    if line_of_business in ['life', 'health']:
        investment_return_adjustment = p.get('investment_return_adjustment', 0)
    else:
        investment_return_adjustment = 0
    
    data['Bond_Returns'] = data['Bonds'] * (data['Interest_Rate'] + np.random.normal(0.003 + investment_return_adjustment, 0.001, n))  # Slightly above base
    data['Equity_Returns'] = data['Equities'] * (data['Equity_Return'] + investment_return_adjustment)
    data['Real_Estate_Returns'] = data['Real_Estate'] * np.random.normal(0.01, 0.02, n)  # +/-1% to 2%
    data['Cash_Returns'] = data['Cash'] * np.random.normal(0.001, 0.0005, n)  # ~0.1%
    
    data['Total_Investment_Returns'] = (
        data['Bond_Returns'] +
        data['Equity_Returns'] +
        data['Real_Estate_Returns'] +
        data['Cash_Returns']
    )
    
    # --------------------------------------
    # Risk Components and SCR
    # --------------------------------------
    
    # Underwriting Risk Component
    data['Underwriting_Risk'] = data['Net_Claims'] * p['underwriting_risk_factor']
    
    # Market Risk Component
    data['Market_Risk'] = (
        data['Bonds'] * 0.015 +    # 1.5% of Bonds
        data['Equities'] * 0.08 +   # 8% of Equities
        data['Real_Estate'] * 0.06  # 6% of Real Estate
    )
    
    # Credit Risk Component
    data['Credit_Risk'] = data['Bonds'] * 0.01  # 1% of Bonds
    
    # Operational Risk Component
    data['Operational_Risk'] = data['NEP'] * p['operational_risk_factor']
    
    # Solvency Capital Requirement (SCR) using square root of sum of squares
    data['SCR'] = np.sqrt(
        data['Underwriting_Risk']**2 +
        data['Market_Risk']**2 +
        data['Credit_Risk']**2 +
        data['Operational_Risk']**2
    )
    
    # --------------------------------------
    # Eligible Own Funds and SCR Ratio
    # --------------------------------------
    
    # Dynamic Eligible Own Funds
    # Base multiplier is around 1.8 to 2.2, adjusted based on NEP, Claims, and Economic Conditions
    base_multiplier = np.random.uniform(1.8, 2.2, n)
    
    # Adjust multiplier based on NEP and Claims
    # Higher NEP and Claims could imply higher Eligible Own Funds needs
    # Similarly, economic downturns may reduce the multiplier
    economic_condition_multiplier = 1 + (data['GDP_Growth_Rate'] / 10)  # Small adjustment, e.g., GDP -0.05 -> 0.995
    
    # Final multiplier with some randomness
    final_multiplier = base_multiplier * economic_condition_multiplier * np.random.uniform(0.95, 1.05, n)
    
    data['Eligible_Own_Funds'] = data['SCR'] * final_multiplier
    
    # SCR Ratio
    data['SCR_Ratio'] = (data['Eligible_Own_Funds'] / data['SCR']) * 100  # Percentage
    
    # --------------------------------------
    # Stress Adjustments
    # --------------------------------------
    
    # SCR under stress: increase SCR by 20% during economic downturns
    data['SCR_Stress'] = data['SCR'] * np.where(data['GDP_Growth_Rate'] < 0, 1.20, 1)
    
    # Eligible Own Funds under stress: decrease Eligible Own Funds by 10% during downturns
    data['Eligible_Own_Funds_Stress'] = data['Eligible_Own_Funds'] * np.where(data['GDP_Growth_Rate'] < 0, 0.90, 1)
    
    # SCR Ratio under stress
    data['SCR_Ratio_Stress'] = (data['Eligible_Own_Funds_Stress'] / data['SCR_Stress']) * 100
    
    # --------------------------------------
    # Ratios
    # --------------------------------------
    
    # Loss Ratio
    data['Loss_Ratio'] = (data['Claims_Incurred'] / data['NEP']) * 100  # Percentage
    
    # Expense Ratio
    data['Expense_Ratio'] = (data['Expenses'] / data['NEP']) * 100  # Percentage
    
    # Combined Ratio
    data['Combined_Ratio'] = data['Loss_Ratio'] + data['Expense_Ratio']  # Percentage
    
    # Cap Combined Ratio at 150% to prevent unrealistic values
    data['Combined_Ratio'] = data['Combined_Ratio'].clip(upper=150)
    
    # --------------------------------------
    # Data Integrity and Finalization
    # --------------------------------------
    
    # Ensure data correctness by clipping negative values to zero
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # Exclude economic indicators from clipping
    exclude_columns = ['GDP_Growth_Rate', 'Inflation_Rate', 'Unemployment_Rate', 'Interest_Rate', 'Equity_Return']
    cols_to_clip = numeric_cols.difference(exclude_columns)
    data[cols_to_clip] = data[cols_to_clip].clip(lower=0)
    
    # Add Line of Business
    data['Line_of_Business'] = line_of_business.capitalize()
    
    return data

# ----------------------------
# Generate data for each line of business and combine
# ----------------------------

def generate_all_lob_data():
    # Simulate economic indicators
    economic_data = simulate_economic_indicators()
    
    lines_of_business = ['life', 'motor', 'property', 'health']
    all_data = []
    
    for lob in lines_of_business:
        lob_data = simulate_insurance_data(economic_data.copy(), lob)
        all_data.append(lob_data)
    
    # Combine all data
    final_data = pd.concat(all_data, ignore_index=True)
    
    return final_data

# Generate final data
final_data = generate_all_lob_data()

#------------------------------------------------------------------------------
property_data = final_data[final_data['Line_of_Business'] == 'Property'].reset_index(drop = True)

# Feature Engineering (adding temporal features)
property_data['Year'] = property_data['Date'].dt.year
property_data['Month'] = property_data['Date'].dt.month
property_data['Quarter'] = property_data['Date'].dt.quarter

# Make a copy of property_data to modify and drop unnecessary columns
property_data_model = property_data.copy()

# Since I'm focusing on property LOB, I drop 'Line_of_Business'
property_data_model.drop(['Line_of_Business'], axis = 1, inplace = True)


# Define crisis periods with monthly granularity
crisis_periods = {
    'financial_crisis': {
        'start': '2008-01-01',
        'end': '2009-12-31',
        'gdp_shock': -0.015,  # -1.5% per month
        'equity_shock': -0.05,  # -5% per month
        'unemployment_shock': 0.005,  # +0.5% per month
    },
    'european_debt_crisis': {
        'start': '2011-01-01',
        'end': '2012-12-31',
        'gdp_shock': -0.005,  # -0.5% per month
        'equity_shock': -0.03,  # -3% per month
        'unemployment_shock': 0.002,  # +0.2% per month
    },
    'covid_pandemic': {
        'start': '2020-01-01',
        'end': '2021-12-31',
        'gdp_shock': -0.02,  # -2% per month
        'equity_shock': -0.04,  # -4% per month
        'unemployment_shock': 0.006,  # +0.6% per month
    },
}


# Creating a full date range for econ_data that covers all relevant dates
start_date = property_data_model['Date'].min()
end_date = property_data_model['Date'].max()

# Create a date range with the same frequency as the data (monthly)
econ_data = pd.DataFrame({'Date': pd.date_range(start = start_date, end = end_date, freq = 'ME')})

econ_data['Is_Crisis'] = 0

# Set Is_Crisis to 1 for crisis periods
for period in crisis_periods.values():
    mask = (econ_data['Date'] >= period['start']) & (econ_data['Date'] <= period['end'])
    econ_data.loc[mask, 'Is_Crisis'] = 1


# Merge this corrected econ_data with final_data to get the Is_Crisis flag in final_data
property_data_model = property_data_model.merge(econ_data[['Date', 'Is_Crisis']], on='Date', how='left')


columns_to_check = [
    'GDP_Growth_Rate', 'Inflation_Rate', 'Unemployment_Rate', 'Interest_Rate',
    'Equity_Return', 'Expenses', 'Month', 'Quarter',
    'SCR', 'Is_Crisis'
]

# Create a new DataFrame with only the selected features
property_data_feature_selected = property_data_model[columns_to_check]


# Function to detect outliers based on IQR with crisis handling
def detect_outliers_iqr(data, column, is_crisis):
    """
    Detect outliers using the IQR method for crisis or non-crisis periods.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column to check for outliers.
    is_crisis (bool): Flag to indicate if checking crisis (True) or non-crisis (False) period.
    
    Returns:
    pd.DataFrame: DataFrame with detected outliers.
    """
    # Separate crisis and non-crisis data
    if is_crisis:
        data = data[data['Is_Crisis'] == 1]
    else:
        data = data[data['Is_Crisis'] == 0]
    
    # Check if data is available for outlier detection
    if data.empty:
        return pd.DataFrame()  # Return empty DataFrame if no data
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range
    
    # Define bounds for outliers
    lower_bound = Q1 - 2 * IQR  # The 1.5 value for boundary is too restrictive, so 2 is better
    upper_bound = Q3 + 2 * IQR
    
    # Return rows where the column value is outside the bounds
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Detect outliers in each selected column for crisis and non-crisis periods
outliers = {'crisis': {}, 'non_crisis': {}}
for col in columns_to_check:
    if col in property_data_feature_selected.columns:
        # Outliers in crisis periods
        outliers['crisis'][col] = detect_outliers_iqr(property_data_feature_selected, col, is_crisis=True)
        print(f"Outliers detected in crisis period for {col}: {outliers['crisis'][col].shape[0]}")
        
        # Outliers in non-crisis periods
        outliers['non_crisis'][col] = detect_outliers_iqr(property_data_feature_selected, col, is_crisis=False)
        print(f"Outliers detected in non-crisis period for {col}: {outliers['non_crisis'][col].shape[0]}")
    else:
        print(f"Column '{col}' not found in property_data_feature_selected.")

# Display the outliers detected in 'Claims_Incurred' for crisis periods
if 'Claims_Incurred' in outliers['crisis']:
    display(outliers['crisis']['Claims_Incurred'])
else:
    print("No outliers detected in 'Claims_Incurred' during crisis periods or column not found.")

# Display the outliers detected in 'Claims_Incurred' for non-crisis periods
if 'Claims_Incurred' in outliers['non_crisis']:
    display(outliers['non_crisis']['Claims_Incurred'])
else:
    print("No outliers detected in 'Claims_Incurred' during non-crisis periods or column not found.")


# Counts of 'Is_Crisis' column
print("Counts of 'Is_Crisis':\n", property_data_feature_selected['Is_Crisis'].value_counts())

# Check distribution of created outlier flags
for col in columns_to_check:
    outlier_col = f'{col}_Is_Outlier'
    if outlier_col in property_data_feature_selected.columns:
        print(f"Counts of '{outlier_col}':\n", property_data_feature_selected[outlier_col].value_counts())



# Include 'Date' if you want to use it for filtering
selected_features_with = columns_to_check + ['Date', 'Claims_Incurred']

# Ensure 'Date' column is included
property_data_feature_selected = property_data_model[selected_features_with]


### Note: The XGBoost library does not support datetime64[ns] data types, therefore Date has to be dropped from X

# Training set: 2008-2019
train_data = property_data_feature_selected[(property_data_feature_selected['Date'].dt.year >= 2008) & (property_data_feature_selected['Date'].dt.year <= 2019)]

# Validation set: 2020-2022
val_data = property_data_feature_selected[(property_data_feature_selected['Date'].dt.year >= 2020) & (property_data_feature_selected['Date'].dt.year <= 2022)]

# Blind test set: 2023-2024
blind_test_data = property_data_feature_selected[(property_data_feature_selected['Date'].dt.year >= 2023) & (property_data_feature_selected['Date'].dt.year <= 2024)]

X_train = train_data.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore') 
y_train = train_data['Claims_Incurred']

X_val = val_data.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
y_val = val_data['Claims_Incurred']

X_blind_test = blind_test_data.drop(['Claims_Incurred', 'Date'], axis=1, errors='ignore')
y_blind_test = blind_test_data['Claims_Incurred']
y_blind_test.index = blind_test_data['Date']  # This ensures that y_blind_test has a datetime index instead of an integer index

# Combine Training and Validation Sets
X_combined = pd.concat([X_train, X_val], ignore_index=True)
y_combined = pd.concat([y_train, y_val], ignore_index=True)

# Export the variables
__all__ = ['X_train', 'y_train', 'X_val', 'y_val', 'X_combined', 'y_combined', 'X_blind_test', 'y_blind_test']