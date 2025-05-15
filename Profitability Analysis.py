import requests
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

bea_api_key = "8E1511D6-3C18-478B-BE27-2D8A73B26FF8"

# Get National Income and Product Accounts Table Data
def get_nipa_table_data(bea_api_key, table_id, line_number, frequency="A", year="X"):
    """Retrieves data from a NIPA table."""
    base_url = "https://apps.bea.gov/api/data/"
    url = f"{base_url}?UserID={bea_api_key}&method=GetData&DataSetName=NIPA&TableName={table_id}&LineNumber={line_number}&Frequency={frequency}&Year={year}&ResultFormat=JSON"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching NIPA data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding NIPA JSON: {e}")
        print(f"Response text: {response.text}")
        return None

# Get Fixed Asset Table Data
def get_fa_table_data(bea_api_key, table_id, line_number, frequency="A", year="X"):
    """Retrieves data from a Fixed Assets table."""
    base_url = "https://apps.bea.gov/api/data/"
    url = f"{base_url}?UserID={bea_api_key}&method=GetData&DataSetName=FixedAssets&TableName=FAAt{table_id}&LineNumber={line_number}&Frequency={frequency}&Year={year}&ResultFormat=JSON"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Fixed Assets data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding Fixed Assets JSON: {e}")
        print(f"Response text: {response.text}")
        return None

# Extract Data
def extract_data(data, line_number):
    """Extracts data for a specific line."""
    if not data or "BEAAPI" not in data or "Results" not in data["BEAAPI"] or "Data" not in data["BEAAPI"]["Results"]:
        return {}

    result = {}
    for item in data["BEAAPI"]["Results"]["Data"]:
        if item["LineNumber"] == str(line_number):
            data_value = item["DataValue"]
            if data_value:
                data_value = data_value.replace(",", "")
                try:
                    value = float(data_value)
                    # Convert to billions
                    result[item["TimePeriod"]] = value / 1000.0
                except ValueError:
                    result[item["TimePeriod"]] = 0
            else:
                result[item["TimePeriod"]] = 0
    return result

# Create NIPA Data Dictionary
nipa_tables_lines = {
        "gross_domestic_product": ("T10105", 1),
        "household_nonprofit_gva": ("T10305", 5),
        "nonprofit_gva": ("T10305", 7),
        "government_gva": ("T10305", 8),
        "compensation_employees_paid": ("T11000", 2),
        "real_nva_nfc": ("T11400", 17),
        "personal_income": ("T20100", 1),
        "proprietors_net_income": ("T20100", 9),
        "gov_social_benefits": ("T20100", 17),
        "gov_social_insurance_contributions": ("T20100", 25),
        "personal_consumption_expenditures": ("T20100", 29),
        "personal_consumption_financial_insurance": ("T20305", 20),
        "nonprofit_sales_goods_services": ("T20305", 24),
        "gov_intermediate_goods_services": ("T31005", 6),
        "gov_sales_other_sectors": ("T31005", 11),
        "exports_other_business_services_a": ("T40205A", 22),
        "imports_other_business_services_a": ("T40205A", 46),
        "exports_other_business_services_b": ("T40205B", 78),
        "imports_other_business_services_b": ("T40205B", 168),
        "housing_output": ("T70405", 1),
        "net_proprietors_income_housing": ("T70405", 20),
        "corporate_profits_adj": ("T70405", 22),
        "consumption_fixed_capital_domestic_business": ("T70500", 3),
        "consumption_fixed_capital_nonfarm_housing": ("T70500", 13),
        "consumption_fixed_capital_farm_housing": ("T70500", 14),
        "depreciation_nonprofit_nonres_fixed_assets": ("T70500", 20),
        "rental_income_households_nonprofits": ("T70900", 8)
    }

nipa_data_dict = {}
for name, (table, line) in nipa_tables_lines.items():
    data = get_nipa_table_data(bea_api_key, table, line)
    nipa_data_dict[name] = extract_data(data, line)

nipa_df_components = pd.DataFrame(nipa_data_dict)

# Check NIPA Data Field by Field
#for column in nipa_df_components.columns:
#    print(f"\nColumn: {column}")
#    print(nipa_df_components[[column]].dropna().to_string(index=True))  # Print each column separately


# Merge Exports and Imports data
exports_other_business_services = {**nipa_data_dict["exports_other_business_services_a"], **nipa_data_dict["exports_other_business_services_b"]}
imports_other_business_services = {**nipa_data_dict["imports_other_business_services_a"], **nipa_data_dict["imports_other_business_services_b"]}

# Create FA Data Dictionary
fa_tables_lines = {
       "current_cost_depreciation_gov_nonres_fixed_assets": ("103", 10),
        "corporate_nonres_fixed_assets_current_cost": ("401", 17),
        "corporate_intellectual_property_products": ("401", 20),
        "current_cost_depreciation_noncorporate_private_fixed_assets": ("604", 5),
        "gross_investment_noncorporate_private_fixed_assets": ("607", 5)
    }

fa_data_dict = {}
for name, (table, line) in fa_tables_lines.items():
    data = get_fa_table_data(bea_api_key, table, line)
    fa_data_dict[name] = extract_data(data, line)

fa_df_components = pd.DataFrame(fa_data_dict)

# Check FA Data Field by Field
#for column in fa_df_components.columns:
#    print(f"\nColumn: {column}")
#    print(fa_df_components[[column]].dropna().to_string(index=True))  # Print each column separately

# Line Up Years
all_years = sorted(set().union(*[set(d.keys()) for d in nipa_data_dict.values()])) # You can grab the years from either nipa_data_dict or fa_data_dict

# Calculate Gross Revised Output (GRO)
gro_data = {}
for year in all_years:
    gro = (
        nipa_data_dict["gross_domestic_product"].get(year, 0)
        - nipa_data_dict["government_gva"].get(year, 0)
        + nipa_data_dict["gov_sales_other_sectors"].get(year, 0)
        - nipa_data_dict["household_nonprofit_gva"].get(year, 0)
        + nipa_data_dict["nonprofit_sales_goods_services"].get(year, 0)
        - nipa_data_dict["personal_consumption_financial_insurance"].get(year, 0)
        - exports_other_business_services.get(year, 0)
        + imports_other_business_services.get(year, 0)
        - nipa_data_dict["net_proprietors_income_housing"].get(year, 0)
        - nipa_data_dict["net_proprietors_income_housing"].get(year, 0)
        - (nipa_data_dict["consumption_fixed_capital_nonfarm_housing"].get(year, 0)
        + nipa_data_dict["consumption_fixed_capital_farm_housing"].get(year, 0))
    )
    gro_data[year] = gro

# Main Execution of GRO
if gro_data:
    df = pd.DataFrame(list(gro_data.items()), columns=["Year", "GRO (Billions)"])
    #print(df)
else:
    print("No data retrieved.")

# Calculate Net Revised Output (NRO)
nro_data = {}
for year in all_years:
    nro = (
        gro_data[year]
        - (nipa_data_dict["consumption_fixed_capital_domestic_business"].get(year, 0)
        - nipa_data_dict["consumption_fixed_capital_nonfarm_housing"].get(year, 0)
        - nipa_data_dict["consumption_fixed_capital_farm_housing"].get(year, 0))
        - (nipa_data_dict["gov_sales_other_sectors"].get(year, 0)
        / nipa_data_dict["government_gva"].get(year, 0))
        * fa_data_dict["current_cost_depreciation_gov_nonres_fixed_assets"].get(year, 0)
        - (nipa_data_dict["nonprofit_sales_goods_services"].get(year, 0)
        / nipa_data_dict["nonprofit_gva"].get(year, 0))
        * nipa_data_dict["depreciation_nonprofit_nonres_fixed_assets"].get(year,0)
    )
    nro_data[year] = nro

# Main Execution of NRO
if nro_data:
    df = pd.DataFrame(list(nro_data.items()), columns=["Year", "NRO (Billions)"])
    #print(df)
else:
    print("No data retrieved.")

# Calculate Average Propensity to Spend (APS)
aps_data = {}
for year in all_years:
    aps = (
        (nipa_data_dict["personal_consumption_expenditures"].get(year, 0)
        - nipa_data_dict["housing_output"].get(year, 0)
        - nipa_data_dict["personal_consumption_financial_insurance"].get(year, 0))
        / (nipa_data_dict["personal_income"].get(year, 0)
        - nipa_data_dict["rental_income_households_nonprofits"].get(year, 0))
    )
    aps_data[year] = aps

# Main Execution of APS
if aps_data:
    df = pd.DataFrame(list(aps_data.items()), columns=["Year", "APS (%)"])
    #print(df)
else:
    print("No data retrieved.")

# Calculate Corporate Surplus Value
csv_data = {}
for year in all_years:
    csv = (
        nro_data[year]
        - ((nipa_data_dict["compensation_employees_paid"].get(year, 0)
        + nipa_data_dict["proprietors_net_income"].get(year, 0))
        * aps_data[year])
        - (nipa_data_dict["gov_social_benefits"].get(year, 0)
        - nipa_data_dict["gov_social_insurance_contributions"].get(year, 0))
        - nipa_data_dict["gov_intermediate_goods_services"].get(year, 0)
        - (fa_data_dict["gross_investment_noncorporate_private_fixed_assets"].get(year, 0)
        - fa_data_dict["current_cost_depreciation_noncorporate_private_fixed_assets"].get(year, 0))
    )
    csv_data[year] = csv

# Main Execution of CSV
if csv_data:
    df = pd.DataFrame(list(csv_data.items()), columns=["Year", "CSV (Billions)"])
    #print(df)
else:
    print("No data retrieved.")

# Calculate Tangible Corporate Non-Residential Fixed Assets at Current Cost
tangible_corp_nonres_fixed_assets_current_cost_data = {}
for year in all_years:
    tangible_corp_nonres_fixed_assets_current_cost = (
            fa_data_dict["corporate_nonres_fixed_assets_current_cost"].get(year, 0)
            - fa_data_dict["corporate_intellectual_property_products"].get(year, 0)
    )

    # Only include years where the value is nonzero
    if tangible_corp_nonres_fixed_assets_current_cost != 0:
        tangible_corp_nonres_fixed_assets_current_cost_data[year] = tangible_corp_nonres_fixed_assets_current_cost

# Create DataFrame only if there's valid data
if tangible_corp_nonres_fixed_assets_current_cost_data:
    df = pd.DataFrame(
        list(tangible_corp_nonres_fixed_assets_current_cost_data.items()),
        columns=["Year", "Tangible Corporate Non-Residential Fixed Assets at Current Cost (Billions)"]
    )
    #print(df)
else:
    print("No data retrieved.")

# Calculate Corporate Profit Rate (CPR)
cpr_data = {}
for year in all_years:
    if year in csv_data and year in tangible_corp_nonres_fixed_assets_current_cost_data:  # Ensure year exists in both datasets
        cpr = (
            csv_data[year] / tangible_corp_nonres_fixed_assets_current_cost_data[year]
        )
        cpr_data[year] = cpr

# Create CPR DataFrame
if cpr_data:
    cpr_df = pd.DataFrame(list(cpr_data.items()), columns=["Year", "CPR (%)"])
    cpr_df['Year'] = pd.to_datetime(cpr_df['Year'])  # Convert year to datetime
    cpr_df = cpr_df.set_index('Year')  # Set year as index

# Save CPR DataFrame to a CSV file
cpr_df.to_csv("CPR_Data.csv", index=False)

# Calculate Growth Rate of Real Net Value Added of Nonfinancial Corporate Business
real_nva_nfc = nipa_data_dict["real_nva_nfc"]
gr_real_nva_nfc = {}
for i in range(1, len(all_years)):  # Start from index 1 to prevent negative indexing
    current_year = all_years[i]
    previous_year = all_years[i - 1]

    # Ensure both current and previous year exist in real_nva_nfc before calculation
    if current_year in real_nva_nfc and previous_year in real_nva_nfc:
        gr_real_nva_nfc[current_year] = (real_nva_nfc[current_year] - real_nva_nfc[previous_year]) / real_nva_nfc[
            previous_year]
    else:
        print(f"Skipping growth rate calculation for {current_year} due to missing data.")

# Create real_nva_nfc DataFrame
if gr_real_nva_nfc:
    gr_real_nva_nfc_df = pd.DataFrame(list(gr_real_nva_nfc.items()), columns=["Year", "real_nva_nfc (%)"])
    gr_real_nva_nfc_df['Year'] = pd.to_datetime(gr_real_nva_nfc_df['Year'])
    gr_real_nva_nfc_df = gr_real_nva_nfc_df.set_index('Year')

# Save Growth Rate of Real NVA NFC DataFrame to a CSV file
gr_real_nva_nfc_df.to_csv("Growth_Rate_Real_NVA_NFC.csv", index=False)

# Apply HP Filter to CPR
cpr_cycle, cpr_trend = hpfilter(cpr_df['CPR (%)'], lamb=100)  # Common lambda value for quarterly data, adjust for your frequency
cpr_df['CPR Trend'] = cpr_trend

# Apply HP Filter to Real NVA NFC Growth Rate
gr_real_nva_nfc_cycle, gr_real_nva_nfc_trend = hpfilter(gr_real_nva_nfc_df['real_nva_nfc (%)'], lamb=100)  # Adjust lambda as needed
gr_real_nva_nfc_df['real_nva_nfc Trend'] = gr_real_nva_nfc_trend

# Create the plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot CPR and its Trend on the left y-axis
ax1.plot(cpr_df.index, cpr_df['CPR (%)'], label="CPR", color='blue', alpha=0.6)
ax1.plot(cpr_df.index, cpr_df['CPR Trend'], label="CPR Trend (HP Filter)", color='blue', linestyle='--')

# Label left y-axis
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("CPR (%)", fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for Growth Rate of Real NVA NFC and its trend
ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
ax2.plot(gr_real_nva_nfc_df.index, gr_real_nva_nfc_df['real_nva_nfc (%)'], label="Growth Rate of Real NVA NFC", color='green', alpha=0.6)
ax2.plot(gr_real_nva_nfc_df.index, gr_real_nva_nfc_df['real_nva_nfc Trend'], label="Growth Rate Trend (HP Filter)", color='orange', linestyle='--')

# Label right y-axis
ax2.set_ylabel("Growth Rate (%)", fontsize=12)
ax2.tick_params(axis='y', labelcolor='green')

# Set title
plt.title("CPR and Growth Rate of Real NVA NFC with HP Filter Trends", fontsize=14)

# Display the legend
fig.tight_layout()  # Automatically adjusts plot to avoid overlap
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Enable grid for better readability
ax1.grid(True)

# Rotate x-axis labels if necessary
plt.xticks(rotation=45)

# Show the plot
plt.show()

# Combine CPR DataFrame with GR Real NVA NFC DataFrame
combined_df = pd.merge(cpr_df, gr_real_nva_nfc_df, left_index=True, right_index=True, how='inner')

# Create Lagged Versions of CPR
max_lag = 10  # Set the maximum number of lags you want to test
lagged_cpr_df = pd.DataFrame()

# Create lagged CPR columns for each lag
for lag in range(1, max_lag + 1):
    lagged_cpr_df[f"CPR_Lag_{lag}"] = combined_df['CPR (%)'].shift(lag)

# Add 'real_nva_nfc (%)' to the lagged_cpr_df
lagged_cpr_df['real_nva_nfc (%)'] = combined_df['real_nva_nfc (%)']

# Remove NaNs from lagged CPR columns and real_nva_nfc columns
lagged_cpr_df_clean = lagged_cpr_df.dropna(subset=['CPR_Lag_1', 'CPR_Lag_2', 'CPR_Lag_3', 'CPR_Lag_4', 'CPR_Lag_5', 'real_nva_nfc (%)'])

# Calculate the Correlation Between Lagged CPR and Growth Rate of Real NVA NFC
correlation_results = {}
for lag in range(1, max_lag + 1):
    correlation = lagged_cpr_df_clean[f"CPR_Lag_{lag}"].corr(lagged_cpr_df_clean['real_nva_nfc (%)'])
    correlation_results[f"CPR_Lag_{lag}"] = correlation

# Print the correlation results
print("Lagged Correlation Results:")
for lag, corr_value in correlation_results.items():
    print(f"{lag}-Year Lag: {corr_value:.4f}")

# Visualize the Lagged Correlation Results
sns.barplot(x=list(correlation_results.keys()), y=list(correlation_results.values()))
plt.title('Lagged Correlation Between CPR and Growth Rate of Real NVA NFC')
plt.xlabel('Lag (Years)')
plt.ylabel('Correlation')
plt.show()

# Check if both series are stationary using the Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series)
    return result[1]  # p-value

# Check for stationarity of both CPR and real_nva_nfc
cpr_stationarity = check_stationarity(combined_df['CPR (%)'])
growth_stationarity = check_stationarity(combined_df['real_nva_nfc (%)'])

if cpr_stationarity < 0.05 and growth_stationarity < 0.05:
    print("Both series are stationary. Proceeding with Granger Causality test.")
else:
    print("One or both series are non-stationary. Differencing is required.")
    # If non-stationary, you can difference the series:
    # combined_df['CPR (%)'] = combined_df['CPR (%)'].diff().dropna()
    # combined_df['real_nva_nfc (%)'] = combined_df['real_nva_nfc (%)'].diff().dropna()

# Step 2: Granger Causality test (for a range of lags, e.g., 1-5 lags)
max_lag = 10  # or however many lags you want to test
gc_result = grangercausalitytests(combined_df[['CPR (%)', 'real_nva_nfc (%)']], max_lag, verbose=True)

# Calculate the lagged variables
combined_df['lagged_CPR'] = combined_df['CPR (%)'].shift(1)
combined_df['lagged_real_nva_nfc'] = combined_df['real_nva_nfc (%)'].shift(1)

# Drop missing values due to shifting
combined_df = combined_df.dropna()

# Define the dependent and independent variables
# The dependent variable is 'growth_rate_nva_nfc', and the independent variable is 'CPR'
X = combined_df[['CPR (%)', 'lagged_CPR']]
y = combined_df[['real_nva_nfc (%)']]

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit a TAR model
# We can model the relation between CPR and growth_rate_nva_nfc using OLS regression.
# The 'lagged_CPR' is used to capture the lagged effects of CPR on growth.

# Fit the model
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()

# Step 4: Display results
print(ols_results.summary())

# Perform the Breusch-Godfrey test with 1 lag (you can change the number of lags)
bg_test = acorr_breusch_godfrey(ols_results, nlags=1)

# Output the results
print(f"Breusch-Godfrey test statistic: {bg_test[0]}")
print(f"P-value: {bg_test[1]}")

# Refit the model with robust standard errors (HAC) and specify maxlags
ols_model_robust = ols_results.get_robustcov_results(cov_type='HAC', maxlags=1)

# Print the results
print(ols_model_robust.summary())