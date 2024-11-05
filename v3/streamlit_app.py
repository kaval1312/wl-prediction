import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, triang
import numpy_financial as npf
import warnings

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Well Abandonment Monte Carlo Simulation with ROI", layout="wide")

st.title("Monte Carlo Simulation for Well Abandonment Decision with ROI Analysis")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Number of simulation runs
num_simulations = st.sidebar.number_input(
    "Number of Simulations", min_value=100, max_value=50000, value=1000, step=100
)

# Simulation years
simulation_years = st.sidebar.number_input(
    "Simulation Duration (years)", min_value=1, max_value=50, value=20, step=1
)

# Distribution selection
st.sidebar.subheader("Parameter Distributions")

distribution_options = ["Normal", "Log-Normal", "Triangular"]

# Function to sample distributions
def sample_distribution(dist_name, params, size):
    if dist_name == "Normal":
        samples = np.random.normal(params['mean'], params['std'], size)
    elif dist_name == "Log-Normal":
        sigma = np.sqrt(np.log(1 + (params['std'] / params['mean']) ** 2))
        mu = np.log(params['mean']) - 0.5 * sigma ** 2
        samples = np.random.lognormal(mu, sigma, size)
    elif dist_name == "Triangular":
        samples = np.random.triangular(params['left'], params['mode'], params['right'], size)
    else:
        samples = np.random.normal(params['mean'], params['std'], size)
    return samples

# Initial Production Rate
st.sidebar.subheader("Initial Production Rate (barrels/day)")
initial_prod_dist = st.sidebar.selectbox("Distribution", distribution_options, key="ip_dist")
initial_prod_params = {}
if initial_prod_dist == "Triangular":
    initial_prod_params['left'] = st.sidebar.number_input("Minimum", value=800.0)
    initial_prod_params['mode'] = st.sidebar.number_input("Most Likely", value=1000.0)
    initial_prod_params['right'] = st.sidebar.number_input("Maximum", value=1200.0)
else:
    initial_prod_params['mean'] = st.sidebar.number_input("Mean", value=1000.0)
    initial_prod_params['std'] = st.sidebar.number_input("Standard Deviation", value=100.0)

# Decline Rate
st.sidebar.subheader("Decline Rate (% per year)")
decline_rate_dist = st.sidebar.selectbox("Distribution", distribution_options, key="dr_dist")
decline_rate_params = {}
if decline_rate_dist == "Triangular":
    decline_rate_params['left'] = st.sidebar.number_input("Minimum", value=5.0)
    decline_rate_params['mode'] = st.sidebar.number_input("Most Likely", value=10.0)
    decline_rate_params['right'] = st.sidebar.number_input("Maximum", value=15.0)
else:
    decline_rate_params['mean'] = st.sidebar.number_input("Mean (%)", value=10.0)
    decline_rate_params['std'] = st.sidebar.number_input("Standard Deviation (%)", value=2.0)

# Decline Model
st.sidebar.subheader("Production Decline Model")
decline_model = st.sidebar.selectbox("Select Decline Model", ["Exponential", "Hyperbolic"])

# Oil Price
st.sidebar.subheader("Oil Price (USD/barrel)")
oil_price_dist = st.sidebar.selectbox("Distribution", distribution_options, key="op_dist")
oil_price_params = {}
if oil_price_dist == "Triangular":
    oil_price_params['left'] = st.sidebar.number_input("Minimum", value=50.0)
    oil_price_params['mode'] = st.sidebar.number_input("Most Likely", value=70.0)
    oil_price_params['right'] = st.sidebar.number_input("Maximum", value=90.0)
else:
    oil_price_params['mean'] = st.sidebar.number_input("Mean", value=70.0)
    oil_price_params['std'] = st.sidebar.number_input("Standard Deviation", value=10.0)

# Operating Costs
st.sidebar.subheader("Operating Costs (USD/barrel)")
operating_cost_dist = st.sidebar.selectbox("Distribution", distribution_options, key="oc_dist")
operating_cost_params = {}
if operating_cost_dist == "Triangular":
    operating_cost_params['left'] = st.sidebar.number_input("Minimum", value=20.0)
    operating_cost_params['mode'] = st.sidebar.number_input("Most Likely", value=30.0)
    operating_cost_params['right'] = st.sidebar.number_input("Maximum", value=40.0)
else:
    operating_cost_params['mean'] = st.sidebar.number_input("Mean", value=30.0)
    operating_cost_params['std'] = st.sidebar.number_input("Standard Deviation", value=5.0)

# Operating Cost Escalation Rate
st.sidebar.subheader("Operating Cost Escalation Rate (% per year)")
op_cost_esc_rate = st.sidebar.number_input("Escalation Rate (%)", value=2.0) / 100

# Oil Price Escalation Rate
st.sidebar.subheader("Oil Price Escalation Rate (% per year)")
oil_price_esc_rate = st.sidebar.number_input("Escalation Rate (%)", value=1.0) / 100

# Maintenance Costs
st.sidebar.subheader("Annual Maintenance Costs (USD/year)")
maintenance_cost = st.sidebar.number_input("Maintenance Cost", value=50000.0)

# Capital Expenditures (CapEx)
st.sidebar.subheader("Capital Expenditures (USD)")
capex = st.sidebar.number_input("Initial CapEx", value=1000000.0)
capex_schedule = st.sidebar.text_input(
    "Additional CapEx Schedule (year:amount, comma-separated)", "5:200000,10:200000"
)

# Abandonment Cost
st.sidebar.subheader("Abandonment Cost (USD)")
abandonment_cost_dist = st.sidebar.selectbox("Distribution", distribution_options, key="ac_dist")
abandonment_cost_params = {}
if abandonment_cost_dist == "Triangular":
    abandonment_cost_params['left'] = st.sidebar.number_input("Minimum", value=400000.0)
    abandonment_cost_params['mode'] = st.sidebar.number_input("Most Likely", value=500000.0)
    abandonment_cost_params['right'] = st.sidebar.number_input("Maximum", value=600000.0)
else:
    abandonment_cost_params['mean'] = st.sidebar.number_input("Mean", value=500000.0)
    abandonment_cost_params['std'] = st.sidebar.number_input("Standard Deviation", value=50000.0)

# Discount Rate
st.sidebar.subheader("Discount Rate (% per year)")
discount_rate_dist = st.sidebar.selectbox("Distribution", distribution_options, key="drate_dist")
discount_rate_params = {}
if discount_rate_dist == "Triangular":
    discount_rate_params['left'] = st.sidebar.number_input("Minimum", value=6.0)
    discount_rate_params['mode'] = st.sidebar.number_input("Most Likely", value=8.0)
    discount_rate_params['right'] = st.sidebar.number_input("Maximum", value=10.0)
else:
    discount_rate_params['mean'] = st.sidebar.number_input("Mean (%)", value=8.0)
    discount_rate_params['std'] = st.sidebar.number_input("Standard Deviation (%)", value=1.0)

# Tax Rate
st.sidebar.subheader("Corporate Tax Rate (%)")
tax_rate = st.sidebar.number_input("Tax Rate (%)", value=30.0) / 100

# Royalties
st.sidebar.subheader("Royalties (% of revenue)")
royalty_rate = st.sidebar.number_input("Royalty Rate (%)", value=12.5) / 100

# Inflation Rate
st.sidebar.subheader("Inflation Rate (% per year)")
inflation_rate = st.sidebar.number_input("Inflation Rate (%)", value=2.0) / 100

# Economic Limit
st.sidebar.subheader("Economic Limit (USD/year)")
economic_limit = st.sidebar.number_input("Economic Limit (minimum net profit per year)", value=0.0)

# Correlations
st.sidebar.subheader("Parameter Correlations")
apply_correlation = st.sidebar.checkbox("Apply Correlation Between Oil Price and Operating Costs")

# Financing Options
st.sidebar.header("Financing Options")
st.sidebar.subheader("Debt Financing")
use_debt = st.sidebar.checkbox("Use Debt Financing?")
if use_debt:
    debt_ratio = st.sidebar.number_input("Debt Ratio (% of CapEx)", value=50.0) / 100
    interest_rate = st.sidebar.number_input("Interest Rate on Debt (%)", value=5.0) / 100
    loan_term = st.sidebar.number_input("Loan Term (years)", value=10, min_value=1, max_value=30)
else:
    debt_ratio = 0
    interest_rate = 0
    loan_term = 0

# Depreciation
st.sidebar.header("Depreciation")
use_depreciation = st.sidebar.checkbox("Apply Depreciation?")
if use_depreciation:
    depreciation_years = st.sidebar.number_input("Depreciation Period (years)", value=10, min_value=1, max_value=30)
else:
    depreciation_years = 0

# Run simulation button
run_simulation = st.sidebar.button("Run Simulation")

if run_simulation:
    st.header("Simulation Results")

    try:
        # Generate random samples for each parameter
        initial_productions = sample_distribution(initial_prod_dist, initial_prod_params, num_simulations)
        decline_rates = sample_distribution(decline_rate_dist, decline_rate_params, num_simulations) / 100
        oil_prices = sample_distribution(oil_price_dist, oil_price_params, num_simulations)
        operating_costs = sample_distribution(operating_cost_dist, operating_cost_params, num_simulations)
        abandonment_costs = sample_distribution(abandonment_cost_dist, abandonment_cost_params, num_simulations)
        discount_rates = sample_distribution(discount_rate_dist, discount_rate_params, num_simulations) / 100

        # Apply correlation if selected
        if apply_correlation:
            # Rank correlation
            oil_price_ranks = oil_prices.argsort().argsort()
            operating_costs = operating_costs[oil_price_ranks]

        # CapEx schedule parsing
        capex_schedule_dict = {}
        if capex_schedule:
            capex_items = capex_schedule.split(',')
            for item in capex_items:
                year_amount = item.strip().split(':')
                if len(year_amount) == 2:
                    year, amount = year_amount
                    capex_schedule_dict[int(year)] = float(amount)
                else:
                    st.warning(f"Invalid CapEx schedule entry: '{item}'. Please use the format 'year:amount'.")
                    continue

        # Initialize arrays to store results
        abandonment_times = np.zeros(num_simulations)
        npvs = np.zeros(num_simulations)
        irr_values = np.zeros(num_simulations)
        payback_periods = np.zeros(num_simulations)
        profitability_indexes = np.zeros(num_simulations)

        # Simulation loop
        for i in range(num_simulations):
            initial_prod = initial_productions[i]
            decline_rate = decline_rates[i]
            oil_price = oil_prices[i]
            operating_cost = operating_costs[i]
            abandonment_cost = abandonment_costs[i]
            discount_rate = discount_rates[i]

            # Financing calculations
            equity_ratio = 1 - debt_ratio
            total_capex = capex + sum(capex_schedule_dict.values())
            debt_amount = total_capex * debt_ratio
            equity_amount = total_capex * equity_ratio
            annual_debt_payment = (
                npf.pmt(interest_rate, loan_term, -debt_amount) if use_debt else 0
            )

            # Depreciation calculations
            annual_depreciation = total_capex / depreciation_years if use_depreciation else 0

            cash_flows = []
            cumulative_cash_flow = -equity_amount  # Initial investment from equity
            production = initial_prod

            # Yearly simulation
            for year in range(1, simulation_years + 1):
                # Production decline
                if decline_model == "Exponential":
                    production = initial_prod * np.exp(-decline_rate * (year - 1))
                elif decline_model == "Hyperbolic":
                    b_factor = 0.5  # Could be an input parameter
                    production = initial_prod / ((1 + b_factor * decline_rate * (year - 1)) ** (1 / b_factor))

                # Escalate oil price and operating cost
                oil_price *= (1 + oil_price_esc_rate)
                operating_cost *= (1 + op_cost_esc_rate)

                # Revenue and costs
                revenue = production * 365 * oil_price
                royalty = revenue * royalty_rate
                cost = production * 365 * operating_cost + maintenance_cost
                capex_year = capex_schedule_dict.get(year, 0)
                depreciation = annual_depreciation if year <= depreciation_years else 0
                taxable_income = revenue - royalty - cost - depreciation - capex_year
                tax = taxable_income * tax_rate if taxable_income > 0 else 0
                net_cash_flow = revenue - royalty - cost - tax - capex_year

                # Financing cash flows
                if use_debt and year <= loan_term:
                    net_cash_flow -= annual_debt_payment

                cash_flows.append(net_cash_flow)
                cumulative_cash_flow += net_cash_flow / ((1 + discount_rate) ** year)

                if cumulative_cash_flow >= 0 and payback_periods[i] == 0:
                    payback_periods[i] = year

                if net_cash_flow < economic_limit:
                    # Account for abandonment cost
                    cumulative_cash_flow -= abandonment_cost / ((1 + discount_rate) ** year)
                    cash_flows[-1] -= abandonment_cost
                    abandonment_times[i] = year
                    npvs[i] = cumulative_cash_flow
                    break
            else:
                # If not abandoned within simulation years
                abandonment_times[i] = simulation_years
                cumulative_cash_flow -= abandonment_cost / ((1 + discount_rate) ** simulation_years)
                cash_flows[-1] -= abandonment_cost
                npvs[i] = cumulative_cash_flow

            # Calculate IRR and Profitability Index
            cash_flow_series = [-equity_amount] + cash_flows
            irr_values[i] = npf.irr(cash_flow_series)
            total_pv_of_cash_flows = npf.npv(discount_rate, cash_flow_series[1:])
            profitability_indexes[i] = total_pv_of_cash_flows / equity_amount

            if payback_periods[i] == 0:
                payback_periods[i] = simulation_years

        # Display results
        st.subheader("Abandonment Time Distribution")

        fig, ax = plt.subplots()
        ax.hist(abandonment_times, bins=range(1, simulation_years + 2), edgecolor='black')
        ax.set_xlabel('Abandonment Time (years)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        st.subheader("ROI Metrics Distribution")

        fig2, axs = plt.subplots(2, 2, figsize=(12, 8))

        axs[0, 0].hist(npvs, bins=50, edgecolor='black')
        axs[0, 0].set_xlabel('NPV (USD)')
        axs[0, 0].set_ylabel('Frequency')
        axs[0, 0].set_title('NPV Distribution')

        axs[0, 1].hist(irr_values * 100, bins=50, edgecolor='black')
        axs[0, 1].set_xlabel('IRR (%)')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('IRR Distribution')

        axs[1, 0].hist(payback_periods, bins=range(1, simulation_years + 2), edgecolor='black')
        axs[1, 0].set_xlabel('Payback Period (years)')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].set_title('Payback Period Distribution')

        axs[1, 1].hist(profitability_indexes, bins=50, edgecolor='black')
        axs[1, 1].set_xlabel('Profitability Index')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].set_title('Profitability Index Distribution')

        plt.tight_layout()
        st.pyplot(fig2)

        st.subheader("Statistical Summary")
        summary_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Abandonment Time (years)': [
                np.mean(abandonment_times),
                np.median(abandonment_times),
                np.std(abandonment_times),
                np.min(abandonment_times),
                np.max(abandonment_times)
            ],
            'NPV (USD)': [
                np.mean(npvs),
                np.median(npvs),
                np.std(npvs),
                np.min(npvs),
                np.max(npvs)
            ],
            'IRR (%)': [
                np.mean(irr_values) * 100,
                np.median(irr_values) * 100,
                np.std(irr_values) * 100,
                np.min(irr_values) * 100,
                np.max(irr_values) * 100
            ],
            'Payback Period (years)': [
                np.mean(payback_periods),
                np.median(payback_periods),
                np.std(payback_periods),
                np.min(payback_periods),
                np.max(payback_periods)
            ],
            'Profitability Index': [
                np.mean(profitability_indexes),
                np.median(profitability_indexes),
                np.std(profitability_indexes),
                np.min(profitability_indexes),
                np.max(profitability_indexes)
            ]
        })
        st.table(summary_df)

        st.subheader("Detailed Simulation Data")
        detailed_df = pd.DataFrame({
            'Simulation Run': np.arange(1, num_simulations + 1),
            'Abandonment Time (years)': abandonment_times,
            'NPV (USD)': npvs,
            'IRR (%)': irr_values * 100,
            'Payback Period (years)': payback_periods,
            'Profitability Index': profitability_indexes
        })
        st.dataframe(detailed_df)

    except Exception as e:
        st.error(f"An error occurred during the simulation: {e}")
